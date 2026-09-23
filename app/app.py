import csv
import os
import sys
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.wifi_scanner import scan_wifi_networks

FEATURES = [
    ("RSSI (dBm)", "rssi"),
    ("SNR (dB)", "snr"),
    ("Packet loss (%)", "packet_loss"),
    ("Latency (ms)", "latency"),
    ("Throughput (Mbps)", "throughput"),
    ("Channel utilization (%)", "channel_utilization"),
    ("Current TX rate (Mbps)", "current_transmission_rate"),
]


class TransmissionRateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Based Adaptive Transmission Rate Selection in IEEE 802.11 Network")
        self.root.geometry("980x720")
        self.root.minsize(850, 620)
        self.networks = []
        try:
            self.pipeline = joblib.load(os.path.join(ROOT, "models", "best_model.pkl"))
        except Exception as error:
            self.pipeline = None
            messagebox.showwarning("Model unavailable", f"Run python src/train.py first.\n\n{error}")
        self.build_ui()

    def build_ui(self):
        header = tk.Frame(self.root, bg="#17324d", padx=18, pady=14)
        header.pack(fill=tk.X)
        tk.Label(header, text="Adaptive IEEE 802.11 Transmission Rate Selection", fg="white", bg="#17324d", font=("Segoe UI", 18, "bold")).pack(anchor=tk.W)
        tk.Label(header, text="Scan real networks, select one, then predict the best rate from live wireless conditions.", fg="#d7e6f2", bg="#17324d", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(4, 0))

        controls = tk.Frame(self.root, padx=18, pady=12)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Scan Nearby Wi-Fi", command=self.scan).pack(side=tk.LEFT)
        self.scan_status = tk.Label(controls, text="Not scanned", fg="#555")
        self.scan_status.pack(side=tk.LEFT, padx=12)

        table_frame = ttk.LabelFrame(self.root, text="Available Networks", padding=8)
        table_frame.pack(fill=tk.X, padx=18)
        columns = ("ssid", "bssid", "signal", "channel", "security")
        self.network_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        for column, title, width in [("ssid", "SSID", 220), ("bssid", "BSSID", 170), ("signal", "Signal", 90), ("channel", "Channel", 80), ("security", "Security", 160)]:
            self.network_table.heading(column, text=title)
            self.network_table.column(column, width=width)
        self.network_table.pack(fill=tk.X)
        self.network_table.bind("<<TreeviewSelect>>", self.select_network)

        body = tk.Frame(self.root, padx=18, pady=14)
        body.pack(fill=tk.BOTH, expand=True)
        input_frame = ttk.LabelFrame(body, text="Connected-network measurements", padding=10)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        tk.Label(input_frame, text="Scan data is filled automatically. Enter the live wireless metrics from the selected access point.", wraplength=390, justify=tk.LEFT, fg="#555").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        self.entries = {}
        for row, (label, key) in enumerate(FEATURES, start=1):
            tk.Label(input_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
            entry = ttk.Entry(input_frame, width=24)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=4, padx=(10, 0))
            self.entries[key] = entry
        input_frame.columnconfigure(1, weight=1)
        ttk.Button(input_frame, text="Analyze Selected Network", command=self.analyze).grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=(14, 4))

        result_frame = ttk.LabelFrame(body, text="AI recommendation", padding=14)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.selected_label = tk.Label(result_frame, text="Selected network: none", font=("Segoe UI", 11, "bold"), anchor=tk.W, justify=tk.LEFT)
        self.selected_label.pack(fill=tk.X, pady=(0, 15))
        self.result_label = tk.Label(result_frame, text="Scan and select a network", font=("Segoe UI", 18, "bold"), fg="#176b87", wraplength=330)
        self.result_label.pack(pady=15)
        self.condition_label = tk.Label(result_frame, text="", font=("Segoe UI", 11), justify=tk.LEFT, wraplength=330)
        self.condition_label.pack(pady=8)
        self.info_label = tk.Label(result_frame, text="", justify=tk.LEFT, anchor=tk.W, wraplength=330)
        self.info_label.pack(fill=tk.X, pady=8)

    def scan(self):
        try:
            self.networks = scan_wifi_networks()
            for item in self.network_table.get_children():
                self.network_table.delete(item)
            for index, network in enumerate(self.networks):
                self.network_table.insert("", tk.END, iid=str(index), values=(network["ssid"], network["bssid"], network["signal"] if network["signal"] is not None else "Unavailable", network["channel"] or "Unavailable", network["security"]))
            self.scan_status.config(text=f"Found {len(self.networks)} network(s)")
        except Exception as error:
            self.scan_status.config(text="Scan failed")
            messagebox.showerror("Wi-Fi scan unavailable", str(error))

    def select_network(self, _event=None):
        selected = self.network_table.selection()
        if not selected:
            return
        network = self.networks[int(selected[0])]
        self.selected_label.config(text=f"Selected network: {network['ssid']}\nBSSID: {network['bssid']}\nChannel: {network['channel'] or 'Unavailable'} | Security: {network['security']}")
        if network["signal"] is not None:
            self.entries["rssi"].delete(0, tk.END)
            self.entries["rssi"].insert(0, str(network["signal"]))

    def analyze(self):
        selected = self.network_table.selection()
        if not selected:
            messagebox.showwarning("Select a network", "Scan and select a Wi-Fi network first.")
            return
        if not self.pipeline:
            messagebox.showerror("Model unavailable", "Run python src/train.py first.")
            return
        try:
            values = {key: float(self.entries[key].get()) for _, key in FEATURES}
            prediction_frame = pd.DataFrame([values])
            prediction = int(self.pipeline.predict(prediction_frame)[0])
            confidence = float(max(self.pipeline.predict_proba(prediction_frame)[0]) * 100)
            network = self.networks[int(selected[0])]
            condition = "Good" if values["packet_loss"] < 5 and values["snr"] >= 20 and values["channel_utilization"] < 60 else "Moderate" if values["packet_loss"] < 15 else "Weak"
            self.result_label.config(text=f"Recommended transmission rate\n{prediction} Mbps", fg="#18794e")
            self.condition_label.config(text=f"Network condition: {condition}\nConfidence: {confidence:.2f}%")
            self.info_label.config(text="Prediction uses the live IEEE 802.11 measurements from the adaptive rate-selection dataset. It recommends the best transmission rate and does not change the Wi-Fi adapter settings.")
            self.save_history(network, values, prediction, confidence)
        except ValueError:
            messagebox.showerror("Measurements required", "Enter real measurements for every field before analyzing.")
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error))

    def save_history(self, network, values, prediction, confidence):
        path = os.path.join(ROOT, "results", "predictions", "prediction_history.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "ssid": network["ssid"], "bssid": network["bssid"], "signal": network["signal"], "channel": network["channel"], **values, "predicted_rate_mbps": prediction, "confidence": round(confidence, 2)}
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def main():
    root = tk.Tk()
    TransmissionRateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
