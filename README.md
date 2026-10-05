# Machine Learning Based Adaptive Transmission Rate Selection in IEEE 802.11 Networks

This project implements adaptive IEEE 802.11 transmission-rate selection using machine learning. It reads live wireless measurements such as RSSI, SNR, packet loss, latency, throughput, channel utilization, and current rate, then predicts the best transmission rate for the current network conditions. The main dataset used in this project is `data/WiFi_Transmission_Rate_Recommendation_Dataset_5000.xlsx`.

## Setup and Installation
```bash
pip install -r requirements.txt
```

## How to Train
To run the training pipeline and save the best model:
```bash
python src/train.py
```

Training uses the new Excel dataset and is compatible with the legacy CSV files as well. It removes duplicates, normalizes the fields, and uses the real IEEE 802.11 measurements from the dataset: RSSI, SNR, packet loss, latency, throughput, channel utilization, and current transmission rate. Training saves `models/best_model.pkl`, `models/model_metadata.json`, and evaluation results in `results/metrics/`, including model comparison, classification report, and confusion matrix.

The generated `models/best_model.pkl` file is stored with Git LFS because it is larger than GitHub's standard 100 MB file limit. If LFS is unavailable, recreate it after cloning with `python src/train.py`.

## How to Launch the Scan-and-Analyze GUI
```bash
python main.py
```

Click **Scan Nearby Wi-Fi**, select a network, enter connected-network measurements, and click **Analyze Selected Network**. The GUI records successful predictions in `results/predictions/prediction_history.csv`.

## How to Launch Web App
```bash
streamlit run app/web_app.py
```

Then open `http://localhost:8501`.

The dashboard scans for nearby networks on startup and automatically refreshes the scan every 30 seconds by default. Use **Scan now** for an immediate refresh or turn off **Auto scan** to stop periodic scans. The dashboard also shows the model's test accuracy, macro F1-score, feature importance, rate adjustment from the current setting, and a warning when the link conditions are weak.

## Website policy and local data handling

- The dashboard is designed for local, on-device wireless analysis. It reads nearby Wi-Fi metadata and manually supplied network measurements to estimate link health and recommend a transmission rate.
- The app does not upload scan data to a remote backend during normal prediction use, and it does not use third-party analytics or tracking scripts in the default workflow.
- Prediction history and exported CSV files are stored locally in the project unless the user explicitly downloads or shares them.
- All recommendations are advisory only. The project does not change the adapter's configured rate, connect to protected networks, or force any hardware changes.

## Wi-Fi scanner behavior

- Windows uses `netsh wlan show networks mode=bssid`.
- Linux uses `nmcli`; NetworkManager must be installed.
- The web app runs the scanner on the machine hosting Streamlit. To see networks around your device, run the app locally on that device; a remotely hosted website can only scan the server's Wi-Fi environment.
- A scan can normally expose SSID, BSSID, signal, channel, and security, but drivers may omit some of these fields.
- Packet loss, latency, throughput, SNR, channel utilization, and current TX rate are connected-network measurements. The GUI leaves them blank until the user supplies real measurements.
- The application recommends a rate only. It does not change the adapter's hardware rate or connect to protected networks.

## Dataset and target limitation

The uploaded datasets contain MCS TX values from 1 through 11 and do not provide a valid mapping to 6, 12, 24, 36, 48, and 54 Mbps. The application therefore predicts MCS, not a fabricated Mbps value. Channel utilization is also not present and is not invented. The GUI never uses fabricated scan values: unavailable OS values remain unavailable, and missing prediction inputs cause a validation message.

## Troubleshooting

- If no networks are shown, enable the Wi-Fi adapter and run the command for your operating system manually.
- If Linux reports that `nmcli` is unavailable, install and start NetworkManager.
- If the model is missing, run `python src/train.py` from the project root.
- If the GUI cannot predict, enter all six connected-network measurements.
