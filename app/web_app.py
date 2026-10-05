import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import build_prediction_features
from src.wifi_scanner import scan_wifi_networks


st.set_page_config(
    page_title="Adaptive Wi-Fi Rate Intelligence",
    page_icon=":material/network_check:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #061b2b 0%, #0f2d3d 38%, #122b4d 100%); }
    .main .block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
    .stMetric { background: rgba(14, 42, 60, .9); border: 1px solid rgba(110, 188, 255, .2); border-radius: 16px; padding: .8rem 1rem; }
    .stAlert, div[data-testid="stDataFrame"] { border-radius: 16px; }
    h1, h2, h3 { color: #f1f7ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_LABELS = {
    "rssi": "RSSI",
    "snr": "SNR",
    "packet_loss": "Packet loss",
    "latency": "Latency",
    "throughput": "Throughput",
    "channel_utilization": "Channel utilization",
    "current_transmission_rate": "Current rate",
}
AUTO_SCAN_INTERVAL_SECONDS = 30


def model_paths():
    return ROOT / "models" / "best_model.pkl", ROOT / "models" / "model_metadata.json"


@st.cache_resource
def load_model_bundle(model_path: str, metadata_path: str, model_mtime: float):
    del model_mtime
    pipeline = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return pipeline, metadata


def load_bundle():
    model_path, metadata_path = model_paths()
    if not model_path.exists() or not metadata_path.exists():
        return None, {}, "The trained model or metadata file is missing."
    try:
        pipeline, metadata = load_model_bundle(
            str(model_path), str(metadata_path), model_path.stat().st_mtime
        )
        return pipeline, metadata, None
    except Exception as exc:
        return None, {}, f"Model loading failed: {exc}"


def network_label(network):
    signal = network.get("signal")
    signal_text = f"{signal:.0f} dBm" if signal is not None else "N/A"
    percent = network.get("signal_percent")
    percent_text = f" ({percent}% strength)" if percent is not None else ""
    bssid = network.get("bssid") or "Not reported"
    return f"{network.get('ssid') or '<hidden>'} | BSSID {bssid} | Signal {signal_text}{percent_text} | Ch {network.get('channel') or 'N/A'} | {network.get('security') or 'Unknown'}"


def scan_and_store():
    try:
        st.session_state.networks = scan_wifi_networks()
        st.session_state.scan_error = None
        st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
        st.session_state.last_scan_epoch = time.time()
    except Exception as exc:
        st.session_state.setdefault("networks", [])
        st.session_state.scan_error = str(exc)
        st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
        st.session_state.last_scan_epoch = time.time()


def network_health(rssi, snr, packet_loss, latency, throughput, utilization):
    signal_score = max(0, min(100, (rssi + 100) / 80 * 100))
    snr_score = max(0, min(100, snr / 35 * 100))
    loss_score = max(0, 100 - packet_loss)
    latency_score = max(0, 100 - latency / 300 * 100)
    throughput_score = min(100, throughput / 100 * 100)
    utilization_score = max(0, 100 - utilization)
    score = round(
        signal_score * 0.22
        + snr_score * 0.24
        + loss_score * 0.18
        + latency_score * 0.12
        + throughput_score * 0.16
        + utilization_score * 0.08
    )
    label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"
    return score, label


def record_prediction(record):
    signature = tuple(record.items())
    if st.session_state.get("last_prediction_signature") != signature:
        st.session_state.prediction_history.append(record)
        st.session_state.last_prediction_signature = signature


def render_network_table(networks):
    if not networks:
        st.info(
            "No networks appeared in the latest scan. Use Scan now or enable automatic "
            "scanning to retry every 30 seconds. Manual measurements are still available."
        )
        return
    network_df = pd.DataFrame(networks)
    columns = [
        column
        for column in [
            "ssid",
            "connection_status",
            "bssid",
            "signal",
            "signal_percent",
            "channel",
            "security",
        ]
        if column in network_df
    ]
    display_df = network_df[columns].rename(
        columns={
            "ssid": "SSID",
            "connection_status": "Connection",
            "bssid": "BSSID",
            "signal": "Signal",
            "signal_percent": "Strength",
            "channel": "Channel",
            "security": "Security",
        }
    )
    display_df = display_df.replace(r"^\s*$", "Not reported", regex=True).fillna("Not reported")
    st.dataframe(display_df, hide_index=True, width="stretch")


def render_model_comparison():
    st.subheader("Model comparison")
    comparison_path = ROOT / "results" / "metrics" / "model_comparison.csv"
    report_path = ROOT / "results" / "metrics" / "classification_report.csv"
    confusion_path = ROOT / "results" / "metrics" / "confusion_matrix.csv"
    if not comparison_path.exists():
        st.info("No comparison report is available yet. Train a model from the Data and retraining tab.")
        return
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, hide_index=True, width="stretch")
    chart_columns = [column for column in ["accuracy", "macro_f1"] if column in comparison.columns]
    if chart_columns and "model" in comparison.columns:
        st.bar_chart(comparison.set_index("model")[chart_columns])
    report_col, matrix_col = st.columns(2)
    with report_col:
        st.markdown("**Classification report**")
        if report_path.exists():
            st.dataframe(pd.read_csv(report_path), width="stretch")
    with matrix_col:
        st.markdown("**Confusion matrix**")
        if confusion_path.exists():
            st.dataframe(pd.read_csv(confusion_path, index_col=0), width="stretch")


def render_retraining():
    st.subheader("Dataset management and retraining")
    st.caption("Upload a compatible CSV or Excel file, then retrain the candidate models without leaving the dashboard.")
    uploaded_file = st.file_uploader("Upload training dataset", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("Supported schemas include the adaptive Excel dataset and the legacy CSV format.")
        return
    st.write(f"Selected file: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
    if st.button("Retrain model", type="primary", icon=":material/model_training:"):
        upload_dir = ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / uploaded_file.name
        upload_path.write_bytes(uploaded_file.getvalue())
        try:
            from src.train import run_training_pipeline

            with st.spinner("Training candidate models and writing evaluation reports..."):
                run_training_pipeline([str(upload_path)])
            load_model_bundle.clear()
            st.success("Retraining completed. The dashboard is using the new model.")
            st.rerun()
        except Exception as exc:
            st.error(f"Retraining failed: {exc}")


def render_policy():
    st.subheader("Website policy and local data handling")
    st.markdown(
        """
        This project is a local diagnostic and research tool for adaptive IEEE 802.11 transmission-rate selection.
        It is designed to help users evaluate wireless link quality and receive model-based recommendations, not to
        modify hardware drivers or silently connect to protected networks.

        #### Data we process
        - Wi-Fi scan results exposed by the local operating system, such as SSID, BSSID, signal strength, channel, and security type.
        - User-entered wireless measurements, including RSSI, SNR, packet loss, latency, throughput, and current transmission rate.
        - Prediction history held in the current app session and CSV files explicitly downloaded by users.

        #### How the data is used
        - The app processes measurements on the local device to calculate a wireless link-health score.
        - The trained model uses those measurements to predict the most suitable transmission rate.
        - Measurements are processed by the machine running this Streamlit app. When hosted remotely, submitted values are sent to that host for processing.

        #### Storage and sharing
        - Session history is not a permanent database. Uploaded training files and generated artifacts on a hosted service may be temporary and subject to that host's storage policies.
        - CSV export files are generated only when the user explicitly downloads them.
        - This site does not use third-party tracking or analytics for Wi-Fi measurements by default.

        #### Limitations
        - Recommendations are advisory only; they do not change the adapter's hardware configuration.
        - The system cannot guarantee performance in every environment because wireless conditions vary by device, driver, and location.
        - Protected or hidden networks remain unavailable unless the operating system exposes the required metadata.

        #### Responsibility
        Users are responsible for validating any recommendation against their own network policy, hardware constraints, and operational requirements.
        This tool should be used as an educational or diagnostic aid, not as a production network control system.
        """
    )
    st.info("No separate analytics service is used by normal prediction logic. If this app is hosted remotely, measurements are processed by the hosting provider's server; use only data you are comfortable submitting to that host.")


st.session_state.setdefault("networks", [])
st.session_state.setdefault("scan_error", None)
st.session_state.setdefault("last_scan", "not yet scanned")
st.session_state.setdefault("last_scan_epoch", 0.0)
st.session_state.setdefault("prediction_history", [])
st.session_state.setdefault("last_prediction_signature", None)

pipeline, metadata, model_error = load_bundle()

st.title("Adaptive IEEE 802.11 transmission rate intelligence")
st.caption("Live Wi-Fi diagnostics, explainable model recommendations, and training controls")

with st.sidebar:
    st.header("Wireless link parameters")
    auto_refresh = st.toggle(
        "Auto scan",
        value=True,
        key="auto_scan_enabled",
        help="Automatically rescan nearby Wi-Fi networks every 30 seconds.",
    )
    if not st.session_state.networks and not st.session_state.last_scan_epoch:
        scan_and_store()
    if auto_refresh:
        st.caption(f"Automatic scan every {AUTO_SCAN_INTERVAL_SECONDS} seconds.")
    st.caption(f"Last scan: {st.session_state.last_scan}")
    if st.session_state.scan_error:
        st.warning(f"Wi-Fi scan unavailable: {st.session_state.scan_error}")
    if model_error:
        st.error(model_error)
    else:
        st.success("Model pipeline loaded")

    networks = st.session_state.networks
    network_options = ["Manual input"] + [network_label(network) for network in networks]
    selected_label = st.selectbox("Available nearby Wi-Fi", network_options, key="selected_network")
    selected_network = next((network for network in networks if network_label(network) == selected_label), None)
    default_rssi = float(selected_network.get("signal")) if selected_network and selected_network.get("signal") is not None else -48.0
    rssi = st.number_input("RSSI / RCPI (dBm)", -100.0, -20.0, default_rssi, 1.0)
    snr = st.number_input("SNR (dB)", -5.0, 40.0, 20.0, 1.0)
    packet_loss = st.number_input("Packet loss (%)", 0.0, 100.0, 1.0, 0.1)
    latency = st.number_input("Latency / RTT (ms)", 0.0, 5000.0, 5.0, 1.0)
    throughput = st.number_input("Received throughput (Mbps)", 0.0, 1000.0, 58.0, 0.1)
    channel_utilization = st.number_input("Channel utilization (%)", 0.0, 100.0, 38.0, 0.1)
    current_rate = st.number_input("Current TX rate (Mbps)", 0.0, 1000.0, 48.0, 0.1)


@st.fragment(
    run_every=(
        f"{AUTO_SCAN_INTERVAL_SECONDS}s"
        if st.session_state.get("auto_scan_enabled", True)
        else None
    )
)
def refresh_wifi_networks():
    if not st.session_state.get("auto_scan_enabled", True):
        return
    if time.time() - st.session_state.get("last_scan_epoch", 0.0) < AUTO_SCAN_INTERVAL_SECONDS:
        return

    scan_and_store()
    st.rerun()


refresh_wifi_networks()


dashboard_tab, comparison_tab, data_tab, policy_tab = st.tabs([
    "Dashboard",
    "Model comparison",
    "Data and retraining",
    "Website policy",
])

with dashboard_tab:
    scan_col, scan_status_col = st.columns([1, 3])
    with scan_col:
        if st.button(
            "Scan for nearby Wi-Fi",
            icon=":material/refresh:",
            type="primary",
            key="dashboard_scan",
        ):
            with st.spinner("Scanning for nearby Wi-Fi networks..."):
                scan_and_store()
            st.rerun()
    networks = st.session_state.networks
    with scan_status_col:
        st.caption(
            f"{len(networks)} network(s) reported by {platform.system()} · "
            f"Last scan: {st.session_state.last_scan}"
        )

    st.subheader(f"Nearby Wi-Fi networks ({len(networks)} detected)")
    if not networks:
        st.info(
            "No nearby networks were returned by the operating system. Confirm Wi-Fi is "
            "enabled and retry Scan for nearby Wi-Fi. Available networks depend on the "
            "adapter, driver, location, and the OS scan results."
        )
    elif len(networks) == 1:
        st.info(
            "The operating system currently reports one visible network. The list below "
            "shows it; other nearby access points will appear only if Windows detects them."
        )
    render_network_table(networks)

    health_score, health_label = network_health(
        rssi, snr, packet_loss, latency, throughput, channel_utilization
    )
    st.subheader("Live network summary")
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("RSSI", f"{rssi:.0f} dBm")
    kpi_cols[1].metric("SNR", f"{snr:.1f} dB")
    kpi_cols[2].metric("Packet loss", f"{packet_loss:.1f}%")
    kpi_cols[3].metric("Throughput", f"{throughput:.1f} Mbps")
    kpi_cols[4].metric("Health score", f"{health_score}/100", health_label)

    if health_score < 50:
        st.warning("Poor link health. A lower transmission rate may improve reliability.")
    elif health_score < 70:
        st.info("Fair link health. Monitor packet loss and channel utilization before increasing the rate.")
    else:
        st.success(f"{health_label} link health. Conditions are suitable for adaptive rate selection.")

    prediction_col, input_col = st.columns([1.25, 1])
    with prediction_col:
        st.subheader("AI prediction")
        if pipeline is None:
            st.warning("Prediction is unavailable until a valid model is trained.")
        else:
            input_data = build_prediction_features(
                rssi, snr, packet_loss, latency, throughput, current_rate, channel_utilization
            )
            try:
                prediction = pipeline.predict(input_data)[0]
                probabilities = pipeline.predict_proba(input_data)[0]
                confidence = float(max(probabilities) * 100)
                recommendation = "High-confidence recommendation" if confidence >= 80 else "Moderate-confidence recommendation" if confidence >= 60 else "Low-confidence recommendation"
                st.success(f"Recommended transmission rate: **{prediction} Mbps**")
                st.metric("Model confidence", f"{confidence:.2f}%")
                st.caption(recommendation)
                probability_frame = pd.DataFrame({"Rate": pipeline.classes_, "Probability": probabilities}).set_index("Rate")
                st.bar_chart(probability_frame)
                record_prediction({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "network": selected_network.get("ssid", "Manual input") if selected_network else "Manual input",
                    "rssi": round(rssi, 2),
                    "snr": round(snr, 2),
                    "packet_loss": round(packet_loss, 2),
                    "throughput": round(throughput, 2),
                    "health_score": health_score,
                    "predicted_rate": prediction,
                    "confidence": round(confidence, 2),
                })
            except Exception as exc:
                st.error(f"Prediction failed for the current measurements: {exc}")
    with input_col:
        st.subheader("Feature input")
        st.dataframe(
            pd.DataFrame({"Feature": [FEATURE_LABELS[key] for key in FEATURE_LABELS], "Value": [rssi, snr, packet_loss, latency, throughput, channel_utilization, current_rate]}),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Prediction history")
    history_frame = pd.DataFrame(st.session_state.prediction_history)
    if history_frame.empty:
        st.info("Change a measurement or select another network to begin building prediction history.")
    else:
        st.line_chart(history_frame.set_index("timestamp")[["health_score", "predicted_rate"]])
        st.dataframe(history_frame, hide_index=True, width="stretch")
        st.download_button(
            "Download prediction history",
            history_frame.to_csv(index=False).encode("utf-8"),
            "wifi_prediction_history.csv",
            "text/csv",
            icon=":material/download:",
        )

    if pipeline is not None and not history_frame.empty:
        latest = history_frame.iloc[-1]
        summary = pd.DataFrame([{
            "Selected access point": latest["network"],
            "Predicted rate (Mbps)": latest["predicted_rate"],
            "Confidence (%)": latest["confidence"],
            "Health score": latest["health_score"],
            "RSSI": latest["rssi"],
            "SNR": latest["snr"],
            "Packet loss (%)": latest["packet_loss"],
            "Throughput (Mbps)": latest["throughput"],
        }])
        st.download_button(
            "Download current result report",
            summary.to_csv(index=False).encode("utf-8"),
            "wifi_rate_result_report.csv",
            "text/csv",
            icon=":material/description:",
        )

with comparison_tab:
    render_model_comparison()

with data_tab:
    render_retraining()

with policy_tab:
    render_policy()
