# Machine Learning Based Adaptive Transmission Rate Selection in IEEE 802.11 Networks

This project implements adaptive IEEE 802.11 transmission-rate selection using machine learning. It reads live wireless measurements such as RSSI, SNR, packet loss, latency, throughput, channel utilization, and current rate, then predicts the best transmission rate for the current network conditions. The main dataset used in this project is `data/WiFi_Transmission_Rate_Recommendation_Dataset_5000.xlsx`.

## Live demo

The hosted Streamlit dashboard is available at [Adaptive Wi-Fi Rate Intelligence](https://ieee80211-wireless-network-hamj7j5enebbygraxkuzpv.streamlit.app/). On the hosted instance, use **Manual input** for measurements: Wi-Fi scanning on a cloud server can only see networks available to that server.

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

## Deploy on Streamlit Community Cloud

The web app can be deployed from this GitHub repository on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in to Streamlit Community Cloud using the GitHub account that can access this repository.
2. Select **Create app** and choose **Yup, I have an app**.
3. Set the repository to `satheesh23258/IEEE802.11-wireless-network`, branch to `master`, and app file path to `app/web_app.py`.
4. In **Advanced settings**, select Python 3.12 to match the model's tested Python environment.
5. Deploy. The root `requirements.txt` contains the app dependencies, and `models/best_model.pkl` is tracked with Git LFS.

When hosted remotely, the app runs on the hosting server. Its Wi-Fi scan can only inspect networks visible to that server, not networks near each visitor. If scanning is unavailable on the host, use **Manual input** and enter measurements directly. Values submitted to a remotely hosted app are processed by the hosting provider, so do not enter sensitive network information unless you trust that provider. Session history is not a durable database, and uploaded files or generated files on a hosted instance may be temporary.

## Website policy and local data handling

- When run locally, the dashboard processes scan data and measurements on the local machine. When hosted remotely, submitted measurements are processed by that hosting server.
- The default prediction workflow does not use a separate analytics service or remote prediction backend.
- Prediction history is session-scoped, and CSV files are provided only when the user explicitly downloads them. Hosted filesystem changes may not persist across restarts.
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
