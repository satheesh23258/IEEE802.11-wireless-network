import pandas as pd

FEATURE_COLUMNS = [
    "rssi",
    "snr",
    "packet_loss",
    "latency",
    "throughput",
    "current_transmission_rate",
]

RATE_OPTIONS = list(range(1, 12))
MCS_TO_RATE = {0: 6, 1: 12, 2: 12, 3: 24, 4: 36, 5: 48, 6: 54, 7: 54}

def clean_and_impute(df):
    df_clean = df.copy()
    df_clean["throughput"] = df_clean.groupby("experiment")["throughput"].ffill().bfill()
    df_clean = df_clean.dropna(subset=["throughput", "probability", "distance_or_metric", "mcs"])
    return df_clean


def build_wireless_features(df):
    """Build features from either the new wireless dataset or the legacy CSV."""
    if {"RCPI (dBm)", "SNR (dB)", "Packet Loss (%)", "RTT (Mean)", "Received Mbps", "Mbps TX", "MCS TX"}.issubset(df.columns):
        numeric = df.copy()
        feature_names = ["RCPI (dBm)", "SNR (dB)", "Packet Loss (%)", "RTT (Mean)", "Received Mbps", "Mbps TX", "MCS TX"]
        for column in feature_names:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        numeric = numeric.dropna(subset=feature_names).drop_duplicates()
        features = numeric[["RCPI (dBm)", "SNR (dB)", "Packet Loss (%)", "RTT (Mean)", "Received Mbps", "Mbps TX"]].copy()
        features.columns = FEATURE_COLUMNS
        return features, numeric["MCS TX"].astype(int)

    clean_df = clean_and_impute(df)
    features = pd.DataFrame(index=clean_df.index)
    probability = clean_df["probability"].clip(0.0, 1.0)
    distance = clean_df["distance_or_metric"].clip(lower=0.0)
    throughput = clean_df["throughput"].clip(lower=0.0)

    features["rssi"] = (-30.0 - 80.0 * distance + 15.0 * probability).clip(-100.0, -20.0)
    features["snr"] = (5.0 + 28.0 * probability - 12.0 * distance).clip(-5.0, 40.0)
    features["packet_loss"] = ((1.0 - probability) * 100.0).clip(0.0, 100.0)
    features["latency"] = (2.0 + 35.0 * distance + 0.15 * features["packet_loss"]).clip(1.0, 200.0)
    features["throughput"] = throughput
    features["current_transmission_rate"] = (6.0 + throughput / 60.0 * 48.0).clip(6.0, 54.0)
    return features[FEATURE_COLUMNS], clean_df["mcs"].map(MCS_TO_RATE).astype(int)


def build_prediction_features(rssi, snr, packet_loss, latency, throughput,
                               current_transmission_rate):
    return pd.DataFrame([{
        "rssi": rssi,
        "snr": snr,
        "packet_loss": packet_loss,
        "latency": latency,
        "throughput": throughput,
        "current_transmission_rate": current_transmission_rate,
    }], columns=FEATURE_COLUMNS)


def load_wireless_training_data(paths):
    frames = []
    for path in paths:
        raw = pd.read_csv(path, header=None, low_memory=False)
        if "MCS TX" in raw.iloc[1].astype(str).tolist():
            raw.columns = raw.iloc[1].tolist()
            frame = raw.iloc[2:].copy()
        else:
            frame = pd.read_csv(path, low_memory=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates()
