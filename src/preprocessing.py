import pandas as pd

FEATURE_COLUMNS = [
    "rssi",
    "snr",
    "packet_loss",
    "latency",
    "throughput",
    "channel_utilization",
    "current_transmission_rate",
]

RATE_OPTIONS = list(range(1, 12))
MCS_TO_RATE = {0: 6, 1: 12, 2: 12, 3: 24, 4: 36, 5: 48, 6: 54, 7: 54}


def clean_and_impute(df):
    df_clean = df.copy()
    if "experiment" in df_clean.columns and "throughput" in df_clean.columns:
        df_clean["throughput"] = df_clean.groupby("experiment")["throughput"].ffill().bfill()
    if {"throughput", "probability", "distance_or_metric", "mcs"}.issubset(df_clean.columns):
        df_clean = df_clean.dropna(subset=["throughput", "probability", "distance_or_metric", "mcs"])
    return df_clean


def build_wireless_features(df):
    """Build features from the adaptive IEEE 802.11 Excel dataset or the legacy CSV dataset."""
    numeric = df.copy()

    if {"RSSI", "SNR", "packet_loss", "latency", "throughput", "channel_utilization", "current_transmission_rate", "recommended_transmission_rate"}.issubset(numeric.columns):
        for column in ["RSSI", "SNR", "packet_loss", "latency", "throughput", "channel_utilization", "current_transmission_rate", "recommended_transmission_rate"]:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        numeric = numeric.dropna(subset=["RSSI", "SNR", "packet_loss", "latency", "throughput", "channel_utilization", "current_transmission_rate", "recommended_transmission_rate"]).drop_duplicates()
        features = numeric[["RSSI", "SNR", "packet_loss", "latency", "throughput", "channel_utilization", "current_transmission_rate"]].copy()
        features.columns = FEATURE_COLUMNS
        target = numeric["recommended_transmission_rate"].astype(int)
        target.name = "recommended_transmission_rate"
        return features, target

    if {"RCPI (dBm)", "SNR (dB)", "Packet Loss (%)", "RTT (Mean)", "Received Mbps", "Mbps TX", "MCS TX"}.issubset(df.columns):
        feature_names = ["RCPI (dBm)", "SNR (dB)", "Packet Loss (%)", "RTT (Mean)", "Received Mbps", "Mbps TX", "MCS TX"]
        for column in feature_names:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        numeric = numeric.dropna(subset=feature_names).drop_duplicates()
        features = numeric[["RCPI (dBm)", "SNR (dB)", "Packet Loss (%)", "RTT (Mean)", "Received Mbps", "Mbps TX"]].copy()
        features["channel_utilization"] = 0.0
        features.columns = FEATURE_COLUMNS
        target = numeric["MCS TX"].astype(int)
        target.name = "recommended_transmission_rate"
        return features, target

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
    features["channel_utilization"] = clean_df.get("channel_utilization", pd.Series(0.0, index=clean_df.index)).clip(0.0, 100.0)
    features["current_transmission_rate"] = (6.0 + throughput / 60.0 * 48.0).clip(6.0, 54.0)
    target = clean_df["mcs"].map(MCS_TO_RATE).astype(int)
    target.name = "recommended_transmission_rate"
    return features[FEATURE_COLUMNS], target


def build_prediction_features(rssi, snr, packet_loss, latency, throughput,
                              current_transmission_rate, channel_utilization=0.0):
    return pd.DataFrame([{
        "rssi": rssi,
        "snr": snr,
        "packet_loss": packet_loss,
        "latency": latency,
        "throughput": throughput,
        "channel_utilization": channel_utilization,
        "current_transmission_rate": current_transmission_rate,
    }], columns=FEATURE_COLUMNS)


def load_wireless_training_data(paths):
    frames = []
    for path in paths:
        path_str = str(path)
        if path_str.lower().endswith((".xlsx", ".xls")):
            frame = pd.read_excel(path_str)
        else:
            raw = pd.read_csv(path_str, header=None, low_memory=False)
            if raw.empty:
                continue
            if len(raw) > 1 and "MCS TX" in raw.iloc[1].astype(str).tolist():
                raw.columns = raw.iloc[1].tolist()
                frame = raw.iloc[2:].copy()
            else:
                frame = pd.read_csv(path_str, low_memory=False)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
