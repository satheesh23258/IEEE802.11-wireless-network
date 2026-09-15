import joblib
try:
    from src.preprocessing import build_prediction_features
except ModuleNotFoundError:
    from preprocessing import build_prediction_features

def predict_transmission_rate(rssi, snr, packet_loss, latency, throughput,
                              current_transmission_rate):
    pipeline = joblib.load("models/best_model.pkl")
    input_data = build_prediction_features(
        rssi, snr, packet_loss, latency, throughput,
        current_transmission_rate
    )
    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100
    return int(prediction), confidence
