import streamlit as st
import joblib
import json
import pandas as pd

# Configure page theme
st.set_page_config(
    page_title="AI-Based IEEE 802.11 Transmission Rate Selector",
    page_icon="⚡",
    layout="centered"
)

# Custom CSS to make the website look professional
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI-Based IEEE 802.11 Transmission Rate Selection")
st.write("""
This interactive dashboard uses the selected machine-learning classifier to predict the best IEEE 802.11
transmission rate from current wireless network conditions.
""")

# Load trained model
try:
    pipeline = joblib.load("models/best_model.pkl")
    with open("models/model_metadata.json", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    st.sidebar.success("✅ Model Pipeline Loaded Successfully!")
except Exception as e:
    pipeline = None
    metadata = {}
    st.sidebar.error(f"❌ Model not found. Error: {str(e)}")

st.sidebar.header("⚙️ Wireless Link Parameters")

# Inputs
rssi = st.sidebar.number_input("RSSI / RCPI (dBm)", min_value=-100.0, max_value=-20.0, value=-48.0, step=1.0)
snr = st.sidebar.number_input("SNR (dB)", min_value=-5.0, max_value=40.0, value=20.0, step=1.0)
packet_loss = st.sidebar.number_input("Packet Loss (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
latency = st.sidebar.number_input("Latency / RTT (ms)", min_value=0.0, max_value=5000.0, value=5.0, step=1.0)
throughput = st.sidebar.number_input("Received Throughput (Mbps)", min_value=0.0, max_value=1000.0, value=58.0, step=0.1)
current_rate = st.sidebar.number_input("Current TX Rate (Mbps)", min_value=0.0, max_value=1000.0, value=48.0, step=0.1)

if packet_loss > 20 or snr < 5 or rssi < -80:
    st.warning("The link conditions indicate a weak or unstable wireless connection. A lower rate may improve reliability.")

# Main panel layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("📥 Input Configuration Summary")
    st.info(f"**RSSI/RCPI:** {rssi:.0f} dBm\n\n**SNR:** {snr:.1f} dB\n\n**Packet Loss:** {packet_loss:.1f}%\n\n**Latency/RTT:** {latency:.1f} ms\n\n**Throughput:** {throughput:.1f} Mbps\n\n**Current TX Rate:** {current_rate:.1f} Mbps")

with col2:
    st.subheader("🔮 AI Decision Output")
    if pipeline is None:
        st.warning("Model loading issues. Please ensure models/best_model.pkl exists.")
    else:
        # Prepare input data
        input_data = pd.DataFrame([{
            "rssi": rssi,
            "snr": snr,
            "packet_loss": packet_loss,
            "latency": latency,
            "throughput": throughput,
            "current_transmission_rate": current_rate,
        }])
        
        # Prediction & Probability estimation
        pred = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0]
        confidence = max(proba) * 100
        
        st.success(f"### Recommended MCS: **{pred}**")
        st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
        st.info("The model predicts MCS. The dataset does not provide a valid 6–54 Mbps mapping.")

st.markdown("***")
st.subheader("📊 Dataset Target Classes")
st.write("MCS classes 1 through 11 from the uploaded wireless datasets")

with st.expander("Model quality and feature importance"):
    if metadata:
        quality_col1, quality_col2 = st.columns(2)
        quality_col1.metric("Test accuracy", f"{metadata['test_accuracy']:.2%}")
        quality_col2.metric("Macro F1-score", f"{metadata['macro_f1']:.2%}")
        importance = pd.Series(metadata["feature_importance"], name="importance").sort_values(ascending=False)
        st.bar_chart(importance)
    else:
        st.info("Model metadata is unavailable. Run python src/train.py to regenerate it.")
