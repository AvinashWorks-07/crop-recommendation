import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load trained files
# -------------------------------
model = joblib.load("crop_model.joblib")
scaler = joblib.load("scaler.joblib")
le = joblib.load("label_encoder.joblib")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌱 AI Crop Recommendation System")
st.write("Enter soil & climate details to get crop recommendation")

# Inputs
N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
hum = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
ph_val = st.number_input("pH Value", 0.0, 14.0, 6.5)
rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temp, hum, ph_val, rain]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    crop_name = le.inverse_transform(prediction)[0]
    st.success(f"✅ Recommended Crop: **{crop_name}**")
