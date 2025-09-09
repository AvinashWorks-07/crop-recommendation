import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🌱 AI Crop Recommendation System")
st.write("Enter soil and climate details to get crop suggestions")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
hum = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temp, hum, ph, rain]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

