import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/yield_model.pkl")

st.title("🌾 Agriculture Yield Prediction App")

rainfall = st.number_input("🌧️ Rainfall (mm):", min_value=0)
temperature = st.number_input("🌡️ Temperature (°C):", min_value=0)
humidity = st.number_input("💧 Humidity (%):", min_value=0)

if st.button("Predict Yield"):
    df = pd.DataFrame([[rainfall, temperature, humidity]],
                      columns=["Rainfall", "Temperature", "Humidity"])
    prediction = model.predict(df)[0]
    st.success(f"🌿 Predicted Yield: {prediction:.2f} kg/ha")

