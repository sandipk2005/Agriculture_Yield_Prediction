import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/yield_model.pkl")

# Title
st.title("🌾 Agriculture Yield Prediction")

# Input fields matching training feature names
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)
fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, step=0.1)

if st.button("Predict Yield"):
    # Create dataframe with the SAME column names used in training
    df = pd.DataFrame({
        "Rainfall(mm)": [rainfall],
        "Temperature(C)": [temperature],
        "Fertilizer(kg/ha)": [fertilizer]
    })
    
    prediction = model.predict(df)[0]
    st.success(f"🌾 Predicted Crop Yield: {prediction:.2f} ton/ha")
