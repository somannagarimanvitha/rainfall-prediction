import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Rainfall Prediction", layout="wide")

st.title("🌧️ Rainfall Prediction Using Multiple Algorithms")

# --------------------------------------------------
# Load Models WITHOUT cache (to avoid old loading bug)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))

st.success("✅ Both Models Loaded Successfully")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Enter Weather Details")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

features = np.array([[humidity, pressure, windspeed, winddirection]])

st.write("### Input Values")
st.write(pd.DataFrame(features,
         columns=["humidity","pressure","windspeed","winddirection"]))

# --------------------------------------------------
# Predict Button
# --------------------------------------------------
if st.sidebar.button("Predict Rainfall"):

    # Predict from BOTH models
    rf_result = rf_model.predict(features)[0]
    xgb_result = xgb_model.predict(features)[0]

    st.subheader("🔎 Algorithm Comparison")

    col1, col2 = st.columns(2)

    # ---------------- Random Forest ----------------
    with col1:
        st.markdown("## 🌲 Random Forest Prediction")

        if rf_result == 1:
            st.success("Rainfall Expected")
        else:
            st.warning("No Rainfall")

        st.write("Accuracy: **0.77**")

    # ---------------- XGBoost ----------------
    with col2:
        st.markdown("## ⚡ XGBoost Prediction")

        if xgb_result == 1:
            st.success("Rainfall Expected")
        else:
            st.warning("No Rainfall")

        st.write("Accuracy: **0.79 (Best Model)**")

st.markdown("---")
st.write("This app compares predictions from Random Forest and XGBoost models.")
