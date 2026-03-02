import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Rainfall Prediction", layout="wide")

st.title("🌧️ Rainfall Prediction Using Machine Learning")
st.write("Comparison of Random Forest and XGBoost Algorithms")

# --------------------------------------------------
# Load Files (Safe for Local + Cloud)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "Rainfall.csv")
rf_path = os.path.join(BASE_DIR, "rf_model.pkl")
xgb_path = os.path.join(BASE_DIR, "xgb_model.pkl")

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()

rf_model = joblib.load(rf_path)
xgb_model = joblib.load(xgb_path)

# --------------------------------------------------
# Show Dataset
# --------------------------------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(data.head())

# --------------------------------------------------
# Visualizations
# --------------------------------------------------
st.subheader("📊 Data Visualizations")

# Graph 1 – Rainfall Distribution
fig1, ax1 = plt.subplots()
sns.countplot(x="rainfall", data=data, ax=ax1)
ax1.set_title("Rainfall Class Distribution")
st.pyplot(fig1)

# Graph 2 – Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(data.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Feature Correlation Heatmap")
st.pyplot(fig2)

# Graph 3 – Humidity Histogram
fig3, ax3 = plt.subplots()
ax3.hist(data['humidity'], bins=20, edgecolor='black')
ax3.set_title("Humidity Distribution")
st.pyplot(fig3)

# Graph 4 – Scatter Plot
fig4, ax4 = plt.subplots()
ax4.scatter(data['humidity'], data['pressure'], alpha=0.5)
ax4.set_xlabel("Humidity")
ax4.set_ylabel("Pressure")
ax4.set_title("Humidity vs Pressure")
st.pyplot(fig4)

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Enter Weather Details")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

features = np.array([[humidity, pressure, windspeed, winddirection]])

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
if st.sidebar.button("Predict Rainfall"):

    rf_result = rf_model.predict(features)[0]
    xgb_result = xgb_model.predict(features)[0]

    st.subheader("🔎 Prediction Comparison")

    col1, col2 = st.columns(2)

    # Random Forest Result
    with col1:
        st.markdown("### 🌲 Random Forest")
        if rf_result == 1:
            st.success("Rainfall Expected")
        else:
            st.info("No Rainfall Expected")

        st.write("Accuracy: **0.77**")

    # XGBoost Result
    with col2:
        st.markdown("### ⚡ XGBoost")
        if xgb_result == 1:
            st.success("Rainfall Expected")
        else:
            st.info("No Rainfall Expected")

        st.write("Accuracy: **0.79 (Best Model)**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.write("This application compares two machine learning models to improve rainfall prediction accuracy.")
