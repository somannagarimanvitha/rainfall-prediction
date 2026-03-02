import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Rainfall Prediction", layout="wide")

# -------------------------------
# Safe Path (Works on Cloud)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Rainfall.csv")
model_path = os.path.join(BASE_DIR, "model.pkl")

# -------------------------------
# Load Dataset Automatically
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip().str.lower()
    return data

data = load_data()

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("🌧️ Rainfall Prediction System")
st.write("Predict rainfall using Machine Learning")

# -------------------------------
# Show Dataset
# -------------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Visualization 1
# -------------------------------
st.subheader("Rainfall Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="rainfall", data=data, ax=ax1)
st.pyplot(fig1)

# -------------------------------
# Visualization 2
# -------------------------------
st.subheader("Correlation Heatmap")
numeric_data = data.select_dtypes(include=['int64','float64'])
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# -------------------------------
# Visualization 3
# -------------------------------
st.subheader("Humidity Distribution")
fig3, ax3 = plt.subplots()
ax3.hist(data['humidity'], bins=20, edgecolor='black')
st.pyplot(fig3)

# -------------------------------
# Visualization 4
# -------------------------------
st.subheader("Humidity vs Pressure")
fig4, ax4 = plt.subplots()
ax4.scatter(data['humidity'], data['pressure'], alpha=0.5)
ax4.set_xlabel("Humidity")
ax4.set_ylabel("Pressure")
st.pyplot(fig4)

# -------------------------------
# Prediction Section
# -------------------------------
st.sidebar.header("Enter Weather Details")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

features = np.array([[humidity, pressure, windspeed, winddirection]])

if st.sidebar.button("Predict Rainfall"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("🌧️ Rainfall Expected")
    else:
        st.success("☀️ No Rainfall Expected")

# -------------------------------
# Accuracy Info (Static Display)
# -------------------------------
st.markdown("---")
st.subheader("Model Performance")
st.write("Random Forest Accuracy: **0.77**")
st.write("XGBoost Accuracy: **0.79 (Best Model)**")
