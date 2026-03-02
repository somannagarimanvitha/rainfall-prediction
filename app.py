import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. SAFE PATH HANDLING (Fixes Cloud Error)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_path = os.path.join(BASE_DIR, "models", "rf_model.pkl")
xgb_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
data_path = os.path.join(BASE_DIR, "Rainfall.csv")

# -------------------------------
# 2. LOAD MODELS SAFELY
# -------------------------------
@st.cache_resource
def load_models():
    rf_model = pickle.load(open(rf_path, "rb"))
    xgb_model = pickle.load(open(xgb_path, "rb"))
    return rf_model, xgb_model

rf_model, xgb_model = load_models()

# -------------------------------
# 3. LOAD DATASET
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip().str.lower()
    return data

data = load_data()

# -------------------------------
# 4. APP TITLE
# -------------------------------
st.title("🌧️ Rainfall Prediction Using Machine Learning")
st.markdown("### Predict rainfall using Random Forest and XGBoost")

# -------------------------------
# 5. SIDEBAR INPUT
# -------------------------------
st.sidebar.header("Enter Weather Parameters")

def user_input():
    pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
    humidity = st.sidebar.slider("Humidity", 0, 100, 60)
    dewpoint = st.sidebar.number_input("Dew Point", 0, 50, 25)
    winddirection = st.sidebar.number_input("Wind Direction", 0, 360, 180)
    windspeed = st.sidebar.number_input("Wind Speed", 0, 100, 20)
    cloud = st.sidebar.slider("Cloud Cover", 0, 100, 50)
    sunshine = st.sidebar.number_input("Sunshine", 0, 15, 7)

    features = pd.DataFrame({
        'pressure':[pressure],
        'humidity':[humidity],
        'dewpoint':[dewpoint],
        'winddirection':[winddirection],
        'windspeed':[windspeed],
        'cloud':[cloud],
        'sunshine':[sunshine]
    })

    return features

input_df = user_input()

st.subheader("🔎 Input Data")
st.write(input_df)

# -------------------------------
# 6. MAKE PREDICTIONS
# -------------------------------
rf_pred = rf_model.predict(input_df)
xgb_pred = xgb_model.predict(input_df)

st.subheader("🌦️ Prediction Results")

if rf_pred[0] == 1:
    st.success("Random Forest: Rainfall Expected")
else:
    st.info("Random Forest: No Rainfall")

if xgb_pred[0] == 1:
    st.success("XGBoost: Rainfall Expected")
else:
    st.info("XGBoost: No Rainfall")

# -------------------------------
# 7. SHOW MODEL ACCURACY
# -------------------------------
st.subheader("📊 Model Performance")
st.write("✔ Random Forest Accuracy: **0.77**")
st.write("✔ XGBoost Accuracy: **0.79 (Best Model)**")

# -------------------------------
# 8. DATA VISUALIZATIONS
# -------------------------------
st.subheader("📈 Dataset Visualizations")

# Rainfall Distribution
fig1, ax1 = plt.subplots()
sns.countplot(x="rainfall", data=data, ax=ax1)
ax1.set_title("Rainfall Class Distribution")
st.pyplot(fig1)

# Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(data.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Feature Correlation Heatmap")
st.pyplot(fig2)

# Humidity Distribution
fig3, ax3 = plt.subplots()
ax3.hist(data['humidity'], bins=20, edgecolor='black')
ax3.set_title("Humidity Distribution")
st.pyplot(fig3)

# Scatter Plot
fig4, ax4 = plt.subplots()
ax4.scatter(data['humidity'], data['pressure'], alpha=0.5)
ax4.set_xlabel("Humidity")
ax4.set_ylabel("Pressure")
ax4.set_title("Humidity vs Pressure")
st.pyplot(fig4)

# -------------------------------
# 9. FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Developed for Rainfall Prediction using Machine Learning Deployment")
