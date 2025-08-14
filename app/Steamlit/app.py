import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os

# Dictionary with model names and direct download URLs
MODEL_URLS = {
    "RandomForestClassifier_model": "https://drive.google.com/uc?export=download&id=13ZKcEhIYwOOK-HvBFXa6yVhv70S_7z7I",
    "LogisticRegression_model": "https://drive.google.com/uc?export=download&id=13ZKcEhIYwOOK-HvBFXa6yVhv70S_7z7I",
    "SVC_model": "https://drive.google.com/uc?export=download&id=1doxYGJMHM5V9Z8sHvvq7nEtn_jelWOdP",
    "XGB_model": "https://drive.google.com/uc?export=download&id=1U6IYytsdbvaKqNKSjQ5uXcTNu2SA73qe"
}

# Create a folder for models
os.makedirs("models", exist_ok=True)

# Download models if not exist
for name, url in MODEL_URLS.items():
    file_path = f"models/{name}.pkl"
    if not os.path.exists(file_path):
        st.write(f"📥 Downloading {name}...")
        r = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(r.content)

# Load models
models = {}
for name in MODEL_URLS.keys():
    file_path = f"models/{name}.pkl"
    if os.path.exists(file_path):
        models[name] = joblib.load(file_path)

if not models:
    st.error("❌ No models could be loaded.")
    st.stop()

# App title
st.title("Heart Failure Prediction App 🫀")

# Model selection
model_name = st.selectbox("Select a model", list(models.keys()))
selected_model = models[model_name]
st.write(f"**Selected model:** {model_name}")

# User input fields
st.subheader("Enter Patient Data")
age = st.slider("Age", min_value=40, max_value=95, value=60)
anaemia = st.selectbox("Anaemia", ["Yes", "No"])
creatinine_phosphokinase = st.slider("Creatinine Phosphokinase (mcg/L)", min_value=40, max_value=8000, value=200)
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
ejection_fraction = st.slider("Ejection Fraction (%)", min_value=10, max_value=70, value=40)
high_blood_pressure = st.selectbox("High Blood Pressure", ["Yes", "No"])
platelets = st.slider("Platelet Count (kiloplatelets/mL)", min_value=47000, max_value=500000, value=250000, step=1000)
serum_creatinine = st.slider("Serum Creatinine (mg/dL)", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
serum_sodium = st.slider("Serum Sodium (mEq/L)", min_value=115, max_value=150, value=137)
sex = st.selectbox("Sex", ["Male", "Female"])
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
followup_days = st.slider("Follow-up Days", min_value=4, max_value=300, value=100)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[
        age,
        1 if anaemia == "Yes" else 0,
        creatinine_phosphokinase,
        1 if diabetes == "Yes" else 0,
        ejection_fraction,
        1 if high_blood_pressure == "Yes" else 0,
        platelets,
        serum_creatinine,
        serum_sodium,
        1 if sex == "Male" else 0,
        1 if smoking_status == "Yes" else 0,
        followup_days
    ]], columns=[
        'Age', 'Anaemia', 'CreatininePhosphokinase', 'Diabetes',
        'EjectionFraction', 'HighBloodPressure', 'PlateletCount',
        'SerumCreatinine', 'SerumSodium', 'Sex', 'SmokingStatus',
        'FollowupDays'
    ])

    prediction = selected_model.predict(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ High risk of Heart Failure")
    else:
        st.success("✅ Low risk of Heart Failure")
