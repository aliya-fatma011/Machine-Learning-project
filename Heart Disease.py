import numpy as np
import pandas as pd
import streamlit as st
import pickle

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
.main {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
}
h1 {
    color: #e63946;
    text-align: center;
}
.stButton > button {
    background-color: #e63946;
    color: white;
    font-size: 18px;
    padding: 10px 30px;
    border-radius: 10px;
}
.stButton > button:hover {
    background-color: #1d3557;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Load model
with open("LogisticR.pkl", "rb") as file:
    model = pickle.load(file)

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1,0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [1,0])

with col2:
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [1,0])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak, slope]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
