import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(repo_id="partha90/churn-model", filename="best_churn_model.joblib")
model = joblib.load(model_path)

st.title("Customer Churn Prediction App")
st.write("Internal tool for bank staff to predict whether a customer is at risk of churning.")

CreditScore     = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
Geography       = st.selectbox("Geography", ["France", "Germany", "Spain"])
Age             = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure          = st.number_input("Tenure (years with bank)", value=12)
Balance         = st.number_input("Account Balance", min_value=0.0, value=10000.0)
NumOfProducts   = st.number_input("Number of Products", min_value=1, value=1)
HasCrCard       = st.selectbox("Has Credit Card?", ["Yes", "No"])
IsActiveMember  = st.selectbox("Is Active Member?", ["Yes", "No"])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

input_data = pd.DataFrame([{
    'CreditScore':     CreditScore,
    'Geography':       Geography,
    'Age':             Age,
    'Tenure':          Tenure,
    'Balance':         Balance,
    'NumOfProducts':   NumOfProducts,
    'HasCrCard':       1 if HasCrCard == "Yes" else 0,
    'IsActiveMember':  1 if IsActiveMember == "Yes" else 0,
    'EstimatedSalary': EstimatedSalary
}])

if st.button("Predict"):
    proba = model.predict_proba(input_data)[0, 1]
    result = "churn" if proba >= 0.45 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to **{result}**.")
