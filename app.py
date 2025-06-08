import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model
model = pickle.load(open("Model/churn_model.pkl", "rb"))

# Title
st.title("📊 Customer Churn Prediction App")

# Input features
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Convert categorical input same as model training
def encode_input():
    binary_map = {'Yes': 1, 'No': 0}
    service_map = {'No': 0, 'Yes': 1, 'No phone service': 2, 'No internet service': 2}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
    internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}

    data = {
        "gender": 0 if gender == "Male" else 1,
        "SeniorCitizen": SeniorCitizen,
        "Partner": binary_map[Partner],
        "Dependents": binary_map[Dependents],
        "PhoneService": binary_map[PhoneService],
        "MultipleLines": service_map[MultipleLines],
        "InternetService": internet_map[InternetService],
        "OnlineSecurity": service_map[OnlineSecurity],
        "OnlineBackup": service_map[OnlineBackup],
        "DeviceProtection": service_map[DeviceProtection],
        "TechSupport": service_map[TechSupport],
        "StreamingTV": service_map[StreamingTV],
        "StreamingMovies": service_map[StreamingMovies],
        "Contract": contract_map[Contract],
        "PaperlessBilling": binary_map[PaperlessBilling],
        "PaymentMethod": payment_map[PaymentMethod],
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    return pd.DataFrame([data])

# Prediction
if st.button("Predict Churn"):
    input_df = encode_input()
    
    # Optional: Scale features if required (only if used in training)
    # scaler = StandardScaler()
    # input_df_scaled = scaler.fit_transform(input_df)

    prediction = model.predict(input_df)
    result = "⚠️ Likely to Churn" if prediction[0] == 1 else "✅ Likely to Stay"
    st.success(result)
