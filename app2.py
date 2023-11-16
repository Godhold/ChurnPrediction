import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model



# Load the trained model
model = model.load('/Users/godholdalomenu/Desktop/ChurnPrediction/model.keras ')

# Load the trained scaler
scaler = joblib.load('/Users/godholdalomenu/Desktop/Churn_Deployment/churn_scaler.pkl')

def preprocess_input(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Use the loaded scaler to transform the input data
    input_scaled = scaler.transform(input_df)

    return input_scaled

def predict_churn(input_data):
    # Use your loaded model for prediction
    prediction = model.predict(input_data)  # Adjust this line based on your model's input requirements
    confidence = model.predict_proba(input_data)[:, 1].mean()  # Example assuming a binary classification model

    return prediction, confidence

def main():
    st.title("Churn Prediction")

    # Get input from the user
    contract = st.text_input("Contract", "")
    partner = st.text_input("Partner", "")
    dependents = st.text_input("Dependents", "")
    paperless_billing = st.text_input("PaperlessBilling", "")
    payment_method = st.text_input("PaymentMethod", "")
    total_charges = st.text_input("TotalCharges", "")
    service_features = st.text_input("ServiceFeatures", "")

    # Dummy input data (replace with actual input data for prediction)
    input_data = {
        'Contract': contract,
        'Partner': partner,
        'Dependents': dependents,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'TotalCharges': total_charges,
        'ServiceFeatures': service_features
    }

    # Preprocess the input data using the scaler
    input_scaled = preprocess_input(input_data)

    # Get churn prediction and confidence
    churn_prediction, confidence = predict_churn(input_scaled)

    # Display prediction result
    st.header("Churn Prediction Result")
    st.write(f"Predicted Churn: {churn_prediction}")
    st.write(f"Confidence: {confidence}")

if __name__ == '__main__':
    main()
