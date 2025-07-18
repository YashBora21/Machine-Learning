import streamlit as st
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Streamlit UI
st.title("💰 Insurance Charges Predictor")
st.write("Enter the information below to estimate insurance charges.")

# User Inputs
age = st.slider("Age", 18, 100, step=1)
bmi = st.number_input("BMI (e.g., 25.3)", min_value=10.0, max_value=60.0, step=0.1)
children = st.slider("Number of Children", 0, 5)
smoker = st.selectbox("Smoker", ["yes", "no"])
sex = st.selectbox("Sex", ["male", "female"])

# Derived features to match model
is_smoker = 1 if smoker == "yes" else 0
isfemale = 1 if sex == "female" else 0
bmi_category_obess = 1 if bmi > 29.9 else 0  # Assuming obesity threshold same as during training

# Create feature array
features = np.array([[age, bmi, children, is_smoker, bmi_category_obess, isfemale]])

# Prediction
if st.button("Predict Insurance Charges"):
    charges = model.predict(features)[0]
    st.success(f"💸 Estimated Insurance Charges: ₹{charges:,.2f}")
