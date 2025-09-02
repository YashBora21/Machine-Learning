import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("house_price_model.pkl")

st.title("🏡 House Price Prediction App")

# Input fields
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5)
sqft_living = st.number_input("Living Area (sqft)", min_value=0, step=50)
floors = st.number_input("Floors", min_value=0.0, step=0.5)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.number_input("View Score (0-4)", min_value=0, max_value=4, step=1)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, step=50)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)
yr_renovated_binary = st.selectbox("Renovated?", [0, 1])
zipcode = st.number_input("Zipcode", min_value=98000, max_value=99999, step=1)
city_freq = st.number_input("City Frequency", min_value=0, step=1)

# Predict button
if st.button("Predict Price"):
    features = np.array([[bedrooms, bathrooms, sqft_living, floors, waterfront,
                          view, sqft_basement, yr_built, yr_renovated_binary,
                          zipcode, city_freq]])
    log_pred = model.predict(features)   # ye log(price) predict karega
    prediction = np.expm1(log_pred)  
    price = prediction[0]    
    st.success(f"🏠 Estimated House Price: ${price:,.2f}")

