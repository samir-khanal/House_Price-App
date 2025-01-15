import numpy as np
import streamlit as st
from joblib import load
from model import r2,mse

model = load('house_price_prediction.pkl')

st.title("House Price Prediction App")

floors = st.number_input("Enter the number of floors:", min_value=1, max_value=5, value=1)
year_built = st.number_input("Enter the year the house was built:", min_value=1900, max_value=2025, value=2000)
condition = st.selectbox("Select the house condition:", options=["Excellent", "Good", "Fair", "Poor"])

age = 2025 - year_built
condition_multiplier = {
    "Excellent": 1.5,
    "Good": 1.2,
    "Fair": 1.0,
    "Poor": 0.8
}[condition]

if st.button("Predict Price"):
    # Predict the final price using the model
    input_features = np.array([[floors, age, condition_multiplier]])
    predicted_price = model.predict(input_features)[0]
    st.success(f"Prediction for the price of the house is: RS {predicted_price:,.2f}")

    st.subheader("Model Evaluation (on test data):")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    accuracy_percentage = r2 * 100
    st.success(f"It means this model has accuracy of {accuracy_percentage:.2f}% ")


