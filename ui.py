import streamlit as st
import requests

st.title("Customer Segmentation Project")

age = st.number_input("Enter the age of the customer", min_value=18, max_value=80)
income = st.number_input("Enter the annual income of the customer", min_value=15, max_value=137)
spending_score = st.number_input("Enter the spending score of the customer", min_value=1, max_value=100)

if st.button("Predict"):
    customer = {
        "age": age,
        "income": income,
        "spending_score": spending_score
    }
    response = requests.post(url="http://127.0.0.1:8000/predict", json=customer)

    if response.status_code == 200:
        result = response.json()
        st.success(f"The customer belongs to segment: {result['segment']} - {result['segment_name']}")
    else:
        st.error("Error in prediction. Please try again later.")