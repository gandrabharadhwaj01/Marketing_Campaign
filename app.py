import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Personality Segmentation")

st.write("Enter Customer Details")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Income")
recency = st.number_input("Recency (Days since last purchase)")
spending = st.number_input("Total Spending")
purchases = st.number_input("Total Purchases")
children = st.number_input("Total Children")
web_visits = st.number_input("Web Visits Per Month")

if st.button("Predict Cluster"):

    data = np.array([[age, income, recency, spending, purchases, children, web_visits]])
    
    scaled_data = scaler.transform(data)
    
    cluster = model.predict(scaled_data)

    st.success(f"Customer belongs to Segment {cluster[0]}")

    # Optional: Segment Interpretation
    if cluster[0] == 0:
        st.write("Segment 0: High Value Customer")
    elif cluster[0] == 1:
        st.write("Segment 1: Budget Customer")
    elif cluster[0] == 2:
        st.write("Segment 2: Low Engagement")
    else:
        st.write("Segment 3: Moderate Customer")
