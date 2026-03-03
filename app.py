import streamlit as st

import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as stat
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


# load model 
with open('customer_churn.pkl','rb') as file:
    model = pkl.load(file)

# streamlit app UI 
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction App")

st.markdown("Predict whether a customer is likely to **churn or stay** based on their subscription behavior.")

## Utility function to convert Yes/No to binary
def binary(val): return 1 if val == "Yes" else 0

# Input form
with st.form("churn_form"):
    st.subheader("🧾 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_electronic = st.selectbox("Payment: Electronic Check?", ["No", "Yes"])

    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict Churn")

if submitted:
    # Prepare input
    input_data = pd.DataFrame([{
        'SeniorCitizen': binary(senior_citizen),
        'Partner': binary(partner),
        'Dependents': binary(dependents),
        'PaperlessBilling': binary(paperless_billing),
        'OnlineSecurity_Yes': binary(online_security),
        'OnlineBackup_Yes': binary(online_backup),
        'DeviceProtection_Yes': binary(device_protection),
        'TechSupport_Yes': binary(tech_support),
        'StreamingTV_Yes': binary(streaming_tv),
        'StreamingMovies_Yes': binary(streaming_movies),
        'PaymentMethod_Electronic check': binary(payment_electronic),
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Show result
    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.error(f"🚨  Churn ")
        result_label = "Churn"
    else:
        st.success(f"✅ Not Churn")
        result_label = "Not Churn"

    # Show user input
    with st.expander("📋 See Submitted Details"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}))

    # Display Probability
    st.write(f"### 📊 Churn Probability: {round(probability * 100, 2)}%")

     # Pie Chart Visualization
    st.subheader("📈 Probability Distribution")

    fig, ax = plt.subplots()

    ax.pie(
        [probability, 1 - probability],
        labels=["Churn", "Not Churn"],
        autopct='%1.1f%%',
        colors=["#ff4d4d", "#5cb85c"],
        startangle=90
    )

    ax.axis("equal")  # Makes pie circular

    st.pyplot(fig)
 
   


       