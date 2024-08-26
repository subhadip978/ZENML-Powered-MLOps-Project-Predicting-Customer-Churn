import numpy as np
import pickle

import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader

with open('encoder.pkl', 'rb') as encoder_file:
    ohe_payment_type = pickle.load(encoder_file)

with open('scaler.pkl', 'rb') as scaler_file:
    strn = pickle.load(scaler_file)

st.title("End to End Customer Satisfaction Pipeline with ZenML")

def main():
    price = st.number_input("Price", min_value=0.0, step=0.01, format="%.2f")
    freight_value = st.number_input("Freight Value", min_value=0.0, step=0.01, format="%.2f")
    product_height_cm = st.number_input("Product Height (cm)", min_value=0.0, step=0.1, format="%.1f")
    product_width_cm = st.number_input("Product Width (cm)", min_value=0.0, step=0.1, format="%.1f")
    payment_value = st.number_input("Payment Value", min_value=0.0, step=0.01, format="%.2f")
    purchase_delivery_diff = st.number_input("Purchase-Delivery Difference (days)", min_value=0, step=1)
    estimated_actual_delivery_diff = st.number_input("Estimated-Actual Delivery Difference (days)", min_value=0, step=1)

    payment_type_input = st.selectbox("Payment Type", ['credit_card', 'boleto', 'voucher', 'debit_card'])
    review_availability = st.number_input("Review Availability", min_value=0, step=1)


    X_strn = strn.transform(np.array([[price, freight_value, product_height_cm, product_width_cm, payment_value, purchase_delivery_diff, estimated_actual_delivery_diff]]))

    payment_type_encoded_input = ohe_payment_type.transform(np.array([payment_type_input]).reshape(-1, 1)).toarray()





    input_data = np.concatenate((X_strn,payment_type_encoded_input, np.array([[review_availability]])), axis=1)

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
     
        pred = service.predict(input_data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                pred
            )
        )




if __name__ == "__main__":
    main()



