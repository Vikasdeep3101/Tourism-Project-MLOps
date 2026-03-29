import streamlit as st
import pandas as pd
import joblib
import os

# 🔥 FORCE HF PORT BINDING
port = int(os.environ.get("PORT", 8501))

st.set_page_config(page_title="Wellness Predictor")

st.title("🌿 Wellness Tourism Purchase Predictor")

# FIX MODEL PATH
model_path = os.path.join(os.getcwd(), "model_building", "model.pkl")
model = joblib.load(model_path)

st.write("Enter Customer Details:")

age = st.number_input("Age", 18, 80)
income = st.number_input("Monthly Income")
pitch = st.slider("Pitch Satisfaction Score", 1, 5)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [income],
        "PitchSatisfactionScore": [pitch]
    })

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)

    if pred[0] == 1:
        st.success("Customer is likely to purchase")
    else:
        st.error("Customer is unlikely to purchase")

# 🔥 DEBUG LINE
st.write(f"Running on port: {port}")
