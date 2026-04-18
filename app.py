import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── Load Model & Scaler ───────────────────────────────────────
model   = joblib.load('model.pkl')
scaler  = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# ── Header ────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient details below to predict diabetes risk using a trained **Random Forest model** (ROC-AUC: 0.83)")
st.divider()

# ── Sidebar: About ────────────────────────────────────────────
with st.sidebar:
    st.header("📌 About")
    st.info("""
    This app predicts diabetes risk using the **Pima Indians Diabetes Dataset**.
    
    **Model:** Random Forest  
    **Accuracy:** 77%  
    **ROC-AUC:** 0.83  
    
    Built with Scikit-learn + Streamlit.
    """)
    st.header("⚠️ Disclaimer")
    st.warning("This is an ML demo, not a medical diagnosis tool.")

# ── Input Form ────────────────────────────────────────────────
st.subheader("🔢 Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies",           min_value=0,   max_value=20,  value=1)
    glucose     = st.slider("Glucose Level (mg/dL)",       min_value=50,  max_value=250, value=120)
    blood_pressure = st.slider("Blood Pressure (mm Hg)",   min_value=30,  max_value=130, value=70)
    skin_thickness = st.slider("Skin Thickness (mm)",      min_value=5,   max_value=100, value=20)

with col2:
    insulin     = st.slider("Insulin Level (µU/mL)",       min_value=15,  max_value=900, value=80)
    bmi         = st.slider("BMI",                         min_value=10.0,max_value=70.0,value=25.0)
    dpf         = st.slider("Diabetes Pedigree Function",  min_value=0.05,max_value=2.5, value=0.5)
    age         = st.slider("Age",                         min_value=18,  max_value=90,  value=30)

# ── Feature Engineering (must match notebook exactly) ─────────
def engineer_features(preg, gluc, bp, skin, ins, bmi, dpf, age):
    bmi_age              = bmi * age
    glucose_insulin_ratio = gluc / (ins + 1)
    risk_score           = (gluc * 0.4) + (bmi * 0.3) + (age * 0.2) + (dpf * 0.1)

    return pd.DataFrame([[preg, gluc, bp, skin, ins, bmi, dpf, age,
                          bmi_age, glucose_insulin_ratio, risk_score]],
                        columns=feature_columns)

# ── Predict Button ────────────────────────────────────────────
st.divider()
if st.button("🔍 Predict", use_container_width=True):

    input_df   = engineer_features(pregnancies, glucose, blood_pressure,
                                   skin_thickness, insulin, bmi, dpf, age)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    # Result card
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes** — Confidence: {probability*100:.1f}%")
    else:
        st.success(f"✅ **Low Risk of Diabetes** — Confidence: {(1-probability)*100:.1f}%")

    # Probability gauge
    st.metric(label="Diabetes Probability", value=f"{probability*100:.1f}%")
    st.progress(float(probability))

    # ── SHAP Explanation ──────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this prediction? (SHAP)")

    # ── SHAP Explanation ──────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this prediction? (SHAP)")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    # Handle both old and new SHAP output formats
    if isinstance(shap_values, list):
        sv = shap_values[1][0]          # old format: list of arrays
        ev = explainer.expected_value[1]
    else:
        sv = shap_values[0, :, 1]       # new format: 3D array
        ev = explainer.expected_value[1]

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = sv,
            base_values   = ev,
            data          = input_df.iloc[0].values,
            feature_names = feature_columns
        ),
        show=False
    )
    st.pyplot(fig)
    st.caption("🔴 Red bars push toward Diabetes risk · 🔵 Blue bars push away from risk")