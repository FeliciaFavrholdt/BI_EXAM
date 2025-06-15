import streamlit as st
import pandas as pd
import numpy as np
import joblib

# SET PAGE CONFIGURATION
st.set_page_config(page_title="Prediction", layout="wide")
st.title("Model Prediction Dashboard")

# LOAD PRETRAINED MODEL FUNCTION (ASSUMING YOU HAVE MODEL SAVED LOCALLY)
@st.cache_resource
def load_model():
    return joblib.load('models/alzheimers_model.pkl')

# LOAD MODEL
model = load_model()

# DEFINE INPUT FEATURES (REPLACE WITH YOUR REAL MODEL FEATURES)
feature_names = ['Age', 'MMSE', 'CDR', 'PhysicalActivity', 'DietQuality']

# USER INPUT FORM SECTION
with st.container():
    st.markdown("<h4>Enter Patient Data for Prediction</h4>", unsafe_allow_html=True)
    with st.form("prediction_form"):
        # CREATE COLUMNS FOR NICE LAYOUT
        col1, col2, col3 = st.columns(3)

        # USER INPUT FIELDS
        age = col1.slider("Age", 60, 90, 70)
        mmse = col2.slider("MMSE Score", 0, 30, 24)
        cdr = col3.slider("CDR Score", 0.0, 3.0, 1.0, step=0.1)
        physical_activity = col1.slider("Physical Activity", 0, 10, 5)
        diet_quality = col2.slider("Diet Quality", 0, 10, 5)

        # SUBMIT BUTTON
        submitted = st.form_submit_button("Predict")

    # IF FORM IS SUBMITTED, RUN PREDICTION
    if submitted:
        # CREATE DATAFRAME FOR MODEL INPUT
        input_data = pd.DataFrame([[age, mmse, cdr, physical_activity, diet_quality]], columns=feature_names)

        # MAKE PREDICTION
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        # DISPLAY RESULTS
        st.write("### Prediction Result:")
        if prediction == 1:
            st.error(f"The model predicts: Alzheimer's Disease risk ({prediction_proba:.2%} probability)")
        else:
            st.success(f"The model predicts: No Alzheimer's Disease ({(1 - prediction_proba):.2%} probability)")

# FOOTER SECTION
st.markdown("---")
st.caption("CPH Business 2025 — Group 4 — Business Intelligence Exam Project")
