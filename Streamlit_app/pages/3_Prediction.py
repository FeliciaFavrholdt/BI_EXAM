import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the model
model = joblib.load('models/random_forest_model.pkl') 

# Define feature names
feature_names = [
    "Age", "Gender", "BMI", "PhysicalActivity", "SleepQuality",
    "FunctionalAssessment", "ADL", "MMSE",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Depression"
]

# Page configuration
st.set_page_config(page_title="Alzheimer's Prediction", layout="centered")
st.title("Alzheimer Disease Risk Prediction")

st.write("""
This model predicts the risk of Alzheimer's Disease based on patient information.
Please fill in the form below and click 'Predict'.
""")

# Build the input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    Age = col1.slider("Age", 50, 100, 70)
    Gender = col2.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
    BMI = col3.slider("BMI", 10.0, 40.0, 25.0)

    PhysicalActivity = col1.slider("Physical Activity", 0, 10, 5)
    SleepQuality = col2.slider("Sleep Quality", 0, 10, 5)
    FunctionalAssessment = col3.slider("Functional Assessment", 0, 10, 5)

    ADL = col1.slider("ADL (Activities of Daily Living)", 0, 10, 5)
    MMSE = col2.slider("MMSE", 0, 30, 24)

    FamilyHistoryAlzheimers = col3.selectbox("Family History of Alzheimer's", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    CardiovascularDisease = col1.selectbox("Cardiovascular Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Depression = col2.selectbox("Depression", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input dataframe directly using the simple variable names
        input_data = pd.DataFrame([[
            Age, Gender, BMI, PhysicalActivity, SleepQuality,
            FunctionalAssessment, ADL, MMSE,
            FamilyHistoryAlzheimers, CardiovascularDisease, Depression
        ]], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"Alzheimer's Disease Risk: {prediction_proba:.2%}")
        else:
            st.success(f"No Alzheimer's Disease Risk: {(1 - prediction_proba):.2%}")

        # Probability bar chart
        chart_df = pd.DataFrame({
            "Class": ["No Alzheimer's", "Alzheimer's"],
            "Probability": [(1 - prediction_proba), prediction_proba]
        })

        fig = px.bar(chart_df, x="Class", y="Probability", text="Probability", color="Class",
                     color_discrete_map={"No Alzheimer's": "green", "Alzheimer's": "red"},
                     labels={"Probability": "Probability"})
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)

        # Patient text summary
        st.subheader("Patient Profile Summary")

        Gender = "Female" if Gender == 1 else "Male"
        FamilyHistoryAlzheimers = "Yes" if FamilyHistoryAlzheimers == 1 else "No"
        CardiovascularDisease = "Yes" if CardiovascularDisease == 1 else "No"
        Depression = "Yes" if Depression == 1 else "No"

        summary = f"""
        - Age: {Age} years old
        - Gender: {Gender}
        - BMI: {BMI}
        - Physical Activity: {PhysicalActivity}/10
        - Sleep Quality: {SleepQuality}/10
        - Functional Assessment: {FunctionalAssessment}/10
        - ADL (Daily Living): {ADL}/10
        - MMSE Score: {MMSE}/30
        - Family History of Alzheimer's: {FamilyHistoryAlzheimers}
        - Cardiovascular Disease: {CardiovascularDisease}
        - Depression: {Depression}
        """

        st.markdown(summary)

        # Visual Feature Profile
        st.subheader("Visual Feature Profile")

        feature_values = [
            Age, Gender, BMI, PhysicalActivity, SleepQuality,
            FunctionalAssessment, ADL, MMSE,
            1 if FamilyHistoryAlzheimers == "Yes" else 0,
            1 if CardiovascularDisease == "Yes" else 0,
            1 if Depression == "Yes" else 0
        ]

        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": feature_values
        }).sort_values(by="Feature")

        feature_fig = px.bar(
            feature_df, 
            x="Feature", 
            y="Value", 
            orientation="v", 
            color="Value", 
            color_continuous_scale="Blues"
        )
        st.plotly_chart(feature_fig, use_container_width=True)

# Footer
st.caption("CPH Business 2025 — Group 4 — Business Intelligence Exam Project")
