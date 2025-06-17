import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Alzheimer Dashboard", layout="wide")
st.title("Alzheimer Dataset Interactive Dashboard")

# User Instructions
with st.expander("Instructions"):
    st.markdown("""
    - **Filter the dataset** using the options in the sidebar.
    - All available features can be filtered interactively.
    - The plots and KPIs will automatically update as you apply filters.
    - Use the 2D scatterplot to explore relationships between features.
    - Review the correlation heatmap to observe how variables interact.
    """)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('data/alzheimers_clean.csv') 

df = load_data()

# Sidebar Filters — ALL features
st.sidebar.header("Filter Dataset")

# Age filter
age_min, age_max = st.sidebar.slider("Age", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))
# Gender
gender = st.sidebar.multiselect("Gender", [0, 1], default=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
# BMI
bmi_min, bmi_max = st.sidebar.slider("BMI", float(df.BMI.min()), float(df.BMI.max()), (float(df.BMI.min()), float(df.BMI.max())))
# PhysicalActivity
pa_min, pa_max = st.sidebar.slider("Physical Activity", int(df.PhysicalActivity.min()), int(df.PhysicalActivity.max()), (int(df.PhysicalActivity.min()), int(df.PhysicalActivity.max())))
# SleepQuality
sleep_min, sleep_max = st.sidebar.slider("Sleep Quality", int(df.SleepQuality.min()), int(df.SleepQuality.max()), (int(df.SleepQuality.min()), int(df.SleepQuality.max())))
# FunctionalAssessment
fa_min, fa_max = st.sidebar.slider("Functional Assessment", int(df.FunctionalAssessment.min()), int(df.FunctionalAssessment.max()), (int(df.FunctionalAssessment.min()), int(df.FunctionalAssessment.max())))
# ADL
adl_min, adl_max = st.sidebar.slider("ADL", int(df.ADL.min()), int(df.ADL.max()), (int(df.ADL.min()), int(df.ADL.max())))
# MMSE
mmse_min, mmse_max = st.sidebar.slider("MMSE", int(df.MMSE.min()), int(df.MMSE.max()), (int(df.MMSE.min()), int(df.MMSE.max())))
# FamilyHistoryAlzheimers
family = st.sidebar.multiselect("Family History of Alzheimer's", [0, 1], default=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# CardiovascularDisease
cvd = st.sidebar.multiselect("Cardiovascular Disease", [0, 1], default=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# Depression
depression = st.sidebar.multiselect("Depression", [0, 1], default=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Apply filters
filtered_df = df[
    (df.Age >= age_min) & (df.Age <= age_max) &
    (df.Gender.isin(gender)) &
    (df.BMI >= bmi_min) & (df.BMI <= bmi_max) &
    (df.PhysicalActivity >= pa_min) & (df.PhysicalActivity <= pa_max) &
    (df.SleepQuality >= sleep_min) & (df.SleepQuality <= sleep_max) &
    (df.FunctionalAssessment >= fa_min) & (df.FunctionalAssessment <= fa_max) &
    (df.ADL >= adl_min) & (df.ADL <= adl_max) &
    (df.MMSE >= mmse_min) & (df.MMSE <= mmse_max) &
    (df.FamilyHistoryAlzheimers.isin(family)) &
    (df.CardiovascularDisease.isin(cvd)) &
    (df.Depression.isin(depression))
]

# KPIs
st.subheader("Cleaned Dataset Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", filtered_df.shape[0])
col2.metric("Total Features In Dataset", filtered_df.shape[1])
col3.metric("Missing Values", filtered_df.isna().sum().sum())

# Diagnosis pie chart
st.subheader("Diagnosis Distribution")
diagnosis_counts = filtered_df['Diagnosis'].value_counts().sort_index()
labels = ["No Alzheimer's", "Alzheimer's"]
values = [diagnosis_counts.get(0, 0), diagnosis_counts.get(1, 0)]
fig_pie = px.pie(values=values, names=labels, hole=0.4, color_discrete_sequence=["#4CAF50", "#E74C3C"])
st.plotly_chart(fig_pie, use_container_width=True)

# 2D Scatter Plot (dynamic feature selection)
st.subheader("2D Feature Space")
st.write("Select features for the X and Y axes to explore relationships between different variables in the dataset.")
x_axis = st.selectbox("Select X-axis", filtered_df.columns.drop('Diagnosis'))
y_axis = st.selectbox("Select Y-axis", filtered_df.columns.drop('Diagnosis'), index=1)

fig_2d = px.scatter(
    filtered_df,
    x=x_axis, y=y_axis,
    color=filtered_df['Diagnosis'].map({0: "No Alzheimer's", 1: "Alzheimer's"}),
    symbol='Diagnosis',
    opacity=0.7
)
st.plotly_chart(fig_2d, use_container_width=True)

# Correlation heatmap
st.subheader("Correlation Heatmap")
st.write("This heatmap shows the correlation between numeric features in the dataset. Darker colors indicate stronger correlations.")
numeric_cols = df.select_dtypes(include=np.number).drop(columns=["Diagnosis"])
corr = numeric_cols.corr()

fig_heat = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='Viridis'))
fig_heat.update_layout(width=700, height=700)
st.plotly_chart(fig_heat, use_container_width=True)


# Data Table Output
st.subheader("Data Table for filtered Output")
st.write("The table below shows the filtered dataset based on your selections. You can scroll horizontally to view all columns.")
st.dataframe(filtered_df, use_container_width=True)
