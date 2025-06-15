import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# SET PAGE CONFIGURATION
st.set_page_config(page_title="Dashboard", layout="centered")
st.title("AlzheimerPredictor4u - Dashboard")

# LOAD DATA FUNCTION (CACHED)
@st.cache_data
def load_data():
    return pd.read_csv('data/alzheimers_clean.csv')

# LOAD DATA
df = load_data()

# FILTER SECTION — 2 ROWS FOR COMPACT LAYOUT
with st.container():
    st.markdown("<h4>Filter Dataset</h4>", unsafe_allow_html=True)

    # FIRST ROW FILTERS
    col1, col2 = st.columns(2)

    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_min = col1.number_input("Minimum Age", min_value=min_age, max_value=max_age, value=min_age)
    age_max = col1.number_input("Maximum Age", min_value=min_age, max_value=max_age, value=max_age)

    gender_map = {0: 'Male', 1: 'Female'}
    gender_options = df['Gender'].unique()
    selected_genders = col2.multiselect(
        "Gender",
        gender_options,
        default=gender_options,
        format_func=lambda x: gender_map.get(x, str(x))
    )

    # SECOND ROW FILTERS
    col3, col4 = st.columns(2)

    min_mmse, max_mmse = int(df['MMSE'].min()), int(df['MMSE'].max())
    mmse_min = col3.number_input("Minimum MMSE", min_value=min_mmse, max_value=max_mmse, value=min_mmse)
    mmse_max = col4.number_input("Maximum MMSE", min_value=min_mmse, max_value=max_mmse, value=max_mmse)

# APPLY FILTERS BASED ON INPUT
filtered_df = df[
    (df['Age'] >= age_min) & (df['Age'] <= age_max) &
    (df['Gender'].isin(selected_genders)) &
    (df['MMSE'] >= mmse_min) & (df['MMSE'] <= mmse_max)
]

# KPI SUMMARY SECTION — COMPACT CARDS
with st.container():
    st.markdown("<h4>Dataset Summary</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", filtered_df.shape[0])
    col2.metric("Features", filtered_df.shape[1])
    col3.metric("Missing Values", filtered_df.isna().sum().sum())

# DIAGNOSIS PIE CHART SECTION
with st.container():
    st.markdown("<h4>Diagnosis Distribution</h4>", unsafe_allow_html=True)
    diagnosis_counts = filtered_df['Diagnosis'].value_counts().sort_index()
    labels = ['No Alzheimer\'s', 'Alzheimer\'s']
    values = [diagnosis_counts.get(0, 0), diagnosis_counts.get(1, 0)]
    fig_diag = px.pie(
        names=labels,
        values=values,
        hole=0.4,
        color_discrete_sequence=["#2c3e50", "#4682B4"]
    )
    st.plotly_chart(fig_diag, use_container_width=True)

# FEATURE IMPORTANCE BAR CHART (STATIC PLACEHOLDER)
with st.container():
    st.markdown("<h4>Feature Importance (Example)</h4>", unsafe_allow_html=True)
    feature_data = {
        'Feature': ['MMSE', 'CDR', 'Age', 'PhysicalActivity', 'DietQuality'],
        'Importance': [0.35, 0.25, 0.20, 0.10, 0.10]
    }
    feature_df = pd.DataFrame(feature_data)
    fig_bar = px.bar(
        feature_df,
        x='Feature',
        y='Importance',
        text_auto=True,
        color='Importance',
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# FOOTER SECTION
st.markdown("---")
st.caption("CPH Business 2025 — Group 4 — Business Intelligence Exam Project")
