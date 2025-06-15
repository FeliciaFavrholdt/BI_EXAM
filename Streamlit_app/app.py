import streamlit as st

# SET PAGE CONFIGURATION
st.set_page_config(
    page_title="AlzheimerPredictor4u - Group 4",
    layout="wide"
)

# HEADER BANNER
st.markdown("""
    <div style="background-color:#003366;padding:30px;border-radius:10px">
        <h1 style="color:white;text-align:center;">AlzheimerPredictor4u</h1>
        <h4 style="color:white;text-align:center;">Early Detection of Alzheimer’s Disease Using Predictive Analytics</h4>
    </div>
""", unsafe_allow_html=True)

st.write("")

# PROJECT OVERVIEW SECTION
st.markdown("""
<div style="background-color:#f5f5f5;padding:30px;border-radius:10px">

### Project Overview

This application presents a full **Data Science pipeline** to predict Alzheimer's disease diagnosis using clinical, demographic, and lifestyle patient data.

The project follows these main steps:

- Problem Statement  
- Data Preparation  
- Exploratory Data Analysis (EDA)  
- Model Development  
- Results & Evaluation  
- Business Application

</div>
""", unsafe_allow_html=True)

st.markdown("""---""")

# PROBLEM STATEMENT SECTION
st.markdown("""
### Problem Statement

**How can we use Business Intelligence and AI techniques to assess the risk of Alzheimer's disease based on demographic and lifestyle factors such as age, gender, health status, and daily habits, in order to support early detection and improve preventive care strategies?**

---
""")

# MOTIVATION SECTION
st.markdown("""
### Motivation

Our goal is to help healthcare professionals identify patients at high risk of developing Alzheimer’s disease before severe symptoms appear. Early detection allows better care, planning, and improves patients’ quality of life.

By analyzing patient data (including age, gender, lifestyle, and health status), we aim to build a system that supports earlier diagnosis using **Business Intelligence (BI)** and **Artificial Intelligence (AI)**.

---
""")

# PROJECT GOALS SECTION
st.markdown("""
### Project Goals

- Build a predictive machine learning model for Alzheimer’s diagnosis
- Identify key features most strongly linked to Alzheimer’s risk
- Create a user-friendly BI dashboard for clinical use
- Document the full BI pipeline (data prep, modeling, deployment)

---
""")

# TEAM MEMBERS SECTION
st.markdown("""
### Contributors

- **Felicia Favrholdt** — *cph-ff62@cphbusiness.dk* — [GitHub](https://github.com/FeliciaFavrholdt)
- **Fatima Majid Shamcizadh** — *cph-fs156@cphbusiness.dk* — [GitHub](https://github.com/Fati01600)

> Group 4 — l25dat4bi1f — CPH Business Lyngby — Business Intelligence 2025

---
""")

# DATASET SECTION
st.markdown("""
### Dataset

- Alzheimer's Disease Dataset by Rabie El Kharoua  
- [Kaggle Source](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data)
- DOI: 10.34740/KAGGLE/DSV/8668279

---
""")

# RESEARCH QUESTIONS SECTION
st.markdown("""
### Research Questions

- Can we predict Alzheimer's risk using demographic and lifestyle data?
- Which features are most predictive of an Alzheimer’s diagnosis?
- Can we build an interactive dashboard to support clinical decision-making?

---
""")

# FOOTER SECTION
st.caption("CPH Business 2025 — Group 4 — Business Intelligence Exam Project")
