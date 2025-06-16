# BI-eksamen 2025

## Project Title
AlzheimerProdictor4u

## Contributors
#### Group 4, l25dat4bi1f 
Business Intelligence 2025  
Copenhagen Business Academy, Lyngby  

GitHub Repository: [AlzheimerPredictor4u_BI_Exam
](https://github.com/FeliciaFavrholdt/AlzheimerPredictor4u_BI_Exam)

### Felicia Favrholdt
- Email: [cph-ff62@cphbusiness.dk](mailto:cph-ff62@cphbusiness.dk)  
- GitHub: [https://github.com/FeliciaFavrholdt](https://github.com/FeliciaFavrholdt)

### Fatima Majid Shamcizadh
- Email: [cph-fs156@cphbusiness.dk](mailto:cph-fs156@cphbusiness.dk)  
- GitHub: [https://github.com/Fati01600](https://github.com/Fati01600)

# Problem Statement
*How can we use Business Intelligence and AI techniques to assess the risk of Alzheimer's disease based on demographic and lifestyle factors such as age, gender, health status, and daily habits, in order to support early detection and improve preventive care strategies?”*

# Research Questions
1. Can we predict the risk of Alzheimer's disease based on demographic and lifestyle factors such as age, gender, physical activity, and diet?
2. Which health and lifestyle features are most predictive of an Alzheimer’s diagnosis?
3. Can we build a predictive dashboard to visualize individual risk levels and support clinical decision-making?

## Project Goals

The main goal of this project is to build a simple and practical system that can help doctors and healthcare-staff assess a patient’s risk of developing Alzheimer’s disease. We want to do this by using Business Intelligence (BI) and Artificial Intelligence (AI) techniques on real patient data. 

### Our goals include:

- Creating a machine learning model that can predict the likelihood of an Alzheimer’s diagnosis.
- Identifying which features—such as age, gender, health conditions, and lifestyle habits are most strongly linked to Alzheimer’s risk.
- Designing an interactive dashboard that presents predictions and feature insights in a clear, user-friendly way for clinical use

## Hypotheses

In this project, we aim to uncover the following patterns in the dataset:
- **H1:** Patients over the age of 75 are more likely to be diagnosed with Alzheimer’s than younger individuals
- **H2:** Lower MMSE (Mini-Mental State Exam) scores and higher CDR (Clinical Dementia Rating) scores are strong indicators of Alzheimer’s diagnosis
- **H3:** Patients who report higher physical activity and better diet quality show lower risk levels for Alzheimer’s disease

## Brief Annotation

**1. Which challenge would you like to address?**  
We want to solve the challenge of detecting Alzheimer’s disease early by analyzing patient data such as age, gender, health history, and lifestyle habits. Our goal is to use data to find patterns that show who might be at higher risk, so healthcare professionals can act sooner and provide the right care.
    
**2. Why is this challenge an important or interesting research goal?**  
Alzheimer’s is a serious illness that affects memory and daily life, and it worsens over time. Early detection is key, but it’s not always easy. If we can use data to spot early warning signs, doctors can respond faster, which can help improve quality of life and slow the progression of the disease.    

**3. What is the expected solution your project would provide?**  
We plan to build a machine learning model that uses real patient data to predict a person’s risk of Alzheimer’s. The results will be shown in a simple, visual dashboard that helps doctors and nurses quickly understand the predictions and which factors matter most.
    
**4. What would be the positive impact of the solution, and which category of users could benefit from it?**  
Our solution can support doctors, nurses, and caregivers by giving them better tools to detect Alzheimer’s earlier. It could be used in clinics, hospitals, or memory care units to make smarter decisions, save time, and improve care for patients who need it most.

---

## Streamlit Application 
To run the Streamlit dashboard locally, follow these steps:

1. **Install Python**  
   Download and install Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Clone the repository and navigate to the folder

2. **Clone the Repository**
   ```bash
   git clone git@github.com:FeliciaFavrholdt/BI_EXAM.git
   cd Streamlit_app
   ```

Install Required Packages In the terminal, run the following command to install all necessary packages:

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

Once dependencies are installed, launch the app with:

4. **Start the App**
   ```bash
   streamlit run app.py
   ```

This will open the Streamlit web application locally in your browser.
