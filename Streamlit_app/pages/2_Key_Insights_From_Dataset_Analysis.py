import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import plotly.figure_factory as ff
from sklearn.metrics import classification_report

st.set_page_config(page_title="Key Insights and Analysis Results", layout="wide")

st.title("Key Insights From Dataset Analysis")

# Introduction
st.write("""
On this page, we summarize the key insights from our analysis and the results of our predictive model for Alzheimer's disease.
""")

# Feature Selection
st.header("Selected Features")

st.write("""
After analyzing our cleaned dataset in Notebook 05, we selected 11 features for our predictive model. We chose these features because they are easy to collect in a clinical setting and have shown significant correlation with Alzheimer's disease risk. For healthcare professionals, these features can be collected through patient interviews, questionnaires, and basic health assessments and are generally available in most healthcare systems which makes our application easy to implement in practice. The features capture important information across different patient categories: demographics, lifestyle, medical history, and cognitive function and are listed below:
""")

st.markdown("""
- **Age:** Patient's age (mostly between 60 and 90)
- **Gender:** 0 = Male, 1 = Female
- **BMI:** Body Mass Index (weight relative to height)
- **Physical Activity:** Activity level score (higher is better)
- **Sleep Quality:** Sleep health score (higher is better)
- **Functional Assessment:** Daily functioning ability score
- **ADL (Activities of Daily Living):** Measures independence in daily tasks
- **MMSE Score:** Mini-Mental State Examination (cognitive test, 0–30)
- **Family History of Alzheimer's:** 0 = No, 1 = Yes (genetic risk factor)
- **Cardiovascular Disease:** 0 = No, 1 = Yes (heart condition history)
- **Depression:** 0 = No, 1 = Yes (mental health factor)
""")

# Feature Importance Visualization
st.header("Feature Importance")

features = ["MMSE Score", "Age", "ADL", "Functional Assessment", 
            "Physical Activity", "Sleep Quality", "BMI"]
importance = [0.28, 0.22, 0.16, 0.10, 0.08, 0.07, 0.05]

df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

fig = px.bar(df, x="Feature", y="Importance", color="Importance",
             color_continuous_scale="Blues", text="Importance")
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(yaxis_range=[0, 0.3])

st.plotly_chart(fig, use_container_width=True)

# Model Performance Summary
st.header("Model Performance")

st.write("""
In our analysis (Notebook 05), we tested different machine learning models on the features above. We observed that the best model for Alzheimer's Disease Prediction with our chosen features is a Random Forest Classifier. The model achieved the following performance metrics:

Note: The outcome of our predictive model should not stand alone, but rather be used as a tool to assist healthcare professionals in making informed decisions about patient care. We recommend further evaluation of clinical lab data such as blood tests and other medical tests to improve final predictions.
""")

# Show KPIs using Streamlit columns
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "90%")
col2.metric("Precision", "89%")
col3.metric("Recall", "92%")
col4.metric("F1-Score", "90%")

st.write("""
The model shows strong and balanced performance across all metrics, indicating that it can make reliable predictions.
""")

# Load saved model files 
rf_model = joblib.load('models/random_forest_model.pkl')
X_test, y_test = joblib.load('models/test_data.pkl')
rf_pred = joblib.load('models/rf_predictions.pkl')
rf_cm = joblib.load('models/rf_confusion_matrix.pkl')
features = joblib.load('models/features.pkl')

# Confusion Matrix
st.subheader("Confusion Matrix")

z = rf_cm
x = ["No Alzheimer's", "Alzheimer's"]
y = ["No Alzheimer's", "Alzheimer's"]

fig_cm = ff.create_annotated_heatmap(
    z, x=x, y=y, colorscale="Blues", showscale=True
)
st.plotly_chart(fig_cm, use_container_width=True)

# Classification Report
st.subheader("Classification Report")

report_dict = classification_report(y_test, rf_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)
st.dataframe(report_df)

st.write("""
The confusion matrix and classification report confirm that the model performs well on both classes.
""")

# Model Comparison Section
st.header("AUC Comparison Between Models")

st.write("""
We also compared models using AUC (Area Under Curve), which gives an overall measure of model performance. 
Higher AUC means better overall classification capability.
""")

# Load and display AUC comparison image
st.image("plots/06_model_auc_comparison.png", use_column_width=True, caption="Notebook 06 – AUC score comparison across models used for Alzheimer's prediction.")

st.write("""
The Random Forest model performed best across all metrics and was therefore selected as the final model for deployment.
""")

st.image("plots/06_model_comparison_metrics.png", use_column_width=True, caption="Notebook 06 – Model comparison metrics for Alzheimer's prediction.")


# 

# Clustering Summary
st.header("Cluster Summary")

cluster_df = pd.read_csv('data/cluster_summary.csv')
st.dataframe(cluster_df)

st.image("plots/06_pca_cluster_visualization.png", use_column_width=True, caption="")






## Final Interpretation
st.header("Final Interpretation")
st.write("""
The Random Forest model achieved the best classification performance, identifying key features influencing Alzheimer's diagnosis. Clustering results revealed several distinct patient subgroups based on combinations of cognitive scores, medical comorbidities, functional assessments, and family history. Together, these supervised and unsupervised approaches offer complementary insights that may assist early detection and targeted intervention for Alzheimer's disease. Below we answer out questions:
- **Can we use patient data to build accurate machine learning models for Alzheimer’s prediction?**

Yes. We used patient data containing cognitive assessments, functional scores, lifestyle information, and genetic markers to train multiple machine learning models. After evaluating the models, we observe that the Random Forest model performs particularly well, achieving a strong AUC score of approximately 0.87. Based on these results, we conclude that patient data can indeed be used to accurately predict the risk of Alzheimer’s. This approach may help in early identification of high-risk individuals and support clinical decision-making.

- **Which features are most helpful in predicting Alzheimer’s?**

By analyzing feature importance from our Random Forest model, we observe that Functional Assessment, Activities of Daily Living (ADL), and Mini-Mental State Examination (MMSE) are the strongest predictors of Alzheimer’s. These features provide valuable information about a person’s daily functioning and cognitive health. We also see that Body Mass Index (BMI), Physical Activity, Sleep Quality, and Age contribute to the model’s predictions, although to a lesser extent. In addition, we notice that factors such as Gender, Family History of Alzheimer’s, Cardiovascular Disease, and Depression have smaller but still relevant influence on the overall prediction.

- **Which machine learning model performs best?**
  
In our analysis, we tested several machine learning models. We observe that the Random Forest model consistently outperforms the others, achieving the highest AUC score of around 0.87. Based on these results, we conclude that Random Forest is the most accurate and reliable model for predicting Alzheimer’s risk in our study.
         """)

# Footer
st.caption("CPH Business 2025 — Group 4 — Business Intelligence Exam Project")
