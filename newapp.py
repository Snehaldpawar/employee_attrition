import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from PIL import Image

# Load models and preprocessing objects
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("👥 Employee Attrition Prediction App")

# Sidebar menu
menu = st.sidebar.selectbox("Navigate", ["Single Prediction", "Batch Prediction", "Model Metrics", "Visualizations"])

# 🔁 Helper functions
def preprocess_input(df):
    for col in label_encoders:
        if col in df.columns:
            le = label_encoders[col]
            known_classes = set(le.classes_)
            incoming_classes = set(df[col].unique())
            if not incoming_classes.issubset(known_classes):
                unseen = incoming_classes - known_classes
                raise ValueError(f"Column '{col}' contains unseen labels: {unseen}")
            df[col] = le.transform(df[col])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return scaler.transform(df)

def download_link(df, filename='predictions.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Predictions</a>'

# 🌟 SINGLE PREDICTION
if menu == "Single Prediction":
    model_choice = st.selectbox("Choose Model", ["XGBoost", "Decision Tree"])
    with st.form("employee_form"):
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        over_time = st.selectbox("OverTime", ["Yes", "No"])
        distance = st.slider("Distance From Home", 1, 30, 10)
        income = st.slider("Monthly Income", 1000, 20000, 5000)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            input_data = pd.DataFrame({
                "Age": [age],
                "BusinessTravel": [business_travel],
                "Department": [department],
                "DistanceFromHome": [distance],
                "Gender": [gender],
                "JobRole": [job_role],
                "MaritalStatus": [marital_status],
                "OverTime": [over_time],
                "MonthlyIncome": [income]
            })
            input_scaled = preprocess_input(input_data)
            model = xgb_model if model_choice == "XGBoost" else dt_model
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.subheader("Prediction Result")
            st.write("🔴 Attrition: **Yes**" if prediction == 1 else "🟢 Attrition: **No**")
            st.write(f"📊 Probability of Attrition: **{probability:.2%}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# 📁 BATCH PREDICTION
elif menu == "Batch Prediction":
    st.subheader("Batch Prediction using CSV")
    uploaded_file = st.file_uploader("Upload employee data CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        model_choice = st.selectbox("Choose Model", ["XGBoost", "Decision Tree"])
        try:
            df_processed = df.copy()
            input_scaled = preprocess_input(df_processed)
            model = xgb_model if model_choice == "XGBoost" else dt_model
            preds = model.predict(input_scaled)
            probs = model.predict_proba(input_scaled)[:, 1]

            df['Attrition Prediction'] = np.where(preds == 1, "Yes", "No")
            df['Attrition Probability'] = np.round(probs * 100, 2)

            st.success("Predictions complete!")
            st.dataframe(df)

            st.markdown(download_link(df), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# 📈 MODEL METRICS
elif menu == "Model Metrics":
    st.subheader("📊 Model Evaluation Results")
    st.image("confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
    st.image("roc_curve.png", caption="ROC Curve", use_column_width=True)
    st.image("pr_curve.png", caption="Precision-Recall Curve", use_column_width=True)
    st.image("feature_importance.png", caption="Feature Importance", use_column_width=True)

# 📊 VISUALIZATION
elif menu == "Visualizations":
    st.subheader("📊 Exploratory Data Analysis")

    st.image("attrition_distribution.png", caption="Attrition Distribution", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("Age_boxplot.png", caption="Age vs Attrition")
        st.image("MonthlyIncome_boxplot.png", caption="Monthly Income vs Attrition")

    with col2:
        st.image("JobRole_attrition_countplot.png", caption="Job Role vs Attrition")
        st.image("OverTime_attrition_countplot.png", caption="OverTime vs Attrition")