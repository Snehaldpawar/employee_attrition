# employee_attrition
This project is a machine learning-based web application that predicts employee attrition using HR data. Built with Streamlit, it provides an interactive interface for both single and batch predictions. The backend uses XGBoost and Decision Tree models, with data preprocessing and evaluation handled by Scikit-learn. 
# 👥 Employee Attrition Prediction App

This project is a machine learning-based web application that predicts whether an employee is likely to leave an organization. Designed for HR professionals, it helps identify potential attrition risks using predictive analytics.

---

## 📌 Features

* **🔮 Single Prediction**: Enter details of one employee to predict the risk of attrition.
* **📂 Batch Prediction**: Upload a CSV file to predict attrition for multiple employees at once.
* **📊 Model Metrics**: View performance metrics including confusion matrix, ROC curve, and precision-recall curve.
* **📈 EDA Visualizations**: Analyze patterns and trends related to employee attrition.

---

## 🧠 Machine Learning Models

* **XGBoost Classifier**
* **Decision Tree Classifier**

---

## 🛠️ Technologies Used

* **Frontend/UI**: Streamlit
* **Machine Learning**: XGBoost, Scikit-learn
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Model Storage**: Pickle

---

## 📁 Project Structure

```
📦project/
 ┣ 📄train.py                # Model training and evaluation
 ┣ 📄app.py                  # Streamlit app
 ┣ 📄xgb_model.pkl           # Trained XGBoost model
 ┣ 📄dt_model.pkl            # Trained Decision Tree model
 ┣ 📄scaler.pkl              # StandardScaler object
 ┣ 📄label_encoders.pkl      # LabelEncoders for categorical columns
 ┣ 📄feature_names.pkl       # List of feature names
 ┣ 📄Employee-Attrition.csv  # Dataset used for training
 ┣ 📊*.png                   # EDA and model evaluation plots
```

---

##  How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/employee-attrition-predictor.git
   cd employee-attrition-predictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

