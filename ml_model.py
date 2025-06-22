#!/usr/bin/env python
# coding: utf-8

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

# Define reusable preprocessing function for inference

def preprocess_input(df, label_encoders):
    for col in df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            known_classes = set(le.classes_)
            incoming_classes = set(df[col].unique())

            if not incoming_classes.issubset(known_classes):
                unseen = incoming_classes - known_classes
                raise ValueError(f"Column '{col}' contains unseen labels: {unseen}")

            df[col] = le.transform(df[col])
    return df

# Initialize XGBClassifier
model = XGBClassifier(eval_metric='logloss')

# 2. Load Dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\final\Employee-Attrition.csv")
df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], inplace=True)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# 3. EDA (Optional display and plot saving skipped for brevity)

# 4. Preprocessing
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Save feature names
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save encoders and scaler
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 7. Model Training
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# Save models
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))
pickle.dump(dt_model, open("dt_model.pkl", "wb"))

# 8. Evaluation
y_pred = xgb_model.predict(X_test_scaled)
y_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 9. Visualizations (optional plotting skipped in this version)

# 10. Decision Tree Prediction
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("\n--- Model Accuracy Summary ---")
print(f"XGBoost Model Accuracy     : {accuracy_score(y_test, y_pred):.4f}")
print(f"Decision Tree Model Accuracy: {accuracy_dt:.4f}")
print("------------------------------")
