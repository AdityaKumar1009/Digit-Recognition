import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained models
logreg = joblib.load("/Users/crazeformarvel/Desktop/Saqib Project/Blood Glucose/logistic_regression_diabetes_model.pkl")

rf = joblib.load("/Users/crazeformarvel/Desktop/Saqib Project/Blood Glucose/random_forest_diabetes_model.pkl")

xgb = joblib.load("/Users/crazeformarvel/Desktop/Saqib Project/Blood Glucose/xgboost_diabetes_model.pkl")

# Define features used in training
features = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]

# Get user input
def get_user_input():
    user_data = {}
    for feature in features:
        user_data[feature] = float(input(f"Enter {feature}: "))
    return pd.DataFrame([user_data])

# Get user input
user_df = get_user_input()

# Load or fit scaler if not available
try:
    scaler = joblib.load("scaler.pkl")
    user_df_scaled = scaler.transform(user_df)
except FileNotFoundError:
    print("Scaler not found. Proceeding without scaling.")
    user_df_scaled = user_df

# Make predictions
print("Predictions:")
print(f"Logistic Regression Prediction: {logreg.predict(user_df_scaled)[0]}")
print(f"Random Forest Prediction: {rf.predict(user_df)[0]}")
print(f"XGBoost Prediction: {xgb.predict(user_df)[0]}")
