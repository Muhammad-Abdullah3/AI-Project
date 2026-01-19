from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Data
df = pd.read_csv('d:/AI Project/dataset.csv')

# 1. Advice Need Logic (Target for Advice Model)
def determine_advice_need(row):
    score = 0
    
    # Socio-economic factors
    if row['Family_Income_PKR'] == 'Low (<30k)': score += 3
    elif row['Family_Income_PKR'] == 'Lower-Middle (30k-60k)': score += 2
    if row['Internet_Access'] == 'No': score += 1
    if row['Device_Available'] == 'None': score += 2
    elif row['Device_Available'] == 'Mobile': score += 1
    if row['Part_Time_Job'] == 'Yes': score += 1
    if row['Electricity_Availability'] == 'Frequent Outages': score += 1
    
    # Academic performance factors
    if row['Attendance_Percentage'] < 70: score += 2
    elif row['Attendance_Percentage'] < 80: score += 1
    
    if row['Study_Hours_Per_Week'] < 10: score += 2
    elif row['Study_Hours_Per_Week'] < 15: score += 1
    
    if row['CGPA'] < 2.0: score += 2
    elif row['CGPA'] < 2.5: score += 1
    
    if row['Motivation_Level'] == 'Low': score += 1
    
    return 'Needs Advice' if score >= 5 else "Doesn't Need Advice"

df['Advice_Status'] = df.apply(determine_advice_need, axis=1)

# Preprocessing Setup
X = df.drop(columns=['Student_ID', 'Performance_Class', 'Advice_Status'], errors='ignore')
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- TRAIN SVM (Performance Classification) ---
y_perf = LabelEncoder().fit_transform(df['Performance_Class']) # Average=0, Good=1, Weak=2
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_perf, test_size=0.2, random_state=42)

svm_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(probability=True, kernel='linear', C=1))])
svm_pipe.fit(X_train_p, y_train_p)
svm_accuracy = accuracy_score(y_test_p, svm_pipe.predict(X_test_p))
print("SVM Accuracy: ", svm_accuracy)
joblib.dump(svm_pipe, 'd:/AI Project/student_performance_svm_model.pkl')
print("SVM Model Saved.")

