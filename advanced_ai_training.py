import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def synthesize_advanced_logic(df):
    """
    Synthesizes complex behavioral and academic logic to simulate a real student ecosystem.
    """
    np.random.seed(42)
    
    # 1. Feature Engineering: Engagement Score (0-100)
    # Weights: Attendance (0.5), Study Hours (0.3), Extracurriculars (0.2)
    extracurricular_map = {
        'Multiple Activities': 100,
        'Debate Club': 80,
        'Sports': 70,
        'Arts Society': 60,
        'Volunteer Work': 50
    }
    ext_score = df['Extracurricular_Participation'].map(extracurricular_map).fillna(30)
    att_score = df['Attendance_Percentage']
    study_score = (df['Study_Hours_per_Week'] / 20) * 100
    
    df['Engagement_Score'] = (att_score * 0.5 + study_score * 0.3 + ext_score * 0.2)
    
    # 2. Feature Engineering: Resource Access Index (0-100)
    # Weights: Internet (0.4), Transport (0.3), Residence (0.3)
    internet_map = {'Yes': 100, 'No': 20}
    transport_map = {'Own Vehicle': 100, 'School Bus': 80, 'Walking': 60, 'Public Transport': 40}
    residence_map = {'Urban': 100, 'Rural': 50}
    
    res_score = (
        df['Internet_Access'].map(internet_map).fillna(50) * 0.4 +
        df['Transport_Mode'].map(transport_map).fillna(50) * 0.3 +
        df['Residence_Type'].map(residence_map).fillna(50) * 0.3
    )
    df['Resource_Index'] = res_score
    
    # 3. Feature Engineering: Performance Stability (Standard Deviation of scores)
    scores = df[['Assignment_Average', 'Quiz_Average', 'Mid_Marks']]
    df['Academic_Stability'] = 100 - scores.std(axis=1) * 2 # Lower std = higher stability
    
    # --- LABELS SYNTHESIS ---
    
    # Label 1: Final_Exam_Status (Holistic Risk)
    # Success depends on Engagement (60%) and Resources (40%) + Noise
    final_score = (df['Engagement_Score'] * 0.6 + df['Resource_Index'] * 0.4) + np.random.normal(0, 5, len(df))
    df['Final_Marks'] = np.clip(final_score, 0, 100)
    df['Final_Exam_Status'] = df['Final_Marks'].apply(lambda x: 'Pass' if x >= 55 else 'Fail')
    
    # Label 2: Performance_Trend (Trend Dynamics)
    # Improving if Mid_Marks > (Assignment + Quiz)/2
    recent_delta = df['Mid_Marks'] - (df['Assignment_Average'] + df['Quiz_Average']) / 2
    df['Performance_Trend'] = recent_delta.apply(lambda x: 'Improving' if x > 2 else 'Declining')
    
    return df

def train_models():
    print("Loading and synthesizing data...")
    df = pd.read_csv("Extra_Expanded_Consistent_Student_Demographics.csv")
    df = synthesize_advanced_logic(df)
    
    # Define features
    categorical_features = [
        'Gender', 'District', 'Schooling_Type', 'Internet_Access',
        'Parental_Education_Level', 'Scholarship_Status',
        'Extracurricular_Participation', 'Residence_Type',
        'School_Medium', 'Education_System', 'Transport_Mode'
    ]
    numerical_features = [
        'Attendance_Percentage', 'Study_Hours_per_Week',
        'Assignment_Average', 'Quiz_Average', 'Mid_Marks',
        'Engagement_Score', 'Resource_Index', 'Academic_Stability'
    ]
    
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ]
    )
    
    # --- TARGET 1: TREND DYNAMICS (Logistic Regression) ---
    print("\nTraining Trend Dynamics Model (Logistic Regression)...")
    X = df[numerical_features + categorical_features]
    y_trend = df['Performance_Trend'].map({'Improving': 1, 'Declining': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_trend, test_size=0.2, random_state=42)
    
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    param_grid_lr = {'classifier__C': [0.1, 1, 10]}
    grid_lr = GridSearchCV(lr_pipeline, param_grid_lr, cv=5)
    grid_lr.fit(X_train, y_train)
    
    print(f"LR Best Params: {grid_lr.best_params_}")
    print(f"LR Accuracy: {grid_lr.score(X_test, y_test):.4f}")
    
    # --- TARGET 2: HOLISTIC RISK (Random Forest) ---
    print("\nTraining Holistic Risk Model (Random Forest)...")
    y_risk = df['Final_Exam_Status'].map({'Pass': 1, 'Fail': 0})
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk, test_size=0.2, random_state=42)
    
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    }
    grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5)
    grid_rf.fit(X_train_r, y_train_r)
    
    print(f"RF Best Params: {grid_rf.best_params_}")
    print(f"RF Accuracy: {grid_rf.score(X_test_r, y_test_r):.4f}")
    
    # --- SAVE MODELS ---
    os.makedirs('models', exist_ok=True)
    with open('models/risk_model_lr.pkl', 'wb') as f:
        pickle.dump(grid_lr.best_estimator_, f)
    with open('models/advising_model_rf.pkl', 'wb') as f:
        pickle.dump(grid_rf.best_estimator_, f)
        
    # --- CALCULATE & SAVE EXHAUSTIVE METRICS ---
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    import json
    
    # LR - Trend Dynamics
    y_pred_lr = grid_lr.predict(X_test)
    precision_lr, recall_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='binary')
    cm_lr = confusion_matrix(y_test, y_pred_lr).tolist()
    
    # RF - Holistic Risk
    y_pred_rf = grid_rf.predict(X_test_r)
    precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test_r, y_pred_rf, average='binary')
    cm_rf = confusion_matrix(y_test_r, y_pred_rf).tolist()
    
    metrics_data = {
        "trend_model_lr": {
            "accuracy": grid_lr.score(X_test, y_test),
            "precision": float(precision_lr),
            "recall": float(recall_lr),
            "f1": float(f1_lr),
            "confusion_matrix": cm_lr
        },
        "risk_model_rf": {
            "accuracy": grid_rf.score(X_test_r, y_test_r),
            "precision": float(precision_rf),
            "recall": float(recall_rf),
            "f1": float(f1_rf),
            "confusion_matrix": cm_rf
        }
    }
    with open('models/metrics_report.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # --- SAVE COHORT STATS FOR ADVISING ENGINE ---
    successful_cohort = df[df['Final_Exam_Status'] == 'Pass']
    cohort_stats = {
        'numerical_means': successful_cohort[numerical_features].mean().to_dict(),
        'categorical_modes': successful_cohort[categorical_features].mode().iloc[0].to_dict(),
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    with open('models/cohort_stats.pkl', 'wb') as f:
        pickle.dump(cohort_stats, f)
        
    print("\nAll models and cohort statistics saved successfully.")

if __name__ == "__main__":
    train_models()
