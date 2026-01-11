from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import os
try:
    from backend.schemas import StudentProfile, PredictionResult
    from backend.cleaning_pipeline import CleaningPipeline
except ImportError:
    from schemas import StudentProfile, PredictionResult
    from cleaning_pipeline import CleaningPipeline

app = FastAPI(title="Student Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_DIR = "d:/AI Project/models"
try:
    with open(f"{MODEL_DIR}/risk_model_lr.pkl", "rb") as f:
        risk_model = pickle.load(f)
    with open(f"{MODEL_DIR}/advising_model_rf.pkl", "rb") as f:
        advising_model = pickle.load(f)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    risk_model = None
    advising_model = None

pipeline = CleaningPipeline()

# Define feature lists (must match advanced_ai_training.py)
categorical_features_list = [
    'Gender', 'District', 'Schooling_Type', 'Internet_Access',
    'Parental_Education_Level', 'Scholarship_Status',
    'Extracurricular_Participation', 'Residence_Type',
    'School_Medium', 'Education_System', 'Transport_Mode'
]
numerical_features_list = [
    'Attendance_Percentage', 'Study_Hours_per_Week',
    'Assignment_Average', 'Quiz_Average', 'Mid_Marks',
    'Engagement_Score', 'Resource_Index', 'Academic_Stability'
]

@app.post("/token")
async def login(password: str):
    if password == "admin123":  # Simple auth for prototype
        return {"access_token": "fake-jwt-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/predict/student", response_model=PredictionResult)
async def predict_student(student: StudentProfile):
    if not risk_model or not advising_model:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # 1. Engineer features via diagnostic pipeline
    input_dict = student.dict()
    df_engineered = pipeline.preprocess_single_student(input_dict)
    
    # 2. Extract internal scores for response
    engagement_score = float(df_engineered['Engagement_Score'].iloc[0])
    
    # 3. Predict Performance Trend (Improving/Declining) using Trend Dynamics Model (LR)
    # The models expect the same features training used (numerical + categorical)
    # The SKlearn pipeline (preprocessor) handles the transformation.
    # We need to ensure we pass ONLY the columns used in training.
    
    # In advanced_ai_training.py:
    # numerical_features = ['Attendance_Percentage', 'Study_Hours_per_Week', 'Assignment_Average', 'Quiz_Average', 'Mid_Marks', 'Engagement_Score', 'Resource_Index', 'Academic_Stability']
    # categorical_features = ['Gender', 'District', 'Schooling_Type', 'Internet_Access', 'Parental_Education_Level', 'Scholarship_Status', 'Extracurricular_Participation', 'Residence_Type', 'School_Medium', 'Education_System', 'Transport_Mode']
    
    features = df_engineered[numerical_features_list + categorical_features_list]
    
    trend_pred = risk_model.predict(features)[0] # 1=Improving, 0=Declining
    trend_label = "Improving" if trend_pred == 1 else "Declining"
    
    # 4. Predict Holistic Risk (Pass/Fail) using Advising Model (RF)
    risk_pred = advising_model.predict(features)[0] # 1=Pass, 0=Fail
    risk_status_label = "Pass" if risk_pred == 1 else "Fail"
    
    # 5. Generate Advanced Teacher Advice
    advice = pipeline.agent(input_dict, trend_label=trend_label)
    
    return {
        "Risk_Status": risk_status_label,
        "Performance_Trend": trend_label,
        "Engagement_Score": engagement_score,
        "Advice": advice
    }

@app.get("/analytics/summary")
async def get_analytics():
    # In a real app, this would query the database.
    # Here we read the static CSV for the dashboard demo.
    df = pd.read_csv("d:/AI Project/Extra_Expanded_Consistent_Student_Demographics.csv")
    
    # Engineer features for the whole dataframe to enable advanced analytics
    df = pipeline.engineer_features(df)
    
    # Advanced Insight Data Preparation
    # 1. Engagement vs Stability (Scatter)
    eng_stab = df[['Engagement_Score', 'Academic_Stability', 'Final_Exam_Status']].to_dict(orient='records')
    
    # 2. Attendance Impact (Attendance vs Final Marks)
    att_impact = df[['Attendance_Percentage', 'Final_Marks', 'Gender']].to_dict(orient='records')
    
    # 3. Study Hours Impact
    study_impact = df[['Study_Hours_per_Week', 'Mid_Marks', 'Final_Marks']].to_dict(orient='records')
    
    # 4. Parental Education vs Pass Rate
    parent_pass_rate = df.groupby('Parental_Education_Level')['Final_Exam_Status'].apply(
        lambda x: (x == 'Pass').mean() * 100
    ).round(2).to_dict()
    
    # 5. Performance Trend Distribution (Predictive Insight)
    if risk_model:
        # Align features
        feat_df = df[numerical_features_list + categorical_features_list]
        trends = risk_model.predict(feat_df)
        trend_counts = pd.Series(trends).map({1: 'Improving', 0: 'Declining'}).value_counts().to_dict()
    else:
        trend_counts = {"Unknown": len(df)}

    summary = {
        "total_students": len(df),
        "pass_rate": (df['Final_Exam_Status'] == 'Pass').mean() * 100,
        "avg_attendance": df['Attendance_Percentage'].mean(),
        "avg_cgpa": df['CGPA'].mean(),
        "gender_distribution": df['Gender'].value_counts().to_dict(),
        "internet_access": df['Internet_Access'].value_counts().to_dict(),
        "parent_education": df['Parental_Education_Level'].value_counts().to_dict(),
        "marks_distribution": {
            "Mid_Marks": df['Mid_Marks'].tolist(),
            "Final_Marks": df['Final_Marks'].tolist()
        },
        "engagement_vs_stability": eng_stab,
        "attendance_impact": att_impact,
        "study_impact": study_impact,
        "parental_pass_rate": parent_pass_rate,
        "trend_distribution": trend_counts,
        "cgpa_by_district": df.groupby('District')['CGPA'].mean().to_dict()
    }
    return summary

@app.get("/analytics/metrics")
async def get_metrics():
    import json
    metrics_path = "d:/AI Project/models/metrics_report.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Metrics report not found")

@app.post("/upload/class")
async def upload_class(file: UploadFile = File(...)):
    if not risk_model or not advising_model:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    filename = file.filename.lower()
    if filename.endswith('.csv'):
        df_batch = pd.read_csv(file.file)
    elif filename.endswith(('.xlsx', '.xls')):
        df_batch = pd.read_excel(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel.")
    
    # Identify missing features
    all_required = numerical_features_list + categorical_features_list
    missing_features = [f for f in all_required if f not in df_batch.columns]
    
    # Fill missing columns with NaN or defaults for preprocessing
    for f in missing_features:
        if f in numerical_features_list:
            df_batch[f] = 0.0 # Or median
        else:
            df_batch[f] = "Unknown"
            
    results = []
    try:
        for idx in range(len(df_batch)):
            row_dict = df_batch.iloc[idx].to_dict()
            df_row = pipeline.preprocess_single_student(row_dict)
            features = df_row[numerical_features_list + categorical_features_list]
            
            trend_pred = risk_model.predict(features)[0]
            trend_label = "Improving" if trend_pred == 1 else "Declining"
            risk_pred = advising_model.predict(features)[0]
            
            results.append({
                "Student_ID": row_dict.get("Student_ID", idx),
                "Name": row_dict.get("Name", "Unknown"),
                "Risk_Status": "Pass" if risk_pred == 1 else "Fail",
                "Performance_Trend": trend_label,
                "Advice": pipeline.agent(row_dict, trend_label=trend_label, missing_features=missing_features)
            })
        return {"results": results, "missing_features": missing_features}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
