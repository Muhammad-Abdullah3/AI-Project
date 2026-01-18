from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import uvicorn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import StudentProfile, PredictionResult
from cleaning_pipeline import CleaningPipeline, AdvisingAgent

app = FastAPI(title="Smart Advisory Portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = "d:/AI Project"
SVM_MODEL_PATH = os.path.join(BASE_DIR, "student_performance_svm_model.pkl")
ASSISTANCE_MODEL_PATH = os.path.join(BASE_DIR, "student_assistance_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

# Load Models
try:
    performance_model = joblib.load(SVM_MODEL_PATH)
    print("Performance SVM model loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    performance_model = None

pipeline = CleaningPipeline()
advising_agent = AdvisingAgent(api_key="AIzaSyD4TD_rjCzMjDQIQSn8Kt7OEUWzYf2oqeg")

@app.get("/")
async def root():
    return {"message": "EduVision API is running", "models_loaded": performance_model is not None}

@app.post("/predict/student", response_model=PredictionResult)
async def predict_student(student: StudentProfile):
    if not performance_model:
        raise HTTPException(status_code=500, detail="Core performance model not loaded.")
    
    input_dict = student.dict()
    df_input = pipeline.preprocess_single_student(input_dict)
    
    # 1. Predict Performance (SVM)
    perf_pred = performance_model.predict(df_input)[0]
    perf_map = {0: 'Average', 1: 'Good', 2: 'Weak'}
    perf_label = perf_map.get(perf_pred, "Unknown")
    
    # 2. Logic-based Advice Induction (Engagement Score)
    engagement_score = advising_agent.calculate_metrics(input_dict)
    advice_label = "Needs Advice" if (perf_label == "Weak" or engagement_score < 6) else "No Immediate Advice Required"
    
    # 3. Generate Analytical AI Advice
    advice = advising_agent.generate_advice(input_dict, perf_label)
    
    return {
        "Performance_Class": perf_label,
        "Needs_Advice": advice_label,
        "Advice": advice
    }

@app.get("/analytics/summary")
async def get_analytics(province: str = None, gender: str = None, performance: str = None, income: str = None):
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = pd.read_csv(DATASET_PATH)
    
    # Apply filters
    if province and province != "all": df = df[df['Province'] == province]
    if gender and gender != "all": df = df[df['Gender'] == gender]
    if performance and performance != "all": df = df[df['Performance_Class'] == performance]
    if income and income != "all": df = df[df['Family_Income_PKR'] == income]
    
    # Deterministic calculation for analytics
    def calc_needs_advice(row):
        engagement = advising_agent.calculate_metrics(row.to_dict())
        return "Needs Advice" if (row['Performance_Class'] == "Weak" or engagement < 6) else "No Advice"
    
    if not df.empty:
        df['Advice_Need'] = df.apply(calc_needs_advice, axis=1)
    
    summary = {
        "total_students": len(df),
        "avg_cgpa": float(df['CGPA'].mean()) if not df.empty else 0,
        "avg_attendance": float(df['Attendance_Percentage'].mean()) if not df.empty else 0,
        "performance_distribution": df['Performance_Class'].value_counts().to_dict() if not df.empty else {},
        "advice_distribution": df['Advice_Need'].value_counts().to_dict() if not df.empty else {},
        "gender_distribution": df['Gender'].value_counts().to_dict() if not df.empty else {},
        "income_distribution": df['Family_Income_PKR'].value_counts().to_dict() if not df.empty else {},
        "province_distribution": df['Province'].value_counts().to_dict() if not df.empty else {},
        "internet_distribution": df['Internet_Access'].value_counts().to_dict() if not df.empty else {},
        "device_distribution": df['Device_Available'].value_counts().to_dict() if not df.empty else {},
        "motivation_distribution": df['Motivation_Level'].value_counts().to_dict() if not df.empty else {},
        "cgpa_by_motivation": df.groupby('Motivation_Level')['CGPA'].mean().to_dict() if not df.empty else {},
        "cgpa_by_province": df.groupby('Province')['CGPA'].mean().to_dict() if not df.empty else {},
        "marks_distribution": {
            "CGPA": df['CGPA'].tolist() if not df.empty else []
        }
    }
    return summary

@app.post("/upload/class")
async def upload_class(file: UploadFile = File(...)):
    if not performance_model:
        raise HTTPException(status_code=500, detail="Performance model not loaded")
        
    filename = file.filename.lower()
    if filename.endswith('.csv'): df_batch = pd.read_csv(file.file)
    elif filename.endswith(('.xlsx', '.xls')): df_batch = pd.read_excel(file.file)
    else: raise HTTPException(status_code=400, detail="Unsupported format")
            
    results = []
    perf_map = {0: 'Average', 1: 'Good', 2: 'Weak'}
    
    try:
        for idx in range(len(df_batch)):
            row_dict = df_batch.iloc[idx].to_dict()
            df_row = pipeline.preprocess_single_student(row_dict)
            
            perf_pred = performance_model.predict(df_row)[0]
            perf_label = perf_map.get(perf_pred, "Unknown")
            
            # Advice Logic
            engagement = advising_agent.calculate_metrics(row_dict)
            advice_label = "Needs Advice" if (perf_label == "Weak" or engagement < 6) else "No Advice"
            
            # AI Advice
            advice_text = advising_agent.generate_advice(row_dict, perf_label)
            
            results.append({
                "Student_ID": row_dict.get("Student_ID", idx + 1),
                "Name": row_dict.get("Name", f"Student {idx + 1}"),
                "Age": row_dict.get("Age", "N/A"),
                "Province": row_dict.get("Province", "N/A"),
                "CGPA": row_dict.get("CGPA", "N/A"),
                "Performance_Class": perf_label,
                "Needs_Advice": advice_label,
                "Advice": advice_text
            })
        return {"results": results, "total_processed": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
