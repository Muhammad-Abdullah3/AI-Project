from pydantic import BaseModel
from typing import Optional

class StudentProfile(BaseModel):
    Gender: str
    District: str
    Schooling_Type: str
    Internet_Access: str
    Parental_Education_Level: str
    Scholarship_Status: str
    Extracurricular_Participation: Optional[str] = "None"
    Residence_Type: str
    School_Medium: str
    Education_System: str
    Transport_Mode: str
    Attendance_Percentage: float
    Study_Hours_per_Week: float
    Assignment_Average: float
    Quiz_Average: float
    Mid_Marks: float
    # Add other columns if used by the model (e.g., Matric_Percentage if we generated it back? 
    # In my synthesis script I mainly used Attendance/StudyHours for the synthetic target, 
    # but the model was trained on ALL X features.
    
    # We must ensure the input matches the columns expected by the preprocessor.
    # The 'Extra_Expanded' dataset had many columns.
    # Ideally, we should list all used Features.
    # For robust MVP, we can allow extra fields or just match the CSV columns.

class PredictionResult(BaseModel):
    Risk_Status: str  # "Pass" or "Fail"
    Performance_Trend: str # "Improving" or "Declining"
    Engagement_Score: float
    Advice: str = ""
