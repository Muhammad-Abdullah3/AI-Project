from pydantic import BaseModel
from typing import Optional

class StudentProfile(BaseModel):
    Age: int
    Gender: str
    City: str
    Province: str
    CGPA: float
    Family_Income_PKR: str
    Parents_Education: str
    Study_Hours_Per_Week: int
    Attendance_Percentage: int
    Internet_Access: str
    Private_Tuition: str
    Device_Available: Optional[str] = "None"
    Electricity_Availability: str
    School_Type: str
    Medium_of_Instruction: str
    Distance_to_Institute_km: float
    Transport_Mode: str
    Parental_Support_Level: str
    Health_Issues: Optional[str] = "None"
    Part_Time_Job: str
    Extra_Curricular: Optional[str] = "None"
    Motivation_Level: str

class PredictionResult(BaseModel):
    Performance_Class: str  # Weak, Average, Good
    Needs_Advice: str  # Needs Advice, Doesn't Need Advice
    Advice: str = ""
