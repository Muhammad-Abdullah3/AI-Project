import pandas as pd
import numpy as np
import os
class AdvisingAgent:
    def __init__(self, api_key: str = None):
        # API completely removed as requested. Standardizing to local engine.
        pass

    def calculate_metrics(self, row):
        """Calculates Engagement Score (0-10) based on academic participation."""
        engagement = 0
        try:
            attendance = float(row.get('Attendance_Percentage', 0))
            study_hours = float(row.get('Study_Hours_Per_Week', 0))
        except (ValueError, TypeError):
            attendance, study_hours = 0, 0
        
        # Attendance component (max 4)
        if attendance >= 90: engagement += 4
        elif attendance >= 75: engagement += 3
        elif attendance >= 60: engagement += 2
        
        # Study hours component (max 4)
        if study_hours >= 20: engagement += 4
        elif study_hours >= 15: engagement += 3
        elif study_hours >= 10: engagement += 2
        
        # Resource components (max 2)
        if row.get('Internet_Access') == 'Yes': engagement += 1
        if row.get('Device_Available') not in ['None', 'Mobile', 'Unknown']: engagement += 1
        
        return engagement

    def generate_advice(self, student_data: dict, performance_class: str):
        """
        Independent Rule-Based Advice Engine.
        Generates structured, analytical plans without external API calls.
        """
        engagement = self.calculate_metrics(student_data)
        hours = student_data.get('Study_Hours_Per_Week', '0')
        internet = student_data.get('Internet_Access', 'No')
        
        advice_blocks = []

        # 1. CORE ANALYSIS SECTION
        analysis = f"Analytical Summary: The student is maintaining a {performance_class} standing with an Engagement Score of {engagement}/10. "
        if performance_class == "Weak" or engagement < 6:
            analysis += "Critical academic drift detected requiring immediate structural intervention."
        else:
            analysis += "Consistent academic parameters observed. Maintaining current trajectory is recommended."
        advice_blocks.append(analysis)

        # 2. ACTIONABLE PLAN (Requirement Based)
        plan = ["Intervention Plan:"]
        
        # Access First
        if internet == 'No':
            plan.append("- Resource Access: Grant priority 1st-tier access to campus computer labs and offline digital repositories.")
        
        # Academic Load
        if float(hours) < 12:
            plan.append(f"- Time Management: Structured increase in study volume required. Target: 15h minimum (Current: {hours}h).")
        
        # Performance Specific
        if performance_class == "Weak":
            plan.append("- Pedagogical Support: Enroll in mandatory core-subject reinforcement workshops and weekly progress monitoring.")
        elif performance_class == "Average":
            plan.append("- Peer Integration: Suggested enrollment in collaborative study groups to stabilize performance variables.")
        
        if len(plan) == 1: # No issues found
            plan.append("- Maintenance: Continue current study schedule. Encourage leadership in peer tutoring groups.")

        advice_blocks.append("\n".join(plan))

        return "\n\n".join(advice_blocks)

class CleaningPipeline:
    """
    Handles data structure consistency for the backend.
    """
    def preprocess_single_student(self, student_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([student_data])
        # Ensure correct data types for scikit-learn pipeline
        # Numerics
        num_cols = ['Age', 'CGPA', 'Study_Hours_Per_Week', 'Attendance_Percentage', 'Distance_to_Institute_km']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Categoricals (ensure strings)
        cat_cols = [
            'Gender', 'City', 'Province', 'Family_Income_PKR', 'Parents_Education', 
            'Internet_Access', 'Private_Tuition', 'Device_Available', 
            'Electricity_Availability', 'School_Type', 'Medium_of_Instruction', 
            'Transport_Mode', 'Parental_Support_Level', 'Health_Issues', 
            'Part_Time_Job', 'Extra_Curricular', 'Motivation_Level'
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', 'Unknown')
                
        return df
