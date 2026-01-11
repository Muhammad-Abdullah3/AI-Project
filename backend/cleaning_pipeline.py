import pandas as pd
import numpy as np
import pickle
import os

class CleaningPipeline:
    def __init__(self):
        self.cohort_stats = None
        self.load_resources()

    def load_resources(self):
        stats_path = "d:/AI Project/models/cohort_stats.pkl"
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                self.cohort_stats = pickle.load(f)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically calculates composite features for the AI intelligence layer.
        """
        # Engagement Score
        extracurricular_map = {
            'Multiple Activities': 100, 'Debate Club': 80, 'Sports': 70, 
            'Arts Society': 60, 'Volunteer Work': 50
        }
        ext_score = df['Extracurricular_Participation'].map(extracurricular_map).fillna(30)
        att_score = df['Attendance_Percentage']
        study_score = (df['Study_Hours_per_Week'] / 20) * 100
        df['Engagement_Score'] = (att_score * 0.5 + study_score * 0.3 + ext_score * 0.2)
        
        # Resource Index
        internet_map = {'Yes': 100, 'No': 20}
        transport_map = {'Own Vehicle': 100, 'School Bus': 80, 'Walking': 60, 'Public Transport': 40}
        residence_map = {'Urban': 100, 'Rural': 50}
        
        df['Resource_Index'] = (
            df['Internet_Access'].map(internet_map).fillna(50) * 0.4 +
            df['Transport_Mode'].map(transport_map).fillna(50) * 0.3 +
            df['Residence_Type'].map(residence_map).fillna(50) * 0.3
        )
        
        # Stability
        scores = df[['Assignment_Average', 'Quiz_Average', 'Mid_Marks']]
        df['Academic_Stability'] = 100 - scores.std(axis=1) * 2
        
        return df

    def preprocess_single_student(self, student_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([student_data])
        df = self.engineer_features(df)
        return df

    def agent(self, student_data: dict, trend_label: str = "Stable", missing_features: list = None):
        """
        Teacher-Oriented Advising Agent:
        Translates statistical gaps into pedagogical advice for teachers.
        """
        if not self.cohort_stats:
            return "Advising engine initialization pending. Recommended: Monitor student's classroom presence."

        # Convert input to DF for feature engineering
        df_input = self.preprocess_single_student(student_data)
        student_profile = df_input.iloc[0]
        
        numerical_means = self.cohort_stats['numerical_means']
        advice_blocks = []

        # Prepend missing features warning
        if missing_features:
            features_str = ", ".join([f.replace('_', ' ') for f in missing_features])
            advice_blocks.append(f"⚠️ [DATA GAP]: The following factors are missing from the student record: {features_str}. This absence may reduce the precision of the diagnostic advice.")
        
        # 1. Dimension: Attendance (Presence)
        if 'Attendance_Percentage' not in (missing_features or []):
            att_gap = numerical_means['Attendance_Percentage'] - student_profile['Attendance_Percentage']
            if att_gap > 15:
                advice_blocks.append("The student's classroom presence is significantly below the baseline. I recommend encouraging them to attend core lectures to better grasp fundamental concepts.")
            elif att_gap > 5:
                advice_blocks.append("Minor attendance gaps detected. A gentle reminder about the importance of consistent class participation could be beneficial.")

        # 2. Dimension: Study Discipline (Hours/Week)
        if 'Study_Hours_per_Week' not in (missing_features or []):
            study_gap = numerical_means['Study_Hours_per_Week'] - student_profile['Study_Hours_per_Week']
            if study_gap > 5:
                advice_blocks.append("Their self-directed study hours are lower than their high-performing peers. Suggesting a structured weekly study timetable might help them manage their workload.")

        # 3. Dimension: Academic Momentum (Trend)
        if trend_label == "Declining":
            advice_blocks.append("I've noticed a downward trend in their recent performance (Mid-marks vs earlier assessments). It might be a good time for a supportive check-in to identify any new challenges they are facing.")
        elif trend_label == "Improving":
            advice_blocks.append("The student is showing positive academic momentum. A word of encouragement on their recent progress would further boost their confidence.")

        # 4. Dimension: Resource Support
        if 'Internet_Access' not in (missing_features or []):
            if student_profile['Internet_Access'] == 'No':
                advice_blocks.append("The student lacks internet access at home. Providing them with offline study materials or ensuring they have priority access to campus computer labs would be highly supportive.")
        
        if 'Transport_Mode' not in (missing_features or []) and 'Attendance_Percentage' not in (missing_features or []):
            if student_profile['Transport_Mode'] == 'Public Transport' and student_profile['Attendance_Percentage'] < 80:
                advice_blocks.append("Transport difficulties may be impacting their attendance. Let's see if we can connect them with university shuttle services or local carpooling options.")

        # 5. Dimension: Composite Engagement
        eng_gap = numerical_means['Engagement_Score'] - student_profile['Engagement_Score']
        if eng_gap > 20:
            advice_blocks.append("Overall classroom engagement is low. Encouraging them to participate in at least one extracurricular society or group project could help them feel more connected to the academic community.")

        if not advice_blocks or (len(advice_blocks) == 1 and missing_features):
             if not advice_blocks:
                return "The student is currently performing exceptionally well. I recommend encouraging them to take on a peer-mentoring role to further refine their leadership skills."
             else:
                advice_blocks.append("Despite missing data, the current metrics suggest the student is maintaining a stable path. Suggest a routine review once full data is available.")
            
        # Return top 4 most critical advice pieces (including warning)
        return " ".join(advice_blocks[:4])
