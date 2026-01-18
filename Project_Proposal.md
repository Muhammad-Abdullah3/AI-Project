# Project Proposal: EduVision - AI-Driven Academic Diagnostic & Advising System

## 1. Title
**EduVision: An AI-Driven Academic Diagnostic and Intelligent Advising System**

## 2. Table of Contents
1. [Title](#1-title)
2. [Introduction](#3-introduction)
3. [Literature Review](#4-literature-review)
4. [Aims & Objectives](#5-aims--objectives)
5. [Research Questions](#6-research-questions)
6. [Methods](#7-methods)
7. [Plan for Analysis](#8-plan-for-analysis)
8. [References](#9-references)
9. [Timeline](#10-timeline)
10. [Budget](#11-budget)

---

## 3. Introduction
In the contemporary educational landscape, the ability to predict student performance and provide timely interventions is paramount. "EduVision" is a proposed AI-driven academic diagnostic and advising system designed to bridge the gap between raw academic data and actionable pedagogical insights. Unlike traditional systems that rely solely on historical grades, EduVision leverages a multi-dimensional approach, analyzing behavioral markers, socio-economic factors, and academic trends to predict student success and momentum shifts.

The core philosophy of this project is to move from *reactive* value judgment (grading) to *proactive* diagnostic support, enabling educators to identify at-risk students before critical failures occur.

## 4. Literature Review
Educational Data Mining (EDM) and Learning Analytics (LA) have traditionally focused on analyzing standardized test scores to predict future performance. However, recent literature suggests that these metrics are often lagging indicators. Research by Baker et al. (2010) and others highlights the significance of "behavioral engagement"—factors such as attendance regularity, study habits, and extracurricular participation—as superior predictors of academic outcomes.

Current systems often fail to integrate these heterogeneous data points (demographic, behavioral, academic) into a unified predictive model. Furthermore, while many systems can predict *failure*, few offer automated, granular *advising* logic. EduVision aims to fill this void by implementing a "Synthesis of Behavioral Markers" approach, using composite features like an **Engagement Score** and **Academic Stability Index** to provide a holistic view of the learner, akin to the "whole child" approach advocated in modern pedagogical theory.

## 5. Aims & Objectives
**Aim:**
To develop a robust, open-source AI system capable of predicting student academic outcomes with high accuracy and providing automated, personalized diagnostic advice.

**Objectives:**
1.  **Data Curation:** To process and clean a synthetic dataset of 10,000+ student records, ensuring integrity through advanced imputation strategies.
2.  **Feature Engineering:** To design and implement composite behavioral indicators (e.g., Engagement Score, Resource Index) that quantify student effort and environment.
3.  **Predictive Modeling:** To train and fine-tune ensemble machine learning models (Logistic Regression for trend analysis and Random Forest for holistic risk assessment).
4.  **System Development:** To build a decoupled Client-Server architecture using Python, FastAPI, and Streamlit for real-time diagnostics.
5.  **Diagnostic Agent:** To develop a heuristic-based AI agent that translates statistical predictions into natural language pedagogical advice.

## 6. Research Questions
1.  **RQ1:** To what extent do engineered behavioral markers (Attendance, Study Hours, Participation) improve the accuracy of student performance prediction compared to raw academic scores alone?
2.  **RQ2:** Can an ensemble of linear (Trend Dynamics) and non-linear (Random Forest) models effectively differentiate between students with "improving" momentum vs. those at "critical risk"?
3.  **RQ3:** How can quantitative risk probabilities be effectively translated into qualitative, actionable advice for teachers and parents?

## 7. Methods
The project will employ a quantitative research methodology centered on supervised machine learning.

### 7.1 Dataset & Preprocessing
The study utilizes a dataset of **10,000+ student records**.
-   **Cleaning strategy:** Median imputation for quantitative variables and mode imputation for categorical variables.
-   **Leakage Prevention:** Strict isolation of target variables (e.g., Final Marks) during the training of trend detection models.

### 7.2 The Intelligence Layer (Feature Engineering)
We will construct a "Feature Engineering Matrix" to synthesize raw inputs:
-   **Engagement Score (ES):** A weighted composite of Attendance (50%), Study Hours (30%), and Extracurriculars (20%).
-   **Academic Stability (AS):** Derived from the standard deviation of historical scores ($100 - 2\sigma$), identifying volatility.
-   **Resource Index (RI):** A measure of socio-economic support infrastructure (Internet, Transport, Residence).

### 7.3 Algorithm Selection
-   **Logistic Regression:** Employed for **Trend Dynamics** (Linear Momentum) to classify students as "Improving" or "Declining" based on temporal changes. Hyperparameters: *Solver=lbfgs, C=1.0*.
-   **Random Forest Classifier:** Employed for **Holistic Risk prediction**, utilizing 200 estimators to capture non-linear interactions between demographics and performance.

### 7.4 Architecture
The system follows a **PEAS** (Performance, Environment, Actuators, Sensors) analytical desgin:
-   **Tech Stack:** Python (Backend), Scikit-Learn (ML), Streamlit (Frontend/Dashboard), FastAPI (API).

## 8. Plan for Analysis
The analysis will proceed in three phases:

1.  **Exploratory Data Analysis (EDA):** multivariate analysis to understand correlations between the new "Engagement Score" and "Final Marks".
2.  **Model Evaluation:**
    -   Models will be evaluated using **F1-Score** and **Accuracy** metrics.
    -   We target a Risk Prediction Accuracy > 92% and Trend Interpretation Accuracy > 98%.
    -   Confusion Matrices will be generated to minimize False Negatives (identifying at-risk students as safe).
3.  **Agent Logic Verification:** The "AI Advising Agent" will be tested against a "Successful Cohort" baseline. The agent's logic checks for specific gaps (e.g., "Attendance Gap > 15%") and triggers specific advisory blocks.

## 9. References
*Since this project utilizes standard open-source libraries and methodologies, we primarily reference the foundational tools and concepts:*

1.  **Scikit-Learn Community:** *User Guide: Ensemble Methods (Random Forests)*.
2.  **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3.  **Baker, R.S.J.d., & Yacef, K.** (2009). *The State of Educational Data Mining in 2009: A Review and Future Visions*. Journal of Educational Data Mining.
4.  **Python Software Foundation.** *pandas: powerful Python data analysis toolkit*.
5.  **Streamlit.** *The fastest way to build and share data apps*.

## 10. Timeline
The project plan covers a 10-week development lifecycle:

| Phase | Weeks | key Activities |
| :--- | :--- | :--- |
| **Phase 1: Preparation** | Week 1-2 | Requirement gathering, Dataset acquisition, Literature study. |
| **Phase 2: Data Engineering** | Week 3-4 | Cleaning, Imputation, construction of Engagement Score & Stability metrics. |
| **Phase 3: Model Development** | Week 5-6 | Training Logistic Regression & Random Forest models, Hyperparameter tuning (GridSearchCV). |
| **Phase 4: System Integration** | Week 7-8 | Backend API development, Streamlit Dashboard implementation, Agent logic coding. |
| **Phase 5: Reporting** | Week 9-10 | Final testing, Documentation, Proposal defense preparation. |

## 11. Budget
**Total Budget: $0.00 (Open Source)**

This project is designed to be completely open-source and cost-efficient.
-   **Software:** All languages and libraries (Python, Pandas, Scikit-learn, VS Code) are free and open-source (FOSS).
-   **Infrastructure:** Development will occur on local machines. Deployment can be hosted on free-tier community cloud services (e.g., Streamlit Community Cloud) if remote access is required.
-   **Data:** The dataset is synthetic/anonymized, incurring no acquisition costs.
