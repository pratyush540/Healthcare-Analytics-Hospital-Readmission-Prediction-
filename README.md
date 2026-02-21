# 🏥 Healthcare Analytics: Hospital Readmission Prediction

## 👤 Author
Pratyush Anand  
Data Analyst | Machine Learning | Python  

---

## 📌 Project Overview

Hospital readmissions significantly impact healthcare costs and patient outcomes.  
This project develops a machine learning model to predict whether a patient is likely to be readmitted within 30 days of discharge.

The objective is to help healthcare providers:

- Identify high-risk patients early
- Reduce avoidable readmissions
- Improve operational efficiency
- Optimize resource allocation
- Lower treatment costs

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Statsmodels
- Jupyter Notebook

---

## 📂 Dataset Description

The dataset includes patient-level hospital records such as:

- Patient demographics (Age, Gender)
- Admission type & discharge details
- Diagnosis codes
- Number of previous admissions
- Length of stay
- Lab results
- Medical history indicators
- Readmission status (Target Variable)

---

## 🔎 Project Workflow

### 1️⃣ Business Problem Understanding
Hospital readmissions increase operational burden and cost.  
Predicting readmission risk enables preventive care planning.

---

### 2️⃣ Data Cleaning & Preprocessing

- Handled missing values
- Removed duplicates
- Encoded categorical variables
- Standardized numerical features
- Addressed class imbalance

---

### 3️⃣ Exploratory Data Analysis (EDA)

- Readmission rate distribution
- Age group vs readmission trend
- Length of stay impact
- Diagnosis category analysis
- Correlation heatmap
- Feature distribution analysis

Key observation:
Patients with longer hospital stays and multiple prior admissions showed higher readmission probability.

---

### 4️⃣ Feature Engineering

- Created age bands
- Derived chronic condition indicators
- Transformed admission dates
- Generated risk-related features

---

### 5️⃣ Model Development

Implemented and compared multiple models:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting (Optional)

Data was split using train-test split with cross-validation.

---

### 6️⃣ Model Evaluation

Performance metrics used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

The optimized model demonstrated strong recall performance, ensuring high-risk patients were correctly identified.

---

## 📊 Key Insights

- Prior admission frequency is a strong predictor of readmission.
- Longer length of stay correlates with higher risk.
- Certain diagnosis categories show elevated readmission rates.
- Feature importance analysis highlights operational risk drivers.

---

## 🚀 Business Impact

This model can help hospitals:

- Proactively identify high-risk patients
- Design targeted intervention programs
- Reduce avoidable readmissions
- Improve healthcare quality metrics
- Optimize staffing and bed allocation

---

## 📁 Repository Structure
