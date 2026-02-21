# Hospital Readmission Prediction – Healthcare Analytics Project

End-to-end **Hospital Readmission Prediction** project using Python and Scikit-learn: from business problem definition through EDA, feature engineering, model building, evaluation, and deployment considerations.

## Quick start (after clone)

```bash
git clone <your-repo-url>
cd Health
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py
jupyter notebook notebooks/01_hospital_readmission_analysis.ipynb
```

## Project structure

```
Health/
├── data/                    # Generated dataset (run generator first)
│   └── hospital_readmission.csv
├── notebooks/
│   └── 01_hospital_readmission_analysis.ipynb   # Main analysis
├── src/
│   └── generate_data.py    # Synthetic dataset generator
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   # source venv/bin/activate   # Linux/Mac
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset**

   From the project root (`Health/`):

   ```bash
   python src/generate_data.py
   ```

   This creates `data/hospital_readmission.csv` (~15,000 synthetic patient records). The notebook can also generate the data automatically if the file is missing.

## Run the analysis

- Open `notebooks/01_hospital_readmission_analysis.ipynb` in Jupyter.
- Run all cells. Recommended: run from the **project root** (e.g. start Jupyter from `Health/`) so paths resolve correctly; or run from inside `notebooks/` (the notebook detects this and uses the parent folder for `data/`).

```bash
jupyter notebook notebooks/01_hospital_readmission_analysis.ipynb
```

## Contents of the notebook

1. **Business problem** – Why readmission matters, cost impact, operational efficiency.
2. **Dataset description** – Demographics, medical history, admission details, lab results, diagnosis codes.
3. **Data cleaning** – Missing values, duplicates, categorical encoding, outlier handling.
4. **Exploratory data analysis** – Readmission rate, age/diagnosis/LOS vs readmission, correlation heatmap.
5. **Feature engineering** – Risk scores, admission date features, age bins, chronic condition indicators.
6. **Model building** – Logistic Regression, Random Forest, XGBoost (optional); train-test split and cross-validation.
7. **Model evaluation** – Accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
8. **Feature importance** – Random Forest feature importance and top drivers.
9. **Business interpretation** – High-risk identification, risk segmentation, operational recommendations.
10. **Deployment** – How hospitals can use the model and a simple monitoring strategy.

## Dataset (synthetic)

The synthetic data includes:

- **Demographics**: age group, gender, race  
- **Utilization**: prior outpatient, emergency, inpatient visits  
- **Admission**: type, source, discharge disposition, length of stay, admission date  
- **Labs/meds**: number of lab procedures, procedures, medications; max glucose, A1C  
- **Diagnosis**: primary/secondary/tertiary diagnosis categories, number of diagnoses  
- **Chronic conditions**: diabetes, hypertension, chronic kidney disease, heart failure (binary)  
- **Target**: `readmitted` (1 = readmitted within 30 days, 0 = not)

You can replace this with a real dataset (e.g. UCI Diabetes 130-US hospitals) by changing the loading cell to read your file and aligning column names and types.

## License

Use for learning and internal analytics. For production or research, ensure compliance with data and healthcare regulations (e.g. HIPAA, local laws).
