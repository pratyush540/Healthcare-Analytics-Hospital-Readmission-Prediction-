import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES = 15_000


def generate_synthetic_readmission_data(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    np.random.seed(seed)

    age_groups = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age = np.random.choice(age_groups, size=n, p=[0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.10, 0.02])

    gender = np.random.choice(['Female', 'Male'], size=n, p=[0.52, 0.48])

    race = np.random.choice(
        ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'Unknown'],
        size=n,
        p=[0.55, 0.22, 0.12, 0.06, 0.03, 0.02]
    )

    number_outpatient = np.random.poisson(lam=4, size=n).clip(0, 20)
    number_emergency = np.random.poisson(lam=1.5, size=n).clip(0, 15)
    number_inpatient = np.random.poisson(lam=1.2, size=n).clip(0, 12)

    admission_type = np.random.choice(
        ['Emergency', 'Elective', 'Urgent', 'Newborn', 'Trauma', 'Other'],
        size=n, p=[0.55, 0.15, 0.22, 0.02, 0.03, 0.03]
    )
    admission_source = np.random.choice(
        ['Physician Referral', 'Emergency', 'Transfer', 'Clinic', 'Other'],
        size=n, p=[0.25, 0.45, 0.12, 0.15, 0.03]
    )
    discharge_disposition = np.random.choice(
        ['Home', 'Transfer', 'Rehab', 'Expired', 'Other'],
        size=n, p=[0.72, 0.12, 0.08, 0.02, 0.06]
    )

    time_in_hospital = np.random.gamma(shape=2.5, scale=2.0, size=n).astype(int).clip(1, 14)
    time_in_hospital = np.where(
        np.random.random(n) < 0.4,
        np.random.gamma(shape=4, scale=2, size=n).astype(int).clip(3, 14),
        time_in_hospital
    )

    num_procedures = np.random.poisson(lam=1.5, size=n).clip(0, 10)
    num_medications = np.random.poisson(lam=12, size=n).clip(1, 30)
    num_lab_procedures = np.random.poisson(lam=45, size=n).clip(1, 120)

    max_glu_serum = np.random.choice(['None', 'Norm', '>200', '>300'], size=n, p=[0.85, 0.08, 0.05, 0.02])
    A1Cresult = np.random.choice(['None', 'Norm', '>7', '>8'], size=n, p=[0.78, 0.12, 0.07, 0.03])

    diag_categories = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasm', 'Other']
    diag_1 = np.random.choice(diag_categories, size=n, p=[0.22, 0.15, 0.12, 0.18, 0.10, 0.08, 0.06, 0.04, 0.05])
    diag_2 = np.random.choice(diag_categories, size=n, p=[0.18, 0.14, 0.14, 0.15, 0.12, 0.10, 0.08, 0.04, 0.05])
    diag_3 = np.random.choice(diag_categories, size=n, p=[0.15, 0.12, 0.15, 0.12, 0.14, 0.12, 0.10, 0.05, 0.05])

    number_diagnoses = np.random.poisson(lam=7, size=n).clip(1, 16)

    diabetes = np.random.binomial(1, 0.42, size=n)
    hypertension = np.random.binomial(1, 0.55, size=n)
    chronic_kidney = np.random.binomial(1, 0.18, size=n)
    heart_failure = np.random.binomial(1, 0.22, size=n)

    readmission_base = 0.12
    risk_score = (
        0.03 * (number_inpatient > 2) +
        0.04 * (time_in_hospital >= 7) +
        0.03 * (num_medications >= 20) +
        0.02 * (number_diagnoses >= 10) +
        0.03 * (admission_type == 'Emergency') +
        0.02 * (diabetes == 1) +
        0.02 * (chronic_kidney == 1) +
        0.02 * (heart_failure == 1) +
        np.random.normal(0, 0.02, size=n)
    )
    readmission_prob = np.clip(readmission_base + risk_score + np.random.uniform(-0.02, 0.02, n), 0.05, 0.55)
    readmitted = (np.random.random(n) < readmission_prob).astype(int)

    def inject_missing(arr, frac=0.03):
        mask = np.random.random(n) < frac
        arr = arr.astype(object)
        arr[mask] = np.nan
        return arr

    race = inject_missing(race, 0.03)
    max_glu_serum = inject_missing(max_glu_serum, 0.04)
    A1Cresult = inject_missing(A1Cresult, 0.04)
    diag_2 = inject_missing(diag_2, 0.02)
    diag_3 = inject_missing(diag_3, 0.02)

    admission_dates = pd.to_datetime(np.random.choice(
        pd.date_range('2018-01-01', periods=n, freq='h').values, size=n, replace=False
    ))

    df = pd.DataFrame({
        'patient_id': range(1, n + 1),
        'age': age,
        'gender': gender,
        'race': race,
        'admission_type': admission_type,
        'admission_source': admission_source,
        'discharge_disposition': discharge_disposition,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'max_glu_serum': max_glu_serum,
        'A1Cresult': A1Cresult,
        'diag_1': diag_1,
        'diag_2': diag_2,
        'diag_3': diag_3,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'chronic_kidney_disease': chronic_kidney,
        'heart_failure': heart_failure,
        'admission_date': admission_dates,
        'readmitted': readmitted,
    })

    return df


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / 'hospital_readmission.csv'

    print(f"Generating {N_SAMPLES} synthetic patient records...")
    df = generate_synthetic_readmission_data(n=N_SAMPLES, seed=RANDOM_STATE)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Readmission rate: {df['readmitted'].mean():.2%}")
    return df


if __name__ == '__main__':
    main()
