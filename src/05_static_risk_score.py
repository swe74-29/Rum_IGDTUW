"""Step 5: Static EHR risk score from Synthea-generated synthetic patients."""
import pandas as pd

SYNTHEA_CSV_DIR = r'D:\synthea-master\synthea-master\output\csv'  # <- change if your path differs
OUT_CSV = 'synthetic_ehr_osa_risk.csv'

RISK_WEIGHTS = {
    'Body mass index 30+': 2,
    'Body mass index 40+': 3,
    'Essential hypertension': 2,
    'Prediabetes': 1,
    'Diabetes mellitus type 2': 3,
    'Obstructive sleep apnea': 5,
    'Sleep apnea': 5,
}

conditions = pd.read_csv(fr'{SYNTHEA_CSV_DIR}\conditions.csv')
patients = pd.read_csv(fr'{SYNTHEA_CSV_DIR}\patients.csv')


def score_patient(patient_id):
    descs = conditions.loc[conditions['PATIENT'] == patient_id, 'DESCRIPTION']
    return sum(w for d in descs for k, w in RISK_WEIGHTS.items() if k.lower() in d.lower())


risk_patients = patients[patients['Id'].isin(conditions['PATIENT'])].copy()
risk_patients['risk_score'] = risk_patients['Id'].apply(score_patient)
risk_patients = risk_patients[risk_patients['risk_score'] > 0]
risk_patients['static_risk'] = risk_patients['risk_score'] / risk_patients['risk_score'].max()

risk_patients.to_csv(OUT_CSV, index=False)
print(risk_patients[['Id', 'risk_score', 'static_risk']].describe())