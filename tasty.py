import pandas as pd

conditions = pd.read_csv(r'D:\synthea-master\synthea-master\output\csv\conditions.csv')

risk_keywords = ['Obesity', 'Hypertension', 'Diabetes', 'Sleep apnea', 'Body mass index']
mask = conditions['DESCRIPTION'].str.contains('|'.join(risk_keywords), case=False, na=False)
risk_conditions = conditions[mask]

print(risk_conditions['DESCRIPTION'].value_counts())
print("Matching rows:", len(risk_conditions))
print("Unique patients with a risk factor:", risk_conditions['PATIENT'].nunique())

patients = pd.read_csv(r'D:\synthea-master\synthea-master\output\csv\patients.csv')
risk_patient_ids = risk_conditions['PATIENT'].unique()

risk_patients = patients[patients['Id'].isin(risk_patient_ids)]
risk_patients.to_csv('synthetic_ehr_osa_risk.csv', index=False)
print("Saved:", risk_patients.shape)