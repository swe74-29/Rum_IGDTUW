"""Step 8: Export data for the conceptual dashboard (dashboard/data.js)."""
import json
import os
import pandas as pd

SYNTHEA_CSV_DIR = r'D:\synthea-master\synthea-master\output\csv'  # match 05_static_risk_score.py
N_PATIENTS = 6
OUT_DIR = 'dashboard'
RISK_KEYWORDS = ['Obesity', 'Hypertension', 'Diabetes', 'Sleep apnea', 'Prediabetes']

os.makedirs(OUT_DIR, exist_ok=True)

timeline = pd.read_csv('fusion_demo_timeline.csv')
timeline_data = {
    'minute': timeline['minute'].round(1).tolist(),
    'dynamic_prob': timeline['dynamic_prob'].round(3).tolist(),
    'event_truth': timeline['event_now_truth'].astype(int).tolist(),
}

ehr = pd.read_csv('synthetic_ehr_osa_risk.csv')
ehr['static_pct'] = ehr['risk_score'].rank(pct=True)

conditions_path = os.path.join(SYNTHEA_CSV_DIR, 'conditions.csv')
conditions = pd.read_csv(conditions_path) if os.path.exists(conditions_path) else None

targets = [0.95, 0.80, 0.60, 0.40, 0.20, 0.05][:N_PATIENTS]
patients, used = [], set()

for t in targets:
    remaining = ehr[~ehr['Id'].isin(used)]
    row = remaining.iloc[(remaining['static_pct'] - t).abs().argmin()]
    used.add(row['Id'])

    tags = []
    if conditions is not None:
        descs = conditions.loc[conditions['PATIENT'] == row['Id'], 'DESCRIPTION']
        for kw in RISK_KEYWORDS:
            match = descs[descs.str.contains(kw, case=False, na=False)]
            if not match.empty:
                tags.append(match.iloc[0].split(' (')[0])
    tags = list(dict.fromkeys(tags))[:4]

    age = None
    if 'BIRTHDATE' in row:
        try:
            age = 2026 - int(str(row['BIRTHDATE'])[:4])
        except Exception:
            pass

    patients.append({
        'id': str(row['Id'])[:8],
        'age': age,
        'gender': str(row.get('GENDER', '?')),
        'risk_score': float(row['risk_score']),
        'static_pct': round(float(row['static_pct']), 2),
        'tags': tags or ['No major comorbidities on record'],
    })

patients.sort(key=lambda p: -p['static_pct'])

with open(os.path.join(OUT_DIR, 'data.js'), 'w') as f:
    f.write('const TIMELINE = ' + json.dumps(timeline_data) + ';\n')
    f.write('const PATIENTS = ' + json.dumps(patients) + ';\n')
    f.write('const DEMO_RECORD = ' + json.dumps('slp16') + ';\n')
    f.write('const WEIGHTS = ' + json.dumps({'static': 0.3, 'dynamic': 0.7}) + ';\n')
    f.write('const THRESHOLDS = ' + json.dumps({'high': 0.65, 'moderate': 0.50}) + ';\n')

print(f"Wrote dashboard/data.js — {len(patients)} patients, {len(timeline_data['minute'])} timeline points")