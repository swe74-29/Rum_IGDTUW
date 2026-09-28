"""Step 3: Early-warning framing. Predict whether an event occurs in the next ~90s (3 windows)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

LOOKAHEAD = 3
FEATURES = ['heart_rate', 'rr_sdnn', 'n_r_peaks', 'lf_power', 'hf_power', 'lf_hf_ratio',
            'breath_rate', 'breath_interval_std', 'n_breaths',
            'breath_amplitude_mean', 'breath_amplitude_std', 'resp_peak_to_peak']
PARAMS = {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 3,
          'learning_rate': 0.1, 'colsample_bytree': 1.0}

# ---- build future labels ----
df = pd.read_csv('osa_windowed_dataset_v2.csv')
df = df.sort_values(['record', 'window_index']).reset_index(drop=True)

parts = []
for rec, g in df.groupby('record'):
    g = g.reset_index(drop=True)
    labels = g['label'].values
    future = np.zeros(len(labels), dtype=int)
    for i in range(len(labels) - 1):
        end = min(i + 1 + LOOKAHEAD, len(labels))
        future[i] = int(labels[i + 1:end].any())
    g['future_label'] = future
    parts.append(g.iloc[:-LOOKAHEAD] if len(g) > LOOKAHEAD else g)  # last rows: future unobserved

df = pd.concat(parts, ignore_index=True)
df.to_csv('osa_predictive_dataset.csv', index=False)
print(df['future_label'].value_counts(), "\n")


def logo_eval(data, label_name, title):
    X, y, groups = data[FEATURES], data[label_name], data['record']
    aucs = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        patient = groups.iloc[test_idx].unique()[0]
        if y.iloc[test_idx].nunique() < 2 or y.iloc[train_idx].nunique() < 2:
            print(f"Held out: {patient:8s} | ROC-AUC: nan")
            continue
        spw = (y.iloc[train_idx] == 0).sum() / max((y.iloc[train_idx] == 1).sum(), 1)
        clf = XGBClassifier(**PARAMS, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        auc = roc_auc_score(y.iloc[test_idx], clf.predict_proba(X.iloc[test_idx])[:, 1])
        aucs.append(auc)
        print(f"Held out: {patient:8s} | ROC-AUC: {auc:.3f}")
    print(f"{title} MEAN ROC-AUC across {len(aucs)} subjects: {np.mean(aucs):.3f}\n")


# ---- 1) all windows ----
logo_eval(df, 'future_label', "[All windows]")

# ---- 2) persistence baseline: does the CURRENT label alone predict the future label? ----
base = [roc_auc_score(g['future_label'], g['label'])
        for _, g in df.groupby('record')
        if g['future_label'].nunique() == 2 and g['label'].nunique() == 2]
print(f"Persistence baseline (current label only): {np.mean(base):.3f}\n")

# ---- 3) onset-only: currently-normal windows, predict upcoming event ----
df0 = df[df['label'] == 0].reset_index(drop=True)
logo_eval(df0, 'future_label', "[Onset-only]")