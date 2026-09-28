import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv('osa_spo2_dataset.csv')
LOOKAHEAD = 3

parts = []
for rec, g in df.groupby('record'):
    g = g.sort_values('window_index').reset_index(drop=True)
    ev = set(g.loc[g['label'] == 1, 'window_index'])
    last = g['window_index'].max()
    g['future_label'] = [int(any((w + k) in ev for k in range(1, LOOKAHEAD + 1))) for w in g['window_index']]
    parts.append(g[g['window_index'] <= last - LOOKAHEAD])
df = pd.concat(parts, ignore_index=True)
df = df[df['label'] == 0].reset_index(drop=True)  # onset-only: currently-normal windows

base_cols = ['heart_rate', 'rr_sdnn', 'breath_rate', 'breath_amplitude_mean']
spo2_cols = base_cols + ['spo2_mean', 'spo2_min', 'spo2_std', 'spo2_drop_count', 'spo2_max_drop']
params = {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 1.0}

def run(cols, name):
    X, y, groups = df[cols], df['future_label'], df['record']
    aucs = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        p = groups.iloc[test_idx].unique()[0]
        if y.iloc[test_idx].nunique() < 2:
            continue
        clf = XGBClassifier(**params, eval_metric='logloss', random_state=42,
                            scale_pos_weight=(y.iloc[train_idx]==0).sum() / max((y.iloc[train_idx]==1).sum(), 1))
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        auc = roc_auc_score(y.iloc[test_idx], clf.predict_proba(X.iloc[test_idx])[:, 1])
        aucs.append(auc)
        print(f"[{name}] held out {p:8s} | onset-only ROC-AUC: {auc:.3f}")
    print(f"[{name}] MEAN: {np.mean(aucs):.3f}\n")

run(base_cols, "without SpO2")
run(spo2_cols, "with SpO2")