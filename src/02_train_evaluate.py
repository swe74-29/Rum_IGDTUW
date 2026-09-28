"""Step 2: Tune XGBoost and evaluate current-window event detection (Leave-One-Group-Out)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

FEATURES = ['heart_rate', 'rr_sdnn', 'n_r_peaks', 'lf_power', 'hf_power', 'lf_hf_ratio',
            'breath_rate', 'breath_interval_std', 'n_breaths',
            'breath_amplitude_mean', 'breath_amplitude_std', 'resp_peak_to_peak']

df = pd.read_csv('osa_windowed_dataset_v2.csv')
X, y, groups = df[FEATURES], df['label'], df['record']

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist, n_iter=20, scoring='roc_auc',
    cv=GroupKFold(n_splits=5), random_state=42, n_jobs=-1)
search.fit(X, y, groups=groups)
best_params = search.best_params_
print("Best params:", best_params)

results = []
for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
    patient = groups.iloc[test_idx].unique()[0]
    pos_rate = y.iloc[test_idx].mean()
    if pos_rate in (0, 1) or y.iloc[train_idx].nunique() < 2:
        print(f"Held out: {patient:8s} | ROC-AUC: nan | positive rate: {pos_rate:.2f}")
        continue
    spw = (y.iloc[train_idx] == 0).sum() / max((y.iloc[train_idx] == 1).sum(), 1)
    clf = XGBClassifier(**best_params, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    auc = roc_auc_score(y.iloc[test_idx], clf.predict_proba(X.iloc[test_idx])[:, 1])
    results.append(auc)
    print(f"Held out: {patient:8s} | ROC-AUC: {auc:.3f} | positive rate: {pos_rate:.2f}")

print(f"\nMean ROC-AUC across {len(results)} scoreable subjects: {np.mean(results):.3f}")
print(f"Min: {min(results):.3f} | Max: {max(results):.3f}")