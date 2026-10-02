"""Step 7: Choose alert thresholds from out-of-fold predictions (leave-one-patient-out)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

FEATURES = ['heart_rate', 'rr_sdnn', 'n_r_peaks', 'lf_power', 'hf_power', 'lf_hf_ratio',
            'breath_rate', 'breath_interval_std', 'n_breaths',
            'breath_amplitude_mean', 'breath_amplitude_std', 'resp_peak_to_peak']
PARAMS = {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 3,
          'learning_rate': 0.1, 'colsample_bytree': 1.0}
SMOOTH = 6                     # trailing windows (~3 min), uses past data only
HEALTHY = ['slp41', 'slp45']   # zero-event controls

df = (pd.read_csv('osa_predictive_dataset.csv')
        .sort_values(['record', 'window_index']).reset_index(drop=True))
X, y, groups = df[FEATURES], df['future_label'], df['record']

oof = np.full(len(df), np.nan)
for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
    if y.iloc[train_idx].nunique() < 2:
        continue
    spw = (y.iloc[train_idx] == 0).sum() / max((y.iloc[train_idx] == 1).sum(), 1)
    clf = XGBClassifier(**PARAMS, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

df['score'] = oof
df['score_smooth'] = df.groupby('record')['score'].transform(
    lambda s: s.rolling(SMOOTH, min_periods=1).mean())

yv = df['future_label'].values
is_healthy = df['record'].isin(HEALTHY).values
print(f"Base rate of pre-event windows: {yv.mean():.2f}\n")

for col in ['score', 'score_smooth']:
    print(col)
    print(f"{'thr':>5} {'alert%':>8} {'precision':>10} {'recall':>8} {'healthy alert%':>15}")
    for t in np.arange(0.40, 0.90, 0.05):
        alert = df[col].values >= t
        if alert.sum() == 0:
            continue
        print(f"{t:5.2f} {alert.mean():8.1%} {yv[alert].mean():10.2f} "
              f"{alert[yv == 1].mean():8.2f} {alert[is_healthy].mean():15.1%}")
    print()