"""Step 4: SpO2 experiment on the 4 subjects that have both a nasal channel and an SO2 channel."""
import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SPO2_RECORDS = ['slp59', 'slp60', 'slp66', 'slp67x']
DOWNLOAD_DIR = 'slpdb_data'
RESP_EVENT_CODES = {'H', 'HA', 'OA', 'X', 'CA', 'CAA'}
LOOKAHEAD = 3
PARAMS = {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 3,
          'learning_rate': 0.1, 'colsample_bytree': 1.0}
BASE_COLS = ['heart_rate', 'rr_sdnn', 'breath_rate', 'breath_amplitude_mean']
SPO2_COLS = BASE_COLS + ['spo2_mean', 'spo2_min', 'spo2_std', 'spo2_drop_count', 'spo2_max_drop']

# ---------- feature extraction ----------
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
wfdb.dl_database('slpdb', dl_dir=DOWNLOAD_DIR, records=SPO2_RECORDS)
rows = []

for rec in SPO2_RECORDS:
    path = os.path.join(DOWNLOAD_DIR, rec)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'st')
    fs = record.fs
    window_samples = int(30 * fs)
    signal = record.p_signal
    names = [s.lower() for s in record.sig_name]

    ecg_idx = next((i for i, n in enumerate(names) if 'ecg' in n), None)
    resp_idx = next((i for i, n in enumerate(names) if 'resp' in n and 'nasal' in n), None)
    spo2_idx = next((i for i, n in enumerate(names) if 'so2' in n), None)
    if None in (ecg_idx, resp_idx, spo2_idx):
        print(f"Skipping {rec}: missing channel {record.sig_name}")
        continue

    for i, sample_idx in enumerate(ann.sample):
        note = ann.aux_note[i]
        is_event = int(any(c in note.split() for c in RESP_EVENT_CODES))
        start, end = sample_idx, sample_idx + window_samples
        if end > signal.shape[0]:
            continue

        ecg_seg, resp_seg, spo2_seg = (signal[start:end, ecg_idx],
                                       signal[start:end, resp_idx],
                                       signal[start:end, spo2_idx])
        peaks, _ = find_peaks(ecg_seg, distance=fs * 0.3, prominence=np.std(ecg_seg))
        rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([np.nan])
        breaths, props = find_peaks(resp_seg, distance=fs * 1.5, prominence=np.std(resp_seg) * 0.3)
        b_int = np.diff(breaths) / fs if len(breaths) > 1 else np.array([np.nan])
        b_amp = props['prominences']

        baseline = np.percentile(spo2_seg, 95)
        rows.append({
            'record': rec, 'window_index': i, 'label': is_event,
            'heart_rate': 60 / np.nanmean(rr) if not np.isnan(rr).all() else np.nan,
            'rr_sdnn': np.nanstd(rr),
            'breath_rate': 60 / np.nanmean(b_int) if not np.isnan(b_int).all() else np.nan,
            'breath_amplitude_mean': np.mean(b_amp) if len(b_amp) else np.nan,
            'spo2_mean': np.mean(spo2_seg), 'spo2_min': np.min(spo2_seg),
            'spo2_std': np.std(spo2_seg),
            'spo2_drop_count': np.sum(spo2_seg < (baseline - 3)),
            'spo2_max_drop': baseline - spo2_seg.min(),
        })

df = pd.DataFrame(rows).dropna()
df.to_csv('osa_spo2_dataset.csv', index=False)
print(df['label'].value_counts())
print("Subjects included:", df['record'].nunique(), "\n")


def run(data, cols, target, name):
    X, y, groups = data[cols], data[target], data['record']
    aucs = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        patient = groups.iloc[test_idx].unique()[0]
        if y.iloc[test_idx].nunique() < 2:
            continue
        spw = (y.iloc[train_idx] == 0).sum() / max((y.iloc[train_idx] == 1).sum(), 1)
        clf = XGBClassifier(**PARAMS, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        auc = roc_auc_score(y.iloc[test_idx], clf.predict_proba(X.iloc[test_idx])[:, 1])
        aucs.append(auc)
        print(f"[{name}] held out {patient:8s} | ROC-AUC: {auc:.3f}")
    print(f"[{name}] MEAN: {np.mean(aucs):.3f}\n")


# ---------- experiment A: current-window detection ----------
run(df, BASE_COLS, 'label', "current-window, without SpO2")
run(df, SPO2_COLS, 'label', "current-window, with SpO2")

# ---------- experiment B: onset-only early warning ----------
parts = []
for rec, g in df.groupby('record'):
    g = g.sort_values('window_index').reset_index(drop=True)
    events = set(g.loc[g['label'] == 1, 'window_index'])
    last = g['window_index'].max()
    g['future_label'] = [int(any((w + k) in events for k in range(1, LOOKAHEAD + 1)))
                         for w in g['window_index']]
    parts.append(g[g['window_index'] <= last - LOOKAHEAD])
dfe = pd.concat(parts, ignore_index=True)
dfe = dfe[dfe['label'] == 0].reset_index(drop=True)

run(dfe, BASE_COLS, 'future_label', "onset-only, without SpO2")
run(dfe, SPO2_COLS, 'future_label', "onset-only, with SpO2")