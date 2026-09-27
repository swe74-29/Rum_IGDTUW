import wfdb
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

SELECTED = ['slp37', 'slp60', 'slp16', 'slp01a', 'slp41']
DOWNLOAD_DIR = 'slpdb_data'
RESP_EVENT_CODES = {'H', 'HA', 'OA', 'X', 'CA', 'CAA'}
all_windows = []

def find_resp_channel(sig_names_lower):
    for pref in ['nasal', 'sum', 'abdominal']:
        for idx, name in enumerate(sig_names_lower):
            if 'resp' in name and pref in name:
                return idx
    for idx, name in enumerate(sig_names_lower):
        if 'resp' in name:
            return idx
    return None

for rec in SELECTED:
    path = os.path.join(DOWNLOAD_DIR, rec)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'st')
    fs = record.fs
    window_samples = int(30 * fs)
    signal = record.p_signal
    sig_names = record.sig_name
    sig_names_lower = [s.lower() for s in sig_names]

    ecg_idx = next((i for i, n in enumerate(sig_names_lower) if 'ecg' in n), None)
    resp_idx = find_resp_channel(sig_names_lower)

    if ecg_idx is None or resp_idx is None:
        print(f"Skipping {rec} — missing channel. Available: {sig_names}")
        continue

    print(f"{rec}: using ECG idx {ecg_idx}, Resp idx {resp_idx} ({sig_names[resp_idx]})")

    for i, sample_idx in enumerate(ann.sample):
        note = ann.aux_note[i]
        is_event = int(any(code in note.split() for code in RESP_EVENT_CODES))
        start, end = sample_idx, sample_idx + window_samples
        if end > signal.shape[0]:
            continue

        ecg_seg = signal[start:end, ecg_idx]
        resp_seg = signal[start:end, resp_idx]

        peaks, _ = find_peaks(ecg_seg, distance=fs*0.3, prominence=np.std(ecg_seg))
        rr_intervals = np.diff(peaks) / fs if len(peaks) > 1 else np.array([np.nan])

        breaths, _ = find_peaks(resp_seg, distance=fs*1.5, prominence=np.std(resp_seg)*0.3)
        breath_intervals = np.diff(breaths) / fs if len(breaths) > 1 else np.array([np.nan])

        row = {
            'record': rec, 'window_index': i, 'label': is_event, 'raw_annotation': note,
            'heart_rate': 60 / np.nanmean(rr_intervals) if not np.isnan(rr_intervals).all() else np.nan,
            'rr_sdnn': np.nanstd(rr_intervals),
            'n_r_peaks': len(peaks),
            'breath_rate': 60 / np.nanmean(breath_intervals) if not np.isnan(breath_intervals).all() else np.nan,
            'breath_interval_std': np.nanstd(breath_intervals),
            'n_breaths': len(breaths),
        }
        all_windows.append(row)

df = pd.DataFrame(all_windows)
df = df.dropna()
df.to_csv('osa_windowed_dataset_hrv.csv', index=False)
print(df['label'].value_counts())
print(df.describe())

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ddf = pd.read_csv('osa_windowed_dataset_hrv.csv')   # <- not osa_windowed_dataset_normalized.csv
feature_cols = ['heart_rate', 'rr_sdnn', 'n_r_peaks', 'breath_rate', 'breath_interval_std', 'n_breaths']  # <- not the old *_mean/_std/_min/_max columns
X, y, groups = df[feature_cols], df['label'], df['record']

logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    test_patient = groups.iloc[test_idx].unique()[0]
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    probs = clf.predict_proba(X.iloc[test_idx])[:, 1]
    try:
        auc = roc_auc_score(y.iloc[test_idx], probs)
    except ValueError:
        auc = float('nan')
    print(f"Held out: {test_patient:8s} | ROC-AUC: {auc:.3f} | positive rate: {y.iloc[test_idx].mean():.2f}")