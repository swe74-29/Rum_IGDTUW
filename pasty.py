import wfdb
import numpy as np
import pandas as pd
import os

SELECTED = ['slp37', 'slp60', 'slp16', 'slp01a', 'slp41']
DOWNLOAD_DIR = 'slpdb_data'
RESP_EVENT_CODES = {'H', 'HA', 'OA', 'X', 'CA', 'CAA'}

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
wfdb.dl_database('slpdb', dl_dir=DOWNLOAD_DIR, records=SELECTED)

all_windows = []

for rec in SELECTED:
    path = os.path.join(DOWNLOAD_DIR, rec)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'st')
    fs = record.fs
    window_samples = int(30 * fs)
    signal = record.p_signal
    sig_names = record.sig_name

    for i, sample_idx in enumerate(ann.sample):
        note = ann.aux_note[i]
        is_event = int(any(code in note.split() for code in RESP_EVENT_CODES))

        start, end = sample_idx, sample_idx + window_samples
        if end > signal.shape[0]:
            continue

        window_data = signal[start:end, :]
        row = {'record': rec, 'window_index': i, 'label': is_event, 'raw_annotation': note}

        for ch_idx, ch_name in enumerate(sig_names):
            ch = window_data[:, ch_idx]
            row[f'{ch_name}_mean'] = np.mean(ch)
            row[f'{ch_name}_std'] = np.std(ch)
            row[f'{ch_name}_min'] = np.min(ch)
            row[f'{ch_name}_max'] = np.max(ch)

        all_windows.append(row)

df = pd.DataFrame(all_windows)
df.to_csv('osa_windowed_dataset.csv', index=False)
print(df['label'].value_counts())
print("Saved:", df.shape)