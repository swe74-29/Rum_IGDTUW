"""Step 1: Extract per-30s-window features (ECG HRV + nasal respiration) from slpdb."""
import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d

try:
    trapezoid = np.trapezoid
except AttributeError:  # older NumPy
    trapezoid = np.trapz

SELECTED = ['slp37', 'slp66', 'slp60', 'slp67x', 'slp59', 'slp04', 'slp16',
            'slp48', 'slp03', 'slp14', 'slp02a', 'slp32', 'slp02b', 'slp41', 'slp45']
DOWNLOAD_DIR = 'slpdb_data'
RESP_EVENT_CODES = {'H', 'HA', 'OA', 'X', 'CA', 'CAA'}
OUT_CSV = 'osa_windowed_dataset_v2.csv'


def hrv_freq_features(peaks, fs):
    """LF power, HF power, LF/HF ratio from R-peak positions."""
    if len(peaks) < 5:
        return np.nan, np.nan, np.nan
    rr_times = peaks[1:] / fs
    rr_values = np.diff(peaks) / fs
    fs_interp = 4.0
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1 / fs_interp)
    if len(t_uniform) < 8:
        return np.nan, np.nan, np.nan
    f = interp1d(rr_times, rr_values, kind='linear', fill_value='extrapolate')
    rr_interp = f(t_uniform)
    freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=min(len(rr_interp), 64))
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    lf = trapezoid(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else np.nan
    hf = trapezoid(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else np.nan
    lf_hf = lf / hf if hf and hf > 0 else np.nan
    return lf, hf, lf_hf


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    wfdb.dl_database('slpdb', dl_dir=DOWNLOAD_DIR, records=SELECTED)
    rows = []

    for rec in SELECTED:
        path = os.path.join(DOWNLOAD_DIR, rec)
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'st')
        fs = record.fs
        window_samples = int(30 * fs)
        signal = record.p_signal
        names = [s.lower() for s in record.sig_name]

        ecg_idx = next((i for i, n in enumerate(names) if 'ecg' in n), None)
        resp_idx = next((i for i, n in enumerate(names) if 'resp' in n and 'nasal' in n), None)
        if ecg_idx is None or resp_idx is None:
            print(f"Skipping {rec}: no nasal/ECG channel")
            continue

        for i, sample_idx in enumerate(ann.sample):
            note = ann.aux_note[i]
            is_event = int(any(c in note.split() for c in RESP_EVENT_CODES))
            start, end = sample_idx, sample_idx + window_samples
            if end > signal.shape[0]:
                continue

            ecg_seg = signal[start:end, ecg_idx]
            resp_seg = signal[start:end, resp_idx]

            peaks, _ = find_peaks(ecg_seg, distance=fs * 0.3, prominence=np.std(ecg_seg))
            rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([np.nan])
            lf, hf, lf_hf = hrv_freq_features(peaks, fs)

            breaths, props = find_peaks(resp_seg, distance=fs * 1.5, prominence=np.std(resp_seg) * 0.3)
            b_int = np.diff(breaths) / fs if len(breaths) > 1 else np.array([np.nan])
            b_amp = props['prominences']

            rows.append({
                'record': rec, 'window_index': i, 'label': is_event, 'raw_annotation': note,
                'heart_rate': 60 / np.nanmean(rr) if not np.isnan(rr).all() else np.nan,
                'rr_sdnn': np.nanstd(rr),
                'n_r_peaks': len(peaks),
                'lf_power': lf, 'hf_power': hf, 'lf_hf_ratio': lf_hf,
                'breath_rate': 60 / np.nanmean(b_int) if not np.isnan(b_int).all() else np.nan,
                'breath_interval_std': np.nanstd(b_int),
                'n_breaths': len(breaths),
                'breath_amplitude_mean': np.mean(b_amp) if len(b_amp) else np.nan,
                'breath_amplitude_std': np.std(b_amp) if len(b_amp) else np.nan,
                'resp_peak_to_peak': resp_seg.max() - resp_seg.min(),
            })

    df = pd.DataFrame(rows).dropna()
    df.to_csv(OUT_CSV, index=False)
    print(df['label'].value_counts())
    print("Subjects included:", df['record'].nunique())


if __name__ == '__main__':
    main()