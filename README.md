# Rum – OSA Digital Twin: Predicting Obstructive Sleep Apnea Events from Wearable Signals + Synthetic EHR

## Team Details
- **Team Name:** Rum
- **Team Leader:** Sweta Halder
- **College/Incubator:** IGDTUW
- **Team Size:** 1 (Solo)

## Project Title
Digital Twin for Obstructive Sleep Apnea (OSA) Risk Prediction

## Problem Statement & Healthcare Use Case
Obstructive Sleep Apnea is a widely underdiagnosed, lifestyle-linked chronic condition, strongly correlated with the obesity, hypertension, and type 2 diabetes epidemic in India. This project builds a proof-of-concept Digital Twin that fuses:
- **Static/Historical data**: synthetic EHR records (demographics, comorbidities — obesity, hypertension, diabetes) generated via Synthea, converted into a weighted static risk score.
- **Dynamic/Real-time data**: physiological wearable-style signals (ECG, respiration) from real overnight polysomnography recordings, converted into heart-rate-variability and breathing-pattern features.

The system predicts apnea/hypopnea events from short signal windows and combines this with a patient's static risk profile into a single fused risk score, intended to power a conceptual doctor-facing dashboard that flags high-risk patients before a full clinical diagnosis.

## Technical Stack
- **Language:** Python
- **Signal processing:** `wfdb`, `scipy.signal` (R-peak detection, breath-cycle detection)
- **Modeling:** `scikit-learn` (Random Forest Classifier)
- **Data handling:** `pandas`, `numpy`
- **Validation:** Leave-One-Group-Out cross-validation (grouped by patient, to avoid data leakage across windows from the same recording)

## Datasets Used (all open-access)
- **MIT-BIH Polysomnographic Database (slpdb)** — PhysioNet, open-access tier. Provides synchronized ECG, respiration (nasal/abdominal), EEG, and BP signals with expert-scored sleep-stage and apnea/hypopnea annotations.
  - Subjects used: `slp37`, `slp60`, `slp16`, `slp01a`, `slp41` (selected across a range of apnea severities, plus one healthy control with zero events).
- **Synthea** (open-source synthetic patient generator) — used to generate a synthetic patient population, filtered for OSA-relevant comorbidities (obesity, hypertension, prediabetes/diabetes) to build the static EHR risk layer.

## Methodology
1. **Feature engineering:** Each 30-second signal window is converted into heart-rate-variability features (mean heart rate, RR-interval SDNN, R-peak count) and breathing-pattern features (breath rate, breath-interval variability, breath count), rather than raw amplitude statistics — this aligns with standard ECG-based apnea detection literature.
2. **Labeling:** Ground-truth labels come directly from the database's expert-scored annotations (Hypopnea, Obstructive Apnea, Central Apnea event codes), not self-generated proxies.
3. **Validation:** Evaluated with Leave-One-Group-Out cross-validation, holding out one full patient at a time, to test genuine generalization to unseen individuals rather than reporting inflated same-patient scores.
4. **Fusion:** Static EHR risk score (rule-weighted from comorbidities) is combined with the dynamic model's real-time event probability into a single fused risk output, feeding a conceptual dashboard flag (LOW / MODERATE / HIGH).

## Results

All results use Leave-One-Group-Out cross-validation (one full patient held out per fold) on 15 subjects with a consistent nasal-airflow respiration channel (13 scoreable; the 2 healthy controls have no positive events to score against).

| Task | Features | Mean ROC-AUC |
|---|---|---|
| Detect event in current 30s window | ECG + respiration (time- and frequency-domain HRV, breath rate/amplitude) | 0.687 (range 0.56–0.83) |
| Predict event within next ~90s (all windows) | Same | 0.722 |
| Predict event within next ~90s, from currently-normal windows only | Same | **0.658** |
| Same onset-only task, 4 subjects with SpO2 | Without SpO2 | 0.660 |
| Same onset-only task, 4 subjects with SpO2 | With SpO2 | **0.730** |

**Why the onset-only number is the headline early-warning result:** apnea events cluster, so a naive "current label predicts future label" baseline scores 0.776, higher than our all-window predictive AUC of 0.722. The 0.658 onset-only figure removes that shortcut by testing only windows that are currently normal.

**SpO2:** adding oxygen saturation improved every held-out subject (mean onset-only AUC 0.660 to 0.730), suggesting oximetry-equipped wearables would materially improve early warning.

## Limitations & Future Work
- Modest absolute performance (AUC ~0.66 onset-only); 30-second windows are short for frequency-domain HRV, which conventionally uses 2–5 minutes.
- The SpO2 experiment covers only 4 severe-apnea subjects and no controls. Because events cluster, SpO2 in "normal" windows may partly reflect recovery from a preceding event.
- The base model uses 15 of the 18 available subjects; 3 lack a nasal airflow channel. Mixing respiration channel types degraded performance in early experiments (effort-based vs airflow signals differ physiologically).
- Static (synthetic EHR) and dynamic (real, anonymized) data cannot be linked per-patient, so fusion happens at the risk-score level, not via a patient-level join.
- Future work: longer windows, sequence models (LSTM/1D-CNN) on raw signals, a larger and more diverse cohort, and prospective wearable validation.

## Open-Source License
[MIT License / Apache 2.0 — pick one and add a LICENSE file]

## Demo Video
[Link to unlisted YouTube video or PPT recording — add before submission]

## Architecture Diagram
[Link to PDF/PPT — add before submission]

## Presentation
[Link to PDF/PPT — add before submission]