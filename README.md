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
- Leave-One-Group-Out ROC-AUC ranged from ~0.6–0.78 across held-out patients (excluding one healthy control with no positive events to score against).

## Limitations & Future Work
- One held-out subject (`slp01a`) showed degraded performance, traced to a genuine physiological cause: this subject's respiration was recorded via a chest/abdominal "effort" channel rather than nasal airflow, which can behave differently during central apnea events. Future work would standardize respiration channel type across all training subjects, or train channel-aware sub-models.
- Only 5 subjects were used for the working prototype due to time constraints; the same pipeline scales directly to the remaining 13 open-access subjects in the database.
- Static (synthetic) and dynamic (real, anonymized) data streams cannot be linked at the individual level due to privacy constraints, so fusion is performed at the risk-score level rather than a direct per-patient join — a limitation inherent to any privacy-compliant digital twin built on separate real and synthetic sources.

## Open-Source License
[MIT License / Apache 2.0 — pick one and add a LICENSE file]

## Demo Video
[Link to unlisted YouTube video or PPT recording — add before submission]

## Architecture Diagram
[Link to PDF/PPT — add before submission]

## Presentation
[Link to PDF/PPT — add before submission]