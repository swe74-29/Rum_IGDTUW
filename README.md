# Rum: OSA Digital Twin

Predicting obstructive sleep apnea (OSA) events by fusing a synthetic EHR risk profile with wearable-style physiological signals.

## Team
- **Team name:** Rum
- **Team leader:** Sweta Halder | **College:** IGDTUW | **Team size:** 1 (solo)
- **Contact:** swetahalder29@gmail.com | 8810584575

## Problem Statement and Use Case
OSA is a common, badly underdiagnosed condition in India, closely tied to obesity, hypertension and type 2 diabetes. Diagnosis needs an overnight sleep-lab study that most patients never get. This project is a proof-of-concept Digital Twin: a virtual patient that combines
- **static data:** a synthetic EHR (comorbidities such as obesity, hypertension, diabetes), and
- **dynamic data:** overnight ECG and respiration signals (wearable-style, from real anonymized recordings),

to estimate, window by window, the risk that an apnea/hypopnea event is about to occur, and to present a doctor with a single fused risk flag.

## Architecture

```
 Synthea synthetic EHR ──► weighted comorbidity score ──► static risk (percentile) ─┐
                                                                                    ├─► fused risk ─► LOW / MODERATE / HIGH ─► clinician dashboard (concept)
 MIT-BIH PSG signals ──► 30 s windows ──► HRV + breathing features ──► XGBoost ─────┘
 (ECG + nasal airflow)     (labels: expert-scored apnea/hypopnea)     (risk score (event within ~90 s))
```

Architecture diagram (PDF/PPT): [link]

## Datasets (open-access, no request or credentialing)
| Source | Use |
|---|---|
| [MIT-BIH Polysomnographic Database](https://physionet.org/content/slpdb/1.0.0/) (PhysioNet, Open Data Commons Attribution License) | Real ECG, nasal airflow and SpO2 with expert-scored apnea/hypopnea annotations |
| [Synthea](https://github.com/synthetichealth/synthea) | Fully synthetic patient records for the static EHR layer |

No real patient data was used beyond the public, anonymized PhysioNet recordings, in line with the challenge's data rules (DPDP Act / HIPAA).

**Attribution:** Ichimaru Y, Moody GB. *Development of the polysomnographic database on CD-ROM.* Psychiatry Clin Neurosci, 1999. Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000.

## Technical Stack
Python, `wfdb`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib`.

## Repository Structure
```
src/
  01_extract_features.py     windowed HRV + breathing features (15 subjects)
  02_train_evaluate.py       XGBoost tuning + leave-one-patient-out evaluation
  03_predictive_labels.py    early-warning labels, persistence baseline, onset-only test
  04_spo2_experiment.py      oxygen-saturation experiment (4 subjects)
  05_static_risk_score.py    Synthea EHR -> static risk score
  06_fusion.py               trains final model, fuses static + dynamic risk, demo timeline
  07_threshold_analysis.py   out-of-fold threshold selection +healthy-subject false-alarm check
requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt

# Static EHR layer: generate synthetic patients with Synthea (CSV export is off by default)
#   .\run_synthea.bat -s 42 -p 300 --exporter.csv.export=true
# then set SYNTHEA_CSV_DIR at the top of src/05_static_risk_score.py

python src/01_extract_features.py      # downloads the PhysioNet records automatically
python src/02_train_evaluate.py
python src/03_predictive_labels.py
python src/04_spo2_experiment.py
python src/05_static_risk_score.py
python src/06_fusion.py                # writes fusion_demo.png and fusion_demo_timeline.csv
python src/07_threshold_analysis.py
```
Run all commands from the repository root. Raw signal data is downloaded into `slpdb_data/` (git-ignored).

## Methodology
1. **Windows and labels.** Each overnight recording is cut into 30-second windows aligned with the database's expert annotations. A window is positive if it contains a hypopnea, obstructive apnea or central apnea code.
2. **Features.** From ECG: heart rate, RR variability (SDNN), and frequency-domain HRV (LF power, HF power, LF/HF ratio). From nasal airflow: breath rate, breath-interval variability, breath amplitude and peak-to-peak range.
3. **Consistent sensor modality.** Only records with a nasal airflow channel are used (15 of 18). Early experiments that mixed airflow and effort-based channels failed on the mixed subject, so this was a deliberate fix.
4. **Validation.** Leave-one-patient-out cross-validation throughout. A plain random split leaks near-identical neighbouring windows between train and test (an early run scored 0.94 AUC this way and collapsed to 0.65 once split by patient).
5. **Early warning.** Beyond detecting the current window, the model predicts whether an event occurs in the next ~90 seconds.
6. **Fusion.** Static risk (percentile of the synthetic cohort's weighted comorbidity score) and dynamic event probability are combined as `0.3 * static + 0.7 * dynamic` and mapped to LOW / MODERATE / HIGH.

## Results
Leave-one-patient-out ROC-AUC (13 scoreable subjects; the 2 healthy controls have no positive windows).

| Task | Mean ROC-AUC |
|---|---|
| Detect event in the current window | 0.687 (per-patient range 0.56 to 0.83) |
| Predict event within ~90 s, all windows | 0.722 |
| *Persistence baseline (current label only)* | *0.776* |
| **Predict event within ~90 s, currently-normal windows only** | **0.658** |
| Onset-only, 4 subjects with SpO2: without SpO2 | 0.660 |
| Onset-only, 4 subjects with SpO2: **with SpO2** | **0.730** |

**Alert thresholds** were chosen from out-of-fold (leave-one-patient-out) predictions after
causal smoothing (trailing 3-minute average), and validated against the two zero-event healthy
controls as a false-alarm check:

| Threshold | Alert rate | Precision | Recall | False-alarm rate (healthy subjects) |
|---|---|---|---|---|
| HIGH ≥ 0.65 | 22% | 0.86 | 0.41 | 10% |
| MODERATE ≥ 0.50 | 39% | 0.70 | 0.59 | 37% |

Smoothing improved every metric simultaneously versus the unsmoothed score (e.g. at a matched
~40% alert rate: precision 0.70 vs 0.66, healthy false-alarm rate 37% vs 50%), suggesting the
raw per-window score is noisy and a short trailing average meaningfully cleans it up.

**Fusion demonstration(`fusion_demo.png`):** On the same held-out recording (slp16, 51 events/hr), a synthetic
patient at the 89th percentile of EHR risk was flagged HIGH on 46% of windows, versus 8% for
a patient at the 23rd percentile — showing the static risk layer meaningfully shifts clinical
urgency even when the underlying physiological signal is identical.

## Conceptual Dashboard
[Add screenshots or mockup: patient list ranked by fused risk, overnight risk timeline, static risk factors panel, SpO2 panel where available.]

## Limitations and Future Work
- Modest absolute performance (onset-only AUC around 0.66). 30-second windows are short for frequency-domain HRV, which conventionally uses 2 to 5 minutes.
- Hyperparameters were tuned on all subjects before leave-one-patient-out evaluation, so reported scores carry mild optimism. Nested cross-validation is the fix.
- The SpO2 experiment covers only 4 severe-apnea subjects with no controls, and "normal" windows may include recovery from a preceding event.
- Real signals and synthetic EHR cannot be linked per patient, so fusion happens at the score level. The 0.3 / 0.7 weights and the flag thresholds are hand-set and illustrative, not learned or clinically calibrated. Demo pairings of a recording with a synthetic profile are illustrative only.
- Future work: longer windows, sequence models on raw signals, probability calibration, a larger and more diverse cohort, and prospective validation on consumer wearables.
- This is a research prototype, not a diagnostic device.

## License
[MIT / Apache-2.0: add a LICENSE file]

## Submission Materials
- Demo video (20+ min): [link]
- Architecture diagram: [link]
- Presentation: [link]