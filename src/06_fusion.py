"""Step 6: Fusion layer.

Trains the final early-warning model and saves it, then runs a demo: one real overnight
recording (held out from training) paired with a high-risk and a low-risk synthetic patient.
Real signals and synthetic EHR cannot be linked per patient, so the pairing is illustrative.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

FEATURES = ['heart_rate', 'rr_sdnn', 'n_r_peaks', 'lf_power', 'hf_power', 'lf_hf_ratio',
            'breath_rate', 'breath_interval_std', 'n_breaths',
            'breath_amplitude_mean', 'breath_amplitude_std', 'resp_peak_to_peak']
PARAMS = {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 3,
          'learning_rate': 0.1, 'colsample_bytree': 1.0}

DEMO_RECORD = 'slp16'
W_STATIC, W_DYNAMIC = 0.3, 0.7      # hand-set, illustrative
SMOOTH = 6
T_HIGH, T_MODERATE = 0.65, 0.50       # hand-set, illustrative


def train(data):
    X, y = data[FEATURES], data['future_label']
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = XGBClassifier(**PARAMS, eval_metric='logloss', random_state=42, scale_pos_weight=spw)
    return clf.fit(X, y)


def fuse(static_pct, dynamic_prob):
    """static_pct: patient's percentile in the synthetic cohort (0-1).
    dynamic_prob: model probability of an event in the next ~90 s (0-1)."""
    final = W_STATIC * np.asarray(static_pct, dtype=float) + W_DYNAMIC * np.asarray(dynamic_prob, dtype=float)
    flag = np.select([final >= T_HIGH, final >= T_MODERATE], ['HIGH', 'MODERATE'], default='LOW')
    return final, flag


def main():
    df = pd.read_csv('osa_predictive_dataset.csv')

    # 1) final model on all subjects (for the dashboard)
    joblib.dump({'model': train(df), 'features': FEATURES}, 'osa_dynamic_model.pkl')
    print("Saved osa_dynamic_model.pkl (trained on all subjects)")

    # 2) demo model trained WITHOUT the demo recording, so the demo is honest
    demo_model = train(df[df['record'] != DEMO_RECORD])
    demo = df[df['record'] == DEMO_RECORD].sort_values('window_index').reset_index(drop=True)
    dyn = demo_model.predict_proba(demo[FEATURES])[:, 1]
    dyn = pd.Series(dyn).rolling(SMOOTH, min_periods=1).mean().values   # causal smoothing
    minutes = demo['window_index'].values * 0.5

    # 3) pick a high-risk and a low-risk synthetic patient by cohort percentile
    ehr = pd.read_csv('synthetic_ehr_osa_risk.csv')
    ehr['static_pct'] = ehr['risk_score'].rank(pct=True)
    profiles = {}
    for tag, target in [('high_risk', 0.90), ('low_risk', 0.15)]:
        row = ehr.iloc[(ehr['static_pct'] - target).abs().argmin()]
        profiles[tag] = row
        print(f"{tag}: patient {str(row['Id'])[:8]}... risk_score={row['risk_score']:.0f}, "
              f"cohort percentile={row['static_pct']:.2f}")

    # 4) fuse and build timeline
    timeline = pd.DataFrame({
        'minute': minutes,
        'event_now_truth': demo['label'],
        'event_within_90s_truth': demo['future_label'],
        'dynamic_prob': dyn,
    })
    for tag, row in profiles.items():
        final, flag = fuse(row['static_pct'], dyn)
        timeline[f'fused_{tag}'] = final
        timeline[f'flag_{tag}'] = flag
        print(f"\nFlags for {tag} profile on {DEMO_RECORD}:")
        print(pd.Series(flag).value_counts().to_string())
    timeline.to_csv('fusion_demo_timeline.csv', index=False)

    # 5) plot
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    ev = demo['label'].values == 1
    axes[0].plot(minutes, dyn, color='tab:blue', lw=1)
    axes[0].fill_between(minutes, 0, 1, where=ev, color='red', alpha=0.15, step='mid',
                         label='Annotated apnea/hypopnea')
    axes[0].set_ylabel('Dynamic event probability')
    axes[0].set_title(f'Dynamic model output, held-out recording {DEMO_RECORD}')
    axes[0].legend(loc='upper right')

    axes[1].plot(minutes, timeline['fused_high_risk'], color='tab:red', lw=1,
                 label=f"High-risk profile (pct {profiles['high_risk']['static_pct']:.2f})")
    axes[1].plot(minutes, timeline['fused_low_risk'], color='tab:green', lw=1,
                 label=f"Low-risk profile (pct {profiles['low_risk']['static_pct']:.2f})")
    axes[1].axhline(T_HIGH, color='k', ls='--', lw=0.8)
    axes[1].axhline(T_MODERATE, color='gray', ls=':', lw=0.8)
    axes[1].text(minutes[0], T_HIGH + 0.01, 'HIGH', fontsize=8)
    axes[1].text(minutes[0], T_MODERATE + 0.01, 'MODERATE', fontsize=8)
    axes[1].set_ylabel('Fused risk')
    axes[1].set_xlabel('Minutes into recording')
    axes[1].set_title('Same signal, different static risk: fused clinical flag')
    axes[1].legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('fusion_demo.png', dpi=150)
    print("\nSaved fusion_demo.png and fusion_demo_timeline.csv")


if __name__ == '__main__':
    main()