# TODO — Future improvements for wind power forecasting

Current best: Fold-5 **7.62%** (v16.1 CF + specialists). Leaderboard leader
at 7.57%. Remaining gap ~0.05 pp after v16.1 uploads.

Items below are ranked by **expected gain × probability of success / effort**.

---

## Tier 1 — High-value, low-risk (try first)

### 1. Ensemble blending of v14, v15, v16

- **Effort:** 30 min
- **Expected:** 0.02-0.05 pp
- **Rationale:** v14, v15, v16 use progressively improved configurations.
  They're correlated but not identical (CF target changes prediction scale).
  A weighted blend may smooth remaining errors.
- **Action:** Test `0.3·v14 + 0.3·v15 + 0.4·v16` and other weights.

### 2. ~~Fold-5 hyperparameter re-tune for v16 (CF target)~~ ✅ DONE → v29

- **Result:** Fold-5 7.5734% (+0.031 pp vs v16.1 baseline of 7.6041%)
- **Script:** `src/training/tune_v27_cf.py` (K-sweep + 100 Optuna trials)
- **Config:** `configs/model/lgbm_v29_cf_tuned.yaml`
- **Key finding:** Optuna best trial (38, lr=0.047) was 3-seed noise.
  Robust winner is trial 23 (lr=0.026, leaves=55, mdl=43, Fold-5=7.5734%).
  High-LR trials with few iterations are unstable across seed counts.

### 3. 10-seed specialists instead of 5-seed

- **Effort:** ~30 min
- **Expected:** 0.01-0.03 pp
- **Rationale:** Base uses 10 seeds, specialists only 5. More seeds per
  specialist = better variance reduction.
- **Action:** Simple code change in train_v15_regime.py. Training time
  triples (15 seeds × 3 specialists vs 5).

### 4. ~~Additional ERA5 rolling windows~~ ✅ DONE → no net gain

- **Ablation result (v30):**
  - +2h mean/std: Fold-5 7.5772% (+0.013 pp in ablation, rank 9)
  - +48h/72h: Fold-5 7.6361% (−0.046 pp) — displaces better features
  - +all: Fold-5 7.6529% (−0.063 pp)
- **End-to-end result (v31):** 7.6039% — flat vs baseline (7.6041%).
  The ablation gain was noise from shared probe; clean run shows no improvement.
- **Decision:** 2h window rejected. 48h/72h rejected. No windows added.
- **Lesson:** Ablation gains < 0.02 pp are unreliable; always verify end-to-end.

### 5. ~~Feature selection sweep around K=70~~ ✅ DONE → K=70 confirmed

- **Result:** K=70 is optimal (7.5792%). K=65 is 7.5866%, K=75 is 7.5896%.
  The sweet spot hasn't shifted with the CF target.
- **Script:** Phase 1 of `src/training/tune_v27_cf.py`

### 6. Upload v16 variants to get leaderboard readings

- **Effort:** 3 uploads
- **Expected:** measurement, not improvement
- **Rationale:** We have v16.0, v16.1, v16.2 ready but not yet scored.
  Knowing which of base/specialists/blend wins on leaderboard informs next
  steps.

---

## Tier 2 — Medium-value, medium-risk

### 7. Multi-step CV (not just Fold-5)

- **Effort:** 1-2 h
- **Expected:** 0.02-0.05 pp (if it shows true optimum differs from Fold-5)
- **Rationale:** We've been optimizing purely for Fold-5 (Q1 2025). It
  correlates ~0.97 with leaderboard but not perfectly. A multi-fold
  objective (weighted mean) may generalize better.
- **Action:** Tune on `0.5·fold5 + 0.25·fold4 + 0.15·fold3 + 0.1·fold2`.

### 8. Wake lookup with finer ws bins

- **Effort:** 1 h
- **Expected:** 0.02-0.04 pp
- **Rationale:** Current wake lookup uses coarse ws bins [0, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 14, 25]. Finer bins might capture wake physics more
  precisely, especially at 7-10 m/s where wake losses are strongest.
- **Action:** Try 0.5 m/s ws bins (50 bins × 16 sectors).

### 9. Pseudo-labels from best current prediction

- **Effort:** 2 h
- **Expected:** 0.03-0.05 pp
- **Rationale:** Use v16.1 Fold-5 predictions as additional training
  targets for a second-stage model. Variance reduction technique.
- **Action:** Generate OOF predictions for ALL folds, train residual model.

### 10. Target-encoded direction sector × month

- **Effort:** 1 h
- **Expected:** 0.02-0.03 pp
- **Rationale:** 8 direction sectors × 12 months = 96 buckets. Mean power
  in each bucket captures site-specific seasonal wind regimes (e.g.,
  winter northeasterly dominant). Time-safe target encoding.
- **Action:** Add `dir_sector × month` target-encoding feature (exclude
  current fold from encoding to prevent leakage).

### 11. Monthly or quarterly standardization of wind speed

- **Effort:** 30 min
- **Expected:** 0.01-0.02 pp
- **Rationale:** Wind speed climatology shifts by month. Adding
  `ws_120m − ws_120m_monthly_mean` as a feature may help model learn
  anomaly-based patterns.

### 12. Try LightGBM with `boosting_type="dart"`

- **Effort:** 30 min
- **Expected:** 0.02-0.05 pp (or nothing)
- **Rationale:** DART (Dropouts Additive Regression Trees) prevents
  over-specialization of late trees. Worth a quick CV check.

### 13. Huber loss instead of MAE

- **Effort:** 30 min
- **Expected:** 0.01-0.03 pp
- **Rationale:** Huber is less sensitive to the 242 outlier rows we excluded.
  May help if some outliers remain.

---

## Tier 3 — Higher-effort, uncertain value

### 14. Conformal prediction for uncertainty-aware clipping

- **Effort:** 3 h
- **Expected:** 0 pp direct, but may let us use wider pred range
- **Rationale:** Literature review §6 validated CACP-KNN for wind
  forecasting. Main value is disqualification-threshold management,
  not point accuracy.

### 15. Custom LGBM loss: asymmetric (under/over)

- **Effort:** 2 h
- **Expected:** 0.01-0.03 pp
- **Rationale:** Residual analysis showed under-prediction bias at
  10-14 m/s (~3 MW). A loss function with stronger penalty on
  under-prediction in that regime might help. Requires custom gradient.

### 16. TFT with encoder target replaced by LGBM predictions

- **Effort:** 4 h
- **Expected:** 0.05-0.15 pp if it works, 0 pp if it doesn't
- **Rationale:** Our TFT experiments showed val nMAE 4.49% with perfect
  target history. The barrier is autoregressive drift. Using LGBM's
  in-distribution predictions as "target history" for TFT encoder might
  give the temporal signal without drift.
- **Risk:** High failure risk — tested similar with pseudo-labels, failed.

### 17. Small MLP residual head on top of LGBM

- **Effort:** 4 h
- **Expected:** 0.02-0.05 pp, risk of overfitting
- **Rationale:** Some literature (Gijón 2025) reports improvement from
  adding a shallow neural network to predict LGBM residuals.
- **Action:** Train MLP(LGBM_pred, all_features) → residual, sum with
  LGBM prediction.

### 18. Build custom LGBM with monotonic constraints

- **Effort:** 2 h
- **Expected:** 0.01-0.03 pp
- **Rationale:** Physically, power should be monotonic in wind speed
  (up to rated). LightGBM supports `monotone_constraints` per feature.
  Would help in low-data regimes (very low or very high wind).

### 19. Variable-height wind interpolation

- **Effort:** 3 h
- **Expected:** 0.01-0.02 pp
- **Rationale:** Instead of treating 10/80/120/180 m winds as
  independent features, fit a log-linear profile and use parameters
  (exponent, intercept, residual) as features.

### 20. ERA5-Land (11 km, higher surface resolution)

- **Effort:** 1 h (fetch + merge)
- **Expected:** 0.02-0.05 pp (unknown — ERA5 already helps a lot)
- **Rationale:** Higher spatial resolution than ERA5 25 km. May capture
  coastal effects better (our farm is near Sea of Azov).
- **Action:** Fetch via Open-Meteo API, add as parallel to existing ERA5.

---

## Tier 4 — Experimental / speculative

### 21. Fine-tune a small pre-trained time-series foundation model

- **Effort:** 1-2 days
- **Expected:** Unknown — could be 0 or significant
- **Rationale:** TimesFM 2.0 or Chronos-Bolt have ~200M params, first-class
  covariate support. Fine-tuning on wind data could transfer knowledge
  from thousands of other time series.
- **Risk:** GPU time + new tooling complexity.

### 22. Gradient-boosted quantile regression

- **Effort:** 4 h
- **Expected:** 0.02-0.05 pp if used in blend
- **Rationale:** Train P25, P50, P75 quantile regressions. The median
  (P50) often beats MAE-trained predictions in heavy-tailed regimes.
  Plus, quantile ensembling can reduce variance.

### 23. Spatio-temporal model with simulated neighbor turbines

- **Effort:** 3-5 days
- **Expected:** Unknown
- **Rationale:** Our wake lookup is 2D (direction × speed). A full GNN
  would need per-turbine data we lack, but could work with simulated
  turbine positions based on typical wind-farm layouts.

### 24. Retry quantum NN for fun

- **Effort:** Any time wasted is too much
- **Expected:** 0 pp
- **Rationale:** Paper from QNN review was pseudoscience for this domain.
  Do not pursue.

---

## Decision gates & stop conditions

### Stop conditions for further tuning

- **Leaderboard plateau:** if 5 consecutive submissions all within 0.02 pp,
  stop tuning and ship best.
- **Fold-5 regression:** if a Tier 1 change drops Fold-5 by >0.05 pp,
  revert.
- **Training time:** any experiment requiring >4 h on single machine should
  require explicit user approval first.

### Final submission checklist

Before submitting `forecast.csv`:
- [ ] All `pytest tests/` pass
- [ ] Submission CSV has exactly 2126 rows
- [ ] No header, no index column
- [ ] All values ∈ [0, 90.09]
- [ ] No NaN or inf
- [ ] Mean prediction in [30, 45] MW range (sanity bound)
- [ ] Row order matches valid_features.csv
- [ ] Model weights and code committed to Git

### Preparation for deadline day

- [ ] Lock current best (v16.1) as Git tag `baseline-best`
- [ ] Document exact rerun command path
- [ ] Test repro on clean env (Docker) at T-1 day
- [ ] Jury note / writeup with key decisions

---

## Final notes

**What we've proven works for this problem (reusable insights):**
- External reanalysis data (ERA5) > in-sample feature engineering
- Rolling weather statistics > instantaneous values
- Capacity-factor target > raw MW target for maintenance-variant data
- Regime-weighted specialist ensemble > single model with sample weights
- Feature count has a sweet spot around 60-70 (more can hurt)

**What we've proven doesn't work for this problem:**
- Autoregressive target lags (compounding error over 2000+ hours)
- TFT / LSTM / deep sequential models (AR drift)
- Global bias calibration (fold-variable bias)
- Per-month affine calibration on OOF folds 1–4 (same fold-variable bias issue, +0.16 pp)
- Season-specialized training (loses data volume)
- Multi-algorithm ensembles with LGBM as member (other models drag it down)
- Hard or soft cut-in post-processing (model learns better internally)
- 1-D Kalman filter on NWP↔ERA5 bias as features (−0.033 pp; redundant with existing bias features)
- NWP-based ramp features dv_3h/dv_6h (−0.027 pp; ERA5 rolling diffs already cover this)
