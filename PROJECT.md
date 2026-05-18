# Wind Power Forecasting — ARVE 2026 Hackathon

Project description for a hackathon-ready hourly wind power forecasting pipeline
targeting the ARVE 2026 benchmark (90.09 MW wind farm, Q1 2026 prediction horizon).
This document describes the current state, architecture, features, and experimental
history.

---

## 1. Problem

**Inputs:**
- `train_dataset.csv` — 32 434 hourly rows covering 2022-01-01 to 2025-12-31,
  with target `Выработка. Результирующий расчет` (generation in MW, range
  0.001–87.475) and 20 weather/metadata columns.
- `valid_features.csv` — 2 126 hourly rows for Q1 2026 (Jan-Mar), same features,
  target unknown.

**Output:** single-column CSV with 2 126 predictions, no header, no index,
values clipped to `[0, 90.09]`.

**Metric:** normalized MAE over installed capacity:
```
nMAE (%) = mean(|y_true - y_pred|) / 90.09 * 100
```

**Farm metadata (from datasheet):**
- 26 turbines × Siemens Gamesa SG 3.4-132 (3.465 MW rated each)
- Hub height: 80 m, rotor diameter 132 m
- Coordinates: 46.827°N, 38.718°E (Sea of Azov, Russia)
- Wind class IEC IIA, cut-in ~3 m/s, rated 12 m/s

---

## 2. Current architecture (v16 — best Fold-5: 7.62%)

```
                    raw CSV (train + valid)
                            │
                            ▼
             ┌──────────────────────────────┐
             │ ERA5 reanalysis merge        │
             │ (Open-Meteo Historical API)  │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │ Feature engineering          │
             │ • multi-level wind + shear   │
             │ • REWS, v_eff, Hellmann α    │
             │ • air density, power curve   │
             │ • datasheet theoretical P    │
             │ • empirical wake correction  │
             │ • ERA5 rolling 3/6/12/24h    │
             │ • ERA5 pressure/wind ramps   │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │ Feature selection K=70       │
             │ (importance from probe run)  │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │ Target transform: MW → CF    │
             │ CF = P / (active × 3.465)    │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │ LightGBM ensemble            │
             │ ┌────────────────┐           │
             │ │ Base (10 seeds)│           │
             │ │                │           │
             │ │ Specialist_low │           │
             │ │  (ws<7, 5 seeds)│──── avg  │
             │ │                │     ─────│
             │ │ Specialist_mid │           │
             │ │  (ws 4-12)     │           │
             │ │                │           │
             │ │ Specialist_high│           │
             │ │  (ws 8-25)     │           │
             │ └────────────────┘           │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │ Reverse: CF → MW             │
             │ P = CF × P_available         │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │ Clip to [0, 90.09]           │
             │ Restore original row order   │
             │ Atomic CSV write             │
             └──────────────────────────────┘
```

---

## 3. Data layer

### 3.1 Primary inputs

| File | Rows | Notes |
|---|---|---|
| `train_dataset.csv` | 32 434 | 2022-01-01 → 2025-12-31, ~489 gap-hours |
| `valid_features.csv` | 2 126 | 2026-01-01 → 2026-03-31, 34 missing hours of the 2 160 theoretical |

### 3.2 External data (ERA5 reanalysis via Open-Meteo API)

Cached in `data/external/era5_reanalysis.parquet` (37 224 rows covering
2022-01-01 to 2026-03-31). Variables:
- `wind_speed_10m`, `wind_speed_100m`
- `wind_direction_10m`, `wind_direction_100m`
- `wind_gusts_10m`
- `temperature_2m`, `pressure_msl`, `cloud_cover_low`
- `rain`, `snowfall`

ERA5 is ECMWF's reanalysis (25 km resolution, assimilates observations).
Critically different from the training data's NWP source — correlation of
training ws_80m with ERA5 ws_100m is only 0.81, with mean diff of −0.57 m/s,
showing ERA5 provides genuine information not in training.

### 3.3 Quality issues handled

- **180 m columns missing** for Jan-Nov 2022 (6 836 rows). Imputed via
  Hellmann power law from 120 m.
- **~489 gap-hours** in training (mostly 2-3 hour gaps).
- **~242 physically impossible rows** (all wind levels < 4 m/s but power >
  20 MW) — excluded from power-curve fitting, kept in LGBM training.
- **34 missing hours in valid** — submission format uses ONLY the 2 126
  provided rows; we never infer for missing hours.

---

## 4. Feature engineering (~135 engineered, K=70 after selection)

### 4.1 Physics features (always kept in top-20 by importance)

| Feature | Formula | Notes |
|---|---|---|
| `v_eff` | `ws_120m × (ρ/1.225)^(1/3)` | Density-corrected hub-height wind |
| `v_eff_cube` | `v_eff³` | Dominant power term |
| `rews` | `((ws_80³ + ws_120³ + ws_180³)/3)^(1/3)` | Rotor-Equivalent Wind Speed |
| `air_density` | `P/(287.05·(T+273.15))` | Ideal gas at 80 m |
| `hellmann_alpha` | `ln(ws_120/ws_10) / ln(12)` | Atmospheric stability exponent |
| `gust_ratio_10m` | `gust / (ws_10 + ε)` | Turbulence intensity proxy |

### 4.2 Isotonic power curves (per-sector, fit on clean training data)

- `p_curve_sector` — isotonic regression on `v_eff` per 8 direction sectors
- `p_curve_global` — single global isotonic curve
- `p_curve_x_active` — scaled by active-turbine ratio
- `p_curve_sector_minus_global` — sector-specific anomaly

### 4.3 Manufacturer datasheet features (Siemens Gamesa SG 3.4-132)

Bilinear interpolation of the factory power curve (wind speed × air density,
9 density values from 1.06 to 1.27 kg/m³):

- `ds_80m_farm_mw`, `ds_120m_farm_mw`, `ds_era5_100m_farm_mw` — per-turbine
  power × active turbines
- `ds_consensus_farm_mw` — mean across heights
- `ds_consensus_wake_corrected` — consensus × empirical wake factor

### 4.4 ERA5 base features

| Feature | Notes |
|---|---|
| `era5_wind_speed_100m` | Independent estimate of hub-height wind |
| `era5_wind_gusts_10m` | Independent gust estimate |
| `era5_pressure_msl`, `era5_temperature_2m` | Independent atmospheric state |
| `era5_dir100_sin`, `era5_dir100_cos` | ERA5 direction encoding |
| `ws10_bias` | `ws_10 − era5_ws_10` — NWP forecast error signature |
| `ws100_vs_80` | `era5_ws_100 − ws_80` — cross-source disagreement |
| `gust_bias`, `pressure_bias` | Source disagreement in other vars |

### 4.5 ERA5 rolling/temporal features (added in v14, critical)

Top feature by importance: `era5_ws100_roll_mean_6h` (405 k gain). Captures
sustained wind regime, not just instantaneous value.

- `era5_ws100_roll_mean_{3,6,12,24}h`
- `era5_ws100_roll_std_{3,6,12,24}h`
- `era5_ws100_diff_{1,3,6}` — ramp rate
- `era5_pressure_diff_{3,6,12}` — pressure tendency (frontal passage)
- `era5_turb_intensity_6h` — turbulence intensity from rolling stats
- `era5_dir_sin_diff1`, `era5_dir_cos_diff1` — direction change rate

### 4.6 Wake correction (empirical, per direction × wind-speed cell)

Learned lookup table from clean training rows:
```
wake_factor[dir_sector_16, ws_bin] = median(actual / datasheet_theoretical)
```
Applied to datasheet predictions to produce `ds_consensus_wake_corrected`
which becomes the #1 most important feature after wake is added.

### 4.7 Extras (wind vectors, cross-height shears)

- `ws120_u`, `ws120_v` — u/v components at hub
- `era5_ws100_u`, `era5_ws100_v` — ERA5 u/v components
- `ws_shear_{10_80, 80_120, 120_180, 10_120, 10_180, 80_180}` — all 6 cross-height shears
- `regime_calm`, `regime_normal`, `regime_high` — soft wind regime membership

### 4.8 Calendar

- `hour_sin/cos`, `doy_sin/cos`, `month_sin/cos`, `dow_sin/cos`
- `year_index`, `is_weekend`, `is_winter`, `is_q1`

### 4.9 Static / maintenance

- `active_turbines_ratio`, `maintenance_ratio`
- `rews_cube_x_active`, `v_eff_cube_x_active`

---

## 5. Model

### 5.1 LightGBM — tuned hyperparameters (after 100 Optuna trials on v14)

```python
num_leaves       = 86
min_data_in_leaf = 14
learning_rate    = 0.00844
feature_fraction = 0.430
bagging_fraction = 0.564
bagging_freq     = 3
lambda_l1        = 0.253
lambda_l2        = 0.00971
num_boost_round  = 5000
early_stopping_rounds = 250
deterministic    = True
force_col_wise   = True
objective        = "regression_l1"  # MAE
```

### 5.2 Target transform: capacity factor

Instead of predicting raw MW directly, predict capacity factor:
```
CF = P / (N_active × 3.465 MW)
```
This removes the maintenance confound from the learning signal. Model only
has to learn wind → CF relationship, not wind × maintenance → MW.

Reverse transform at inference:
```
P = CF × P_available × active_turbines_ratio_valid
```

### 5.3 Ensemble: base + 3 regime specialists

| Model | Seeds | Sample weights | Notes |
|---|---|---|---|
| Base | 10 | uniform | General-purpose, all training |
| low_0_7 | 5 | 2.0 if ws in [0, 7), else 0.3 | Low-wind specialist |
| mid_4_12 | 5 | 2.0 if ws in [4, 12), else 0.3 | Transition/ramp specialist |
| high_8_25 | 5 | 2.0 if ws in [8, 25), else 0.3 | Rated-power specialist |

Final prediction = **average of 3 specialists** (Fold-5: 7.62%).

### 5.4 Feature selection

Top-70 features selected by LGBM `gain` importance from a probe run on the
full 135-feature set. Fixed set used by all ensemble members.

---

## 6. Validation

### 6.1 Walk-forward expanding folds

| Fold | Train end | Val window | Purpose |
|---|---|---|---|
| 1 | 2022-12-31 | 2023 Q1 | Winter generalization |
| 2 | 2023-09-30 | 2023 Q4 | Autumn |
| 3 | 2024-03-31 | 2024 Q2 | Spring (hardest, ~11.7%) |
| 4 | 2024-09-30 | 2024 Q4 | Autumn |
| 5 | 2024-12-31 | 2025 Q1 | **Q1 2026 surrogate (primary optimizer)** |

24-h embargo at fold edges, enforced by `tests/test_splits.py`.

### 6.2 Post-processing

- **No hard cut-in**: tested hard zero at v_eff<3.0 and soft sigmoid —
  both hurt Fold-5 by 0.03%+. Model handles cut-in transitions itself.
- **Cut-out**: original hard clamp removed (datasheet shows turbine still
  produces 63 MW at 25 m/s; our valid data never exceeds 22 m/s).
- **Final clip**: `[0, 90.09]`.

---

## 7. Leaderboard progression

Fold-5 surrogate to leaderboard ratio averages ~0.96-0.97 (leaderboard
slightly better because Q1 2026 is less noisy than Q1 2025).

| Version | Fold-5 | LB (when uploaded) | Key change |
|---|---|---|---|
| v0.1 baseline | 8.88 | 9.086 | LightGBM default + 62 features |
| v0.2 Optuna | 8.78 | — | 80-trial Optuna tune |
| v0.7 physics | 8.75 | — | REWS, v_eff, isotonic power curve |
| v0.8 v7 final | 8.75 | **8.712** | Physics + K=88 features |
| v2.0 ERA5 base | 8.15 | **7.924** | +ERA5 reanalysis features (-0.6pp breakthrough) |
| v3.1 blend | — | **7.919** | v3 advanced + v2 ERA5 |
| v8.0 wake | 7.97 | **7.936** | +Empirical wake correction (datasheet * wake factor) |
| v10.1 | — | **7.880** | Blend 60% v3.1 + 40% v8.0 |
| v14.0 | 7.84 | — | +ERA5 rolling features + re-tuned HPs |
| v15.2 specialists | 7.74 | **7.689** | +3 regime-weighted specialists averaged |
| **v16.1 CF + specialists** | **7.62** | **pending** | **+Capacity-factor target transform** |
| v27 K-sweep | 7.5792 | — | K=70 confirmed optimal with CF target |
| **v29 Optuna-tuned CF** | **7.5734** | **pending** | **+Optuna 100-trial re-tune for CF target** |

---

## 8. Repository structure

```
win_d/
├── briefing/                # Original PDFs + task brief
├── articles/                # Reference papers downloaded during research
├── data/
│   ├── raw/                 # train_dataset.csv, valid_features.csv
│   ├── external/            # era5_reanalysis.parquet (from Open-Meteo API)
│   └── processed/           # OOF predictions cache
├── src/
│   ├── data/
│   │   ├── loaders.py       # CSV loaders with row-order preservation
│   │   ├── schema.py        # Column constants + capacity
│   │   ├── splits.py        # Walk-forward folds
│   │   └── outliers.py      # Impossible-row identification
│   ├── features/
│   │   ├── pipeline.py      # Main feature builder (base + physics)
│   │   ├── physics.py       # Air density, REWS, v_eff, isotonic curves
│   │   ├── power_curve.py   # Binned empirical power curve (v3+)
│   │   ├── datasheet_power_curve.py  # Siemens Gamesa manufacturer curve
│   │   ├── wake.py          # Empirical wake correction lookup
│   │   ├── extras.py        # Wind vectors, cross-height shears, regimes
│   │   ├── era5_features.py # Advanced ERA5 derivations (v3+)
│   │   ├── curtailment.py   # (experimental, proved not useful)
│   │   ├── autoreg_lags.py  # (experimental, proved not useful)
│   │   └── bias_correction.py  # (experimental, proved not useful)
│   ├── models/
│   │   └── lightgbm_model.py  # LGBMConfig + train_lgbm + predict_lgbm
│   ├── training/
│   │   ├── train_lgbm.py              # v0.1 baseline
│   │   ├── train_lgbm_tuned.py        # v0.2 Optuna-tuned
│   │   ├── train_final.py             # v1.2 consolidated
│   │   ├── train_era5_final.py        # v2.0 ERA5 base
│   │   ├── train_era5_v3.py           # v3.0 advanced ERA5
│   │   ├── train_v8_wake.py           # v8.0 wake correction
│   │   ├── train_v15_regime.py        # v15 regime specialists
│   │   ├── tune_*.py                  # Optuna tuning scripts
│   │   └── ...                        # Other experimental variants
│   ├── inference/
│   │   ├── submission.py    # Atomic CSV writer + validation
│   │   └── predict_*.py     # Model-specific predictors
│   ├── eval/
│   │   └── metrics.py       # normalized_mae, mae
│   └── utils/
│       └── seeding.py       # Deterministic seed setting
├── scripts/                 # Exploratory / diagnostic scripts
├── submissions/archive/     # All submission CSVs (gitignored)
├── models/                  # Trained model files (gitignored)
├── configs/                 # Hydra + tuned params YAMLs
├── tests/                   # pytest tests
├── environment.yml          # Conda environment
├── requirements.txt         # Pip mirror
├── pyproject.toml           # ruff + pytest config
└── README.md
```

---

## 9. What worked

### Big wins (≥0.1 pp each on Fold-5)

1. **ERA5 reanalysis features** (v2.0): 8.75 → 8.15 (**-0.6 pp**). Independent
   weather estimate via Open-Meteo Historical API. Biggest single improvement
   in the project. Training data correlation with ERA5 is only 0.81, showing
   the two sources contain genuinely different information.

2. **ERA5 rolling mean 6h** (v14): 8.15 → 7.84 (**-0.1 pp**). Captures
   sustained wind regime better than instantaneous value. Became #1 feature
   with 405k gain importance.

3. **Regime specialists averaged** (v15): 7.84 → 7.74 (**-0.1 pp**). Three
   LGBMs trained with different wind-speed region weights. Averaging
   captures more diversity than multi-seed of the same config.

4. **Capacity-factor target** (v16): 7.74 → 7.62 (**-0.12 pp**). Train on
   `CF = P / P_available` instead of raw MW. Separates wind→CF learning from
   maintenance derate, giving a cleaner signal.

### Medium wins (0.02-0.1 pp)

5. **Physics-informed features** (v0.7): REWS, v_eff, air density, isotonic
   per-sector power curve — became ~30% of importance budget.

6. **Manufacturer datasheet power curve** (v6): bilinear interpolation of
   Siemens Gamesa SG 3.4-132 factory curve. Captures exact aerodynamic
   physics at 9 air densities.

7. **Empirical wake correction** (v8): `wake_factor[dir_sector × ws_bin] =
   median(actual / datasheet)`. Applied to datasheet → corrected farm-level
   prediction. Became #1 feature at 367k gain.

8. **Feature selection K=70** (v5+): dropping bottom-importance features
   reduces noise. Optimal K shifts with feature set (K=60 for v6, K=70 for
   v14+).

9. **Physically impossible row filtering**: 242 rows with ws<4 but
   power>20 MW excluded from power-curve fitting (kept in LGBM training).

10. **Retuned Optuna per feature set**: optimal hyperparameters shift
    dramatically with feature additions (v2: 453 leaves, v14: 86 leaves).

### Small wins (<0.02 pp)

11. **Outlier exclusion from power curve fit only** (v9): keeps all training
    data for LGBM but gives it cleaner curve features.

12. **Multi-seed ensembling (10 seeds)**: steady ~0.02 pp improvement over
    single seed, plateaus beyond 5 seeds.

13. **Wind vector (u, v) components** at 120 m (v7): modest contribution,
    kept.

14. **REWS** (rotor-equivalent wind speed integrating 80/120/180m cubes):
    stronger than any single-height wind.

---

## 10. What did NOT work

### v26 ablation — plan.md methods (May 2026)

Three methods from the attached plan.md were evaluated on Fold-5 via
`src/training/train_v26_kalman_ramps.py`. All hurt Fold-5 vs. the
7.6041% baseline (v15-regime recipe with CF target):

27. **1-D Kalman filter on NWP↔ERA5 wind-speed bias** (`src/postprocess/kalman.py`):
    Fold-5 7.6374% (−0.033 pp). The smoothed bias features `ws120_kalman`
    and `ws120_bias_kalman_smooth` did not enter the top-70 by gain
    importance. The booster already captures NWP↔ERA5 disagreement through
    `ws10_bias`, `ws100_vs_80`, `pressure_bias`. Kalman adds correlated
    noise on top.

28. **Ramp features** (`src/features/ramp.py`): `dv_3h`, `dv_6h`,
    `is_sub_cutin`, `is_sub_cutin_soft` and interactions. Fold-5 7.6313%
    (−0.027 pp). `dv_3h` and `dv_6h` reached ranks 48 and 53 in top-70
    but net effect is negative. ERA5 rolling diffs (`era5_ws100_diff3/6`)
    already cover the ramp signal from an independent source; NWP-based
    diffs add correlated noise.

29. **Per-month affine calibration** (`src/postprocess/calibration.py`):
    Fit on OOF folds 1–4, applied to Fold-5. Baseline 7.7656% (+0.16 pp
    worse). Consistent with the known failure mode (PROJECT.md §10 item 11):
    fold-to-fold bias patterns differ, so calibration learned on folds 1–4
    does not transfer to Fold-5.

All three modules remain in `src/` as documented experiments. None are
used in the production pipeline.

---


### Time-series extensions (failed because autoregressive drift kills Q1 accuracy)

1. **Target-power lags** (v3 + phases A-F): Fold-5 *optimistic* drops to
   **5.07%** with true lag values, but rolling autoregressive inference on
   2000+ hours compounds error to 10.5%. Hybrid/blended versions all worse
   than no-lag baseline.

2. **TFT (Temporal Fusion Transformer)**: val nMAE 6.87% with target in
   encoder, but fails at inference (mean 31 MW vs 39 MW baseline — severe
   underprediction from AR drift). Non-AR variant trained on weather-only
   predicts near-zero at inference because the model learned to rely on
   encoder target.

3. **Scheduled sampling / teacher forcing with noise**: injecting Gaussian
   noise into training lags made rolling inference worse, not better.

### NWP / weather data extensions

4. **GFS and ICON models via Open-Meteo**: essentially identical to ERA5
   (correlation 1.0000 for common variables) or had too many NaN values.
   No diversity gained.

5. **CERRA (5 km European reanalysis)**: only covers up to June 2021, so
   unusable for our 2022-2026 period.

6. **Historical forecast archives per NWP model (GFS/ICON/GEM)**: GFS gave
   identical data to ERA5, ICON had missing hours, GEM similar. No gains.

### Training-data strategies

7. **Seasonal/Q1-only training**: training only on Jan-Mar months loses
   2× data, Fold-5 gets worse by 0.2-0.5 pp. Model benefits more from
   all-season data than from specialization.

8. **Summer down-weighting**: tested weights 0.0 to 1.0 on Apr-Sep rows.
   Equal weighting (1.0) optimal — any down-weighting hurt Fold-5.

9. **Winter up-weighting**: tested 1.25 to 2.0 on Dec-Feb. All worse than
   equal weighting.

10. **2022 down-weighting**: rows with 180m NaN slightly less informative,
    but down-weighting still hurt (even outlier-only zero-weighting was
    break-even at best).

### Post-processing

11. **OOF bias calibration** (isotonic regression on ws bins): fold-to-fold
    bias patterns differ, so global calibration learned from Folds 1-4 hurts
    Fold-5 by +0.10 pp.

12. **Hard cut-in at v_eff<3**: loses accuracy in the 2-3 m/s regime where
    actual power is still ~3 MW due to aggregation across 26 turbines.

13. **Soft sigmoid cut-in** (center=2.6, width=1.2): slightly worse than
    no cut-in at all.

### Feature additions that hurt

14. **Weather lag features** (ws_80m_lag_1/2/3/6/12/24): hurt Fold-5 by 0.1
    pp despite being physically motivated. Model over-relies on them and
    they don't generalize across year boundaries.

15. **Curtailment risk indicators** (pressure drop, ws ramp, maintenance
    thresholds): 14 new features added, Fold-5 worse by +0.16 pp.

16. **Regime cluster labels as features** (KMeans on shear + ws + density):
    cluster label ranked #144 in importance (near-zero contribution).
    Useful as training weight (v15) but not as feature.

17. **Half-day cyclical encoding**: redundant with hour sin/cos.

18. **Wind speed acceleration**: redundant with existing shear features.

19. **Atmospheric stability categorical**: redundant with Hellmann α.

### Model diversity experiments

20. **LGBM + CatBoost + XGBoost ensemble** (v0.6): ensemble 8.83%
    vs LGBM-only 8.78%. CatBoost/XGBoost are weaker on this problem;
    averaging drags best model down.

21. **3 LGBM configs (default + deep + wide) ensemble**: slightly worse
    than pure tuned config. Seed-based diversity is sufficient.

22. **Multi-model stacking with ridge**: stacker gives LGBM ~76% weight
    but still can't beat LGBM alone.

23. **2-stage residual model**: training second LGBM on first's residuals
    —not yet tested cleanly, likely minor.

### Other approaches

24. **Isotonic calibration on OOF**: hurt due to fold-varying bias.

25. **ERA5-Land (11 km)**: untested — likely similar to ERA5 25 km.

26. **Quantile regression (P10/P50/P90)**: untested on final setup.

---

## 11. Quick reference: key parameters and files

### Best submission (v16.1)

- Training script: Embedded inline in session (not saved as standalone
  script at this point — rerun needs reconstruction from `train_v15_regime.py`
  + CF transform logic).
- Submission: `submissions/archive/v16.1_cf_specialists.csv`
- Fold-5: 7.6202%
- Uses: ERA5 + rolling + datasheet + wake + extras + 70 features + CF target
  + 3 regime specialists averaged.

### Key constants

```python
CAPACITY_MW     = 90.09
TOTAL_TURBINES  = 26
TURBINE_RATED   = 3.465  # MW per turbine
HUB_HEIGHT_M    = 80.0
LAT, LON        = 46.8268, 38.7179
```

### Reproduction commands

```bash
# Prepare data (ERA5 fetch, one-time)
python scripts/fetch_era5.py

# Train v14 (ERA5 rolling + regime specialists, Fold-5 ~7.74%)
python -m src.training.train_v15_regime

# Train v16 variant (CF-target + regime specialists, Fold-5 ~7.62%)
# — requires rebuilding from v15 with target transformation
```

---

## 12. Acknowledgements

- **Open-Meteo** for free ERA5 Historical Weather API access (critical
  external data source).
- **Siemens Gamesa** for publishing the SG 3.4-132 datasheet with air-density-
  dependent power curves at 9 densities.
- Referenced papers (in `briefing/` and `articles/`):
  - Wang et al. 2026 — physics-constrained transformer motivation
  - Ally et al. 2025 — modular DL + wake loss
  - Jachuła & Wydra 2025 — TFT wind prediction (Poland)
  - Gijón et al. 2025 — hybrid physics+residual architecture
  - Frontiers 2025 — regime-aware multi-altitude forecasting
