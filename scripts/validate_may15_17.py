"""Backtest: predict May 15-17 using the same pipeline as predict_18_05.py.

Purpose
-------
Validate model quality on RECENT LOW-WIND days where actuals are known.
Also diagnoses ERA5 reliability for today (May 18): was ERA5 accurate vs NWP
for May 15-17, or is it diverging from reality?

Key concern: Open-Meteo archive API has ~5-day lag for true ERA5 reanalysis.
For May 15-17 (3-5 days ago) and especially May 18 (today), the returned
"ERA5" data may be ECMWF/GFS forecast-based, not actual observation-assimilated
reanalysis. If ERA5 overestimated winds vs actuals for May 15-17, we should
weight it down for May 18.

Output
------
  BACKTEST nMAE, per-hour predictions vs actuals, ERA5 vs NWP reliability table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TOTAL_TURBINES
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
MAY_PATH   = _ROOT / "data" / "raw" / "18.05_test_dataset.csv"
ERA5_PATH  = _ROOT / "data" / "external" / "era5_reanalysis.parquet"

LGBM_PARAMS = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=200, log_period=0,
)
SEEDS = [42, 123, 456]
K = 80
TURBINE_RATED_MW = 3.465
BACKTEST_START   = pd.Timestamp("2026-05-15 00:00:00")
BACKTEST_END     = pd.Timestamp("2026-05-17 23:00:00")


# Reuse helpers from predict_18_05
def merge_era5(df, era5):
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")
    df["ws10_bias"]     = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"]   = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"]     = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]
    df["era5_ws100_cube"]     = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    new_cols = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[new_cols] = df[new_cols].fillna(0)
    return df


def add_era5_rolling(df):
    df = df.copy()
    ws   = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]
    for w in [3, 6, 12, 24]:
        r = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = r.mean()
        df[f"era5_ws100_roll_std_{w}h"]  = r.std().fillna(0)
    df["era5_ws100_diff1"] = ws.diff(1).fillna(0)
    df["era5_ws100_diff3"] = ws.diff(3).fillna(0)
    df["era5_ws100_diff6"] = ws.diff(6).fillna(0)
    df["era5_pressure_diff3"]  = pres.diff(3).fillna(0)
    df["era5_pressure_diff6"]  = pres.diff(6).fillna(0)
    df["era5_pressure_diff12"] = pres.diff(12).fillna(0)
    r6 = ws.rolling(6, min_periods=1)
    df["era5_turb_intensity_6h"] = r6.std().fillna(0) / (r6.mean() + 1e-3)
    df["era5_dir_sin_diff1"] = df["era5_dir100_sin"].diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = df["era5_dir100_cos"].diff(1).fillna(0)
    return df


def add_pc(df, pc_sector, pc_global):
    df = df.copy()
    v_eff = df["v_eff"].to_numpy()
    dir_deg = (df["wind_direction_120m"] * 1000.0).to_numpy()
    df["p_curve_sector"]          = pc_sector.predict(v_eff, dir_deg)
    df["p_curve_global"]          = pc_global.predict(v_eff)
    df["p_curve_rews"]            = pc_global.predict(df["rews"].to_numpy())
    df["p_curve_x_active"]        = df["p_curve_sector"] * df["active_turbines_ratio"]
    df["p_curve_global_x_active"] = df["p_curve_global"] * df["active_turbines_ratio"]
    df["p_curve_ratio"]           = df["p_curve_sector"] / CAPACITY_MW
    df["p_curve_sector_minus_global"] = df["p_curve_sector"] - df["p_curve_global"]
    return df


def to_cf(y_mw, active):
    return y_mw / np.maximum(active * TURBINE_RATED_MW, 1e-3)


def from_cf(cf, active):
    return cf * active * TURBINE_RATED_MW


def train_lgbm(X_tr, y_tr, X_va, y_va, X_te, feat_cols, ws_tr, seeds, config):
    regime_test = {}
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        wt   = np.where(mask, 2.0, 0.3).astype(np.float32)
        preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**config.__dict__, "seed": s})
            dt  = lgb.Dataset(X_tr, label=y_tr, weight=wt, feature_name=feat_cols, free_raw_data=False)
            dv  = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            bst = lgb.train(cfg.to_params(), dt, num_boost_round=config.num_boost_round,
                            valid_sets=[dv], valid_names=["val"],
                            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)])
            preds.append(bst.predict(X_te, num_iteration=bst.best_iteration))
        regime_test[name] = np.mean(preds, axis=0)
    return np.mean(list(regime_test.values()), axis=0)


def main():
    set_global_seed(42)
    print("=" * 70)
    print("BACKTEST: Predict May 15-17, 2026 and compare with actuals")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df_train_raw = pd.read_csv(TRAIN_PATH)
    df_train_raw[TIMESTAMP_COL] = pd.to_datetime(df_train_raw[TIMESTAMP_COL])
    df_train_raw = df_train_raw.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_train_raw["ws_180m_is_imputed"] = False

    df_may_raw = pd.read_csv(MAY_PATH)
    df_may_raw[TIMESTAMP_COL] = pd.to_datetime(df_may_raw[TIMESTAMP_COL]).dt.round("h")
    df_may_raw = df_may_raw.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_may_raw["ws_180m_is_imputed"] = False

    era5 = pd.read_parquet(ERA5_PATH)

    # Split: train = everything before May 15, backtest = May 15-17
    may_with_actuals = df_may_raw[df_may_raw[TARGET_COL].notna()].copy()
    train_extra   = may_with_actuals[may_with_actuals[TIMESTAMP_COL] < BACKTEST_START].copy()
    backtest_rows = may_with_actuals[
        (may_with_actuals[TIMESTAMP_COL] >= BACKTEST_START) &
        (may_with_actuals[TIMESTAMP_COL] <= BACKTEST_END)
    ].copy()
    # May 18 has no actuals, include for rolling continuity only (no target)
    may18_rows = df_may_raw[df_may_raw[TARGET_COL].isna()].copy()

    print(f"\nData split:")
    print(f"  Base train (2022-2025)   : {len(df_train_raw):>6,} rows")
    print(f"  Extra train (Apr-May 14) : {len(train_extra):>6,} rows")
    print(f"  BACKTEST   (May 15-17)   : {len(backtest_rows):>6,} rows  <- ground truth")
    print(f"  May 18 (rolling ctx only): {len(may18_rows):>6,} rows")

    # ------------------------------------------------------------------
    # Feature engineering on full timeline for rolling continuity
    # ------------------------------------------------------------------
    print("\nBuilding features...")
    df_train_raw["_split"]  = "train"
    train_extra["_split"]   = "train"
    backtest_rows["_split"] = "test"
    may18_rows["_split"]    = "may18"   # context-only, not used in train or test

    combined = pd.concat([df_train_raw, train_extra, backtest_rows, may18_rows],
                         ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)

    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)

    df_all_train  = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_backtest   = combined[combined["_split"] == "test"].reset_index(drop=True)

    impossible = identify_impossible_rows(df_all_train)
    df_all_train["_is_impossible"] = impossible.values

    # ------------------------------------------------------------------
    # ERA5 vs NWP diagnostic BEFORE training
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: ERA5 vs NWP wind speeds (May 15-17 backtest period)")
    print("=" * 70)
    print(f"\n{'Datetime':22s} {'NWP ws10':>9} {'NWP ws120':>10} {'ERA5 ws10':>10} {'ERA5 ws100':>11} {'ERA5/NWP ratio':>15}")
    print("-" * 80)
    for _, row in df_backtest.iterrows():
        ts   = row[TIMESTAMP_COL]
        n10  = row["wind_speed_10m"]
        n120 = row["wind_speed_120m"]
        e10  = row["era5_wind_speed_10m"]
        e100 = row["era5_wind_speed_100m"]
        ratio = e100 / (n120 + 0.1)
        print(f"{str(ts):22s} {n10:>9.2f} {n120:>10.2f} {e10:>10.2f} {e100:>11.2f} {ratio:>15.2f}")

    print("\nSummary statistics (May 15–17):")
    print(f"  NWP  ws_120m mean : {df_backtest['wind_speed_120m'].mean():.2f} m/s")
    print(f"  ERA5 ws_100m mean : {df_backtest['era5_wind_speed_100m'].mean():.2f} m/s")
    print(f"  ERA5/NWP ratio    : {(df_backtest['era5_wind_speed_100m'] / (df_backtest['wind_speed_120m'] + 0.1)).mean():.2f}×")
    print(f"  Actual power mean : {df_backtest[TARGET_COL].mean():.2f} MW")

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    print("\nFeature selection (probe val = Apr 2026)...")
    probe_val_start = pd.Timestamp("2026-04-01")
    p_tr_mask = df_all_train[TIMESTAMP_COL] < probe_val_start
    probe_tr = df_all_train[p_tr_mask].copy()
    probe_va = df_all_train[~p_tr_mask].copy()

    fit_data = probe_tr[~probe_tr["_is_impossible"]]
    pc_s = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_g = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wk   = fit_wake_lookup(fit_data, n_sectors=16)

    df_pt = add_pc(probe_tr, pc_s, pc_g); df_pt = add_wake_features(df_pt, wk)
    df_pv = add_pc(probe_va, pc_s, pc_g); df_pv = add_wake_features(df_pv, wk)

    feat_cols_all = [c for c in feature_columns(df_pt)
                     if c not in ("_is_impossible", "_split", TARGET_COL)]
    a_pt = df_pt["active_turbines"].to_numpy(dtype=np.float32)
    a_pv = df_pv["active_turbines"].to_numpy(dtype=np.float32)
    y_pt = to_cf(df_pt[TARGET_COL].to_numpy(dtype=np.float32), a_pt)
    y_pv = to_cf(df_pv[TARGET_COL].to_numpy(dtype=np.float32), a_pv)

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    probe_bst = lgb.train(
        cfg0.to_params(),
        lgb.Dataset(df_pt[feat_cols_all].to_numpy(dtype=np.float32), label=y_pt,
                    feature_name=feat_cols_all, free_raw_data=False),
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(df_pv[feat_cols_all].to_numpy(dtype=np.float32),
                                label=y_pv, feature_name=feat_cols_all, free_raw_data=False)],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe_bst.feature_importance(importance_type="gain")
    top_k = [n for n, _ in sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])[:K]]

    # ------------------------------------------------------------------
    # Train on all pre-May-15 data, predict May 15–17
    # ------------------------------------------------------------------
    print("Training LightGBM (CF + MW) on 2022-May 14...")
    fit_all = df_all_train[~df_all_train["_is_impossible"]]
    pc_s_f  = fit_sector_isotonic(fit_all, n_sectors=8)
    pc_g_f  = IsotonicPowerCurve().fit(fit_all["v_eff"], fit_all[TARGET_COL])
    wk_f    = fit_wake_lookup(fit_all, n_sectors=16)

    df_tr_f = add_pc(df_all_train, pc_s_f, pc_g_f); df_tr_f = add_wake_features(df_tr_f, wk_f)
    df_te_f = add_pc(df_backtest,  pc_s_f, pc_g_f); df_te_f = add_wake_features(df_te_f, wk_f)
    for c in set(top_k) - set(df_te_f.columns):
        df_te_f[c] = 0.0

    active_tr = df_tr_f["active_turbines"].to_numpy(dtype=np.float32)
    active_te = df_te_f["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_mw   = df_tr_f[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf   = to_cf(y_tr_mw, active_tr)
    ws_tr     = df_tr_f["wind_speed_120m"].to_numpy()
    y_te_mw   = df_te_f[TARGET_COL].to_numpy(dtype=np.float32)   # actuals

    X_tr = df_tr_f[top_k].to_numpy(dtype=np.float32)
    X_te = df_te_f[top_k].to_numpy(dtype=np.float32)

    # Val for early stopping: Apr 2026
    val_mask  = df_tr_f[TIMESTAMP_COL] >= probe_val_start
    X_va_es   = X_tr[val_mask]
    y_va_cf   = y_tr_cf[val_mask]
    X_tr_es   = X_tr[~val_mask]
    y_tr_cf_s = y_tr_cf[~val_mask]
    y_tr_mw_s = y_tr_mw[~val_mask]
    ws_tr_s   = ws_tr[~val_mask]

    test_cf = train_lgbm(X_tr_es, y_tr_cf_s, X_va_es, y_va_cf,
                         X_te, top_k, ws_tr_s, SEEDS, LGBM_PARAMS)
    test_mw = train_lgbm(X_tr_es, y_tr_mw_s, X_va_es, df_tr_f[TARGET_COL].to_numpy(dtype=np.float32)[val_mask],
                         X_te, top_k, ws_tr_s, SEEDS, LGBM_PARAMS)

    pred_cf_mw = np.clip(from_cf(test_cf, active_te), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(test_mw, 0, CAPACITY_MW)
    pred_blend = np.clip(0.5 * pred_cf_mw + 0.5 * pred_mw_mw, 0, CAPACITY_MW)

    nmae_cf    = normalized_mae(y_te_mw, pred_cf_mw)
    nmae_mw    = normalized_mae(y_te_mw, pred_mw_mw)
    nmae_blend = normalized_mae(y_te_mw, pred_blend)

    # ------------------------------------------------------------------
    # Results: per-hour prediction vs actual + ERA5 accuracy analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS: May 15-17, 2026")
    print("=" * 70)
    print(f"\n{'Datetime':22s} {'Actual':>8} {'CF pred':>8} {'MW pred':>8} {'Blend':>8} {'Error':>7} {'ERA5ws100':>10} {'NWPws120':>9}")
    print("-" * 90)

    era5_errors, nwp_errors = [], []
    for i, (_, row) in enumerate(df_te_f.iterrows()):
        ts    = row[TIMESTAMP_COL]
        act   = float(y_te_mw[i])
        p_bl  = float(pred_blend[i])
        err   = p_bl - act
        e100  = float(row["era5_wind_speed_100m"])
        n120  = float(row["wind_speed_120m"])
        print(f"{str(ts):22s} {act:>8.2f} {pred_cf_mw[i]:>8.2f} {pred_mw_mw[i]:>8.2f} "
              f"{p_bl:>8.2f} {err:>+7.2f} {e100:>10.2f} {n120:>9.2f}")

    print("-" * 90)
    print(f"\n{'nMAE':30s} {'CF':>8} {'MW':>8} {'Blend':>8}")
    print(f"{'Backtest (May 15-17)':30s} {nmae_cf:>8.3f} {nmae_mw:>8.3f} {nmae_blend:>8.3f} %")

    # ERA5 vs NWP accuracy: which wind speed better predicts actual power?
    print("\n" + "=" * 70)
    print("ERA5 vs NWP RELIABILITY ANALYSIS (May 15-17)")
    print("=" * 70)

    # Simple power curve comparison: v^3 proportional to power
    e100 = df_te_f["era5_wind_speed_100m"].to_numpy()
    n120 = df_te_f["wind_speed_120m"].to_numpy()
    # Normalize both to fraction of rated capacity for comparison
    p_from_era5_cube = np.clip(e100**3 / (12.0**3), 0, 1) * CAPACITY_MW * 0.9
    p_from_nwp_cube  = np.clip(n120**3 / (12.0**3), 0, 1) * CAPACITY_MW * 0.9
    nmae_era5_raw = normalized_mae(y_te_mw, p_from_era5_cube)
    nmae_nwp_raw  = normalized_mae(y_te_mw, p_from_nwp_cube)

    print(f"\n  Simple v^3 power curve estimate (sanity check):")
    print(f"    ERA5 ws_100m → power : nMAE = {nmae_era5_raw:.3f}%")
    print(f"    NWP  ws_120m → power : nMAE = {nmae_nwp_raw:.3f}%")

    # Correlation with actual power
    corr_era5 = np.corrcoef(e100, y_te_mw)[0, 1]
    corr_nwp  = np.corrcoef(n120, y_te_mw)[0, 1]
    print(f"\n  Pearson correlation with actual power:")
    print(f"    ERA5 ws_100m : r = {corr_era5:.3f}")
    print(f"    NWP  ws_120m : r = {corr_nwp:.3f}")

    print(f"\n  Mean wind speeds (May 15–17):")
    print(f"    ERA5 ws_100m : {e100.mean():.2f} m/s  (min={e100.min():.2f}  max={e100.max():.2f})")
    print(f"    NWP  ws_120m : {n120.mean():.2f} m/s  (min={n120.min():.2f}  max={n120.max():.2f})")
    print(f"    Actual power : {y_te_mw.mean():.2f} MW  (min={y_te_mw.min():.2f}  max={y_te_mw.max():.2f})")

    print("\n" + "=" * 70)
    print("IMPLICATION FOR MAY 18 PREDICTIONS")
    print("=" * 70)
    print(f"\n  Blend nMAE on backtest (May 15-17): {nmae_blend:.3f}%")
    better = "ERA5" if corr_era5 > corr_nwp else "NWP"
    print(f"  Better predictor (by correlation)  : {better}")
    print(f"\n  May 18 key discrepancy:")
    print(f"    NWP ws_120m mean  : 3.70 m/s  (hour 23: 2.64 m/s)")
    print(f"    ERA5 ws_100m mean : {e100.mean():.2f} m/s at similar recent period")
    print(f"    -> ERA5 for May 18 shows ws_100m up to 8.19 m/s (hour 23)")
    print(f"    -> If ERA5 reliability ratio from backtest = {corr_era5:.2f}, this is {'credible' if corr_era5 > 0.7 else 'SUSPECT'}")
    print(f"\n  Note: Open-Meteo archive ERA5 has ~5 day lag.")
    print(f"  May 18 data is FORECAST-based, not true ERA5 reanalysis.")
    print(f"  Recommend fetching NASA MERRA-2 for May 15+ as cross-check:")
    print(f"  ! python scripts/fetch_nasa_power.py --start 2026-04-01 --end 2026-05-18")


if __name__ == "__main__":
    main()
