"""Night specialist: retrain LGBM on hours 0-8 ONLY with May 17 lag features.

Architecture:
  - Filter training data to ONLY hours 0-8 (midnight-morning specialist)
  - Add lag features from the previous evening (power at t-1, t-2, ... t-6)
  - Spring seasonal filter (Apr-May) for maximum relevance
  - 3 regime specialists × 3 seeds, CF + MW, 50/50 blend
  - For hours 9-23: use the existing hybrid prediction

Output combines:
  Hours 0-8:  night specialist prediction
  Hours 9-23: existing prediction from 18_05_2026_forecast.csv

Final output: submissions/18_05_2026_night_specialist.csv
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
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TURBINES_IN_MAINTENANCE_COL, TOTAL_TURBINES
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

# Paths
TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
MAY_PATH   = _ROOT / "data" / "raw" / "18.05_test_dataset.csv"
EXISTING_FORECAST = _ROOT / "submissions" / "18_05_2026_forecast.csv"
OUTPUT_PATH = _ROOT / "submissions" / "18_05_2026_night_specialist.csv"

# Config
LGBM_PARAMS = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=200, log_period=0,
)
SEEDS = [42, 123, 456]
K = 60  # fewer features since we have fewer training rows
TURBINE_RATED_MW = 3.465
SPRING_MONTHS = {3, 4, 5}
NIGHT_HOURS = set(range(0, 9))  # 0-8 inclusive


def load_train_raw(path):
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)


def load_may_file(path):
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    has_target = df[TARGET_COL].notna()
    hist = df[has_target].copy()
    pred = df[~has_target].copy()
    return hist, pred


def to_cf(y_mw, active):
    return y_mw / np.maximum(active * TURBINE_RATED_MW, 1e-3)


def from_cf(cf, active):
    return np.clip(cf, 0.0, 1.0) * active * TURBINE_RATED_MW


def add_power_lags(df):
    """Add lagged power from previous hours (the key advantage of the night specialist).
    
    For training rows: use actual power shifted by 1-6 hours.
    For test rows (May 18 0-8): use May 17 actuals as the lag source.
    """
    out = df.copy()
    out = out.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    
    if TARGET_COL in out.columns:
        power = out[TARGET_COL].copy()
    else:
        power = pd.Series(np.nan, index=out.index)
    
    for lag in [1, 2, 3, 6]:
        out[f"power_lag{lag}h"] = power.shift(lag)
    
    # Rolling mean of last 3h and 6h of power
    out["power_rmean3h"] = power.rolling(3, min_periods=1).mean().shift(1)
    out["power_rmean6h"] = power.rolling(6, min_periods=1).mean().shift(1)
    
    return out


def add_power_curve_features(df, pc_sector, pc_global):
    df = df.copy()
    v_eff = df["v_eff"].to_numpy()
    dir_deg = (df["wind_direction_120m"] * 1000.0).to_numpy()
    df["p_curve_sector"] = pc_sector.predict(v_eff, dir_deg)
    df["p_curve_global"] = pc_global.predict(v_eff)
    df["p_curve_x_active"] = df["p_curve_sector"] * df["active_turbines_ratio"]
    df["p_curve_ratio"] = df["p_curve_sector"] / CAPACITY_MW
    return df


def train_specialist(X_tr, y_tr, X_va, y_va, X_te, feat_cols, ws_tr, sw, seeds):
    """3 specialists × N seeds. Returns (val_pred, test_pred)."""
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_5", (0, 5)), ("mid_3_8", (3, 8)), ("high_6_25", (6, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = (np.where(mask, 2.0, 0.3) * sw).astype(np.float32)
        vp, tp = [], []
        for s in seeds:
            cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights,
                             feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            vp.append(b.predict(X_va, num_iteration=b.best_iteration))
            tp.append(b.predict(X_te, num_iteration=b.best_iteration))
        regime_val[name] = np.mean(vp, axis=0)
        regime_test[name] = np.mean(tp, axis=0)
    return np.mean(list(regime_val.values()), axis=0), np.mean(list(regime_test.values()), axis=0)


def main():
    set_global_seed(42)
    print("=" * 65)
    print("Night Specialist: LGBM trained on hours 0-8 ONLY + power lags")
    print("=" * 65)

    # --- Load data ---
    print("\n[1/5] Loading...")
    df_train = load_train_raw(TRAIN_PATH)
    df_may_hist, df_may_pred = load_may_file(MAY_PATH)

    # Combine train + may history
    df_train["_split"] = "train"
    df_may_hist["_split"] = "train"
    df_may_pred["_split"] = "test"
    df_may_pred["_orig_row"] = range(len(df_may_pred))

    combined = pd.concat([df_train, df_may_hist, df_may_pred], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # --- Build features ---
    print("[2/5] Building features + power lags...")
    combined = build_features(combined, sort_by_time=False)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_power_lags(combined)

    # Split back
    df_all_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_test = combined[combined["_split"] == "test"].reset_index(drop=True)

    # For test rows (May 18 0-8), fill power lags from May 17 actuals
    # The combined frame already has May 17 rows before May 18 → shift should work
    # Check if lags are populated
    test_0_8 = df_test[df_test[TIMESTAMP_COL].dt.hour.between(0, 8)].copy()
    print(f"  Test rows (hours 0-8): {len(test_0_8)}")
    print(f"  power_lag1h on test: NaN={test_0_8['power_lag1h'].isna().sum()}/{len(test_0_8)}")
    
    # If lags are NaN on test (because May 18 target is NaN), manually fill
    # from May 17 actuals
    if test_0_8["power_lag1h"].isna().sum() > 0:
        print("  Filling test power lags from May 17 actuals...")
        may17_power = df_may_hist[df_may_hist[TIMESTAMP_COL].dt.day == 17].sort_values(TIMESTAMP_COL)
        # Build a simple hour → power lookup (last value if duplicates)
        may17_by_hour = {}
        for _, r in may17_power.iterrows():
            may17_by_hour[r[TIMESTAMP_COL].hour] = float(r[TARGET_COL])
        
        for idx in test_0_8.index:
            h = df_test.loc[idx, TIMESTAMP_COL].hour
            for lag in [1, 2, 3, 6]:
                source_hour = (h - lag) % 24
                if source_hour in may17_by_hour:
                    df_test.at[idx, f"power_lag{lag}h"] = may17_by_hour[source_hour]
            # Rolling means
            recent3 = [may17_by_hour.get((h - i) % 24, 0.0) for i in range(1, 4)]
            df_test.at[idx, "power_rmean3h"] = np.mean(recent3)
            recent6 = [may17_by_hour.get((h - i) % 24, 0.0) for i in range(1, 7)]
            df_test.at[idx, "power_rmean6h"] = np.mean(recent6)

    # Fill remaining NaN lags with 0
    lag_cols = [c for c in df_test.columns if "power_lag" in c or "power_rmean" in c]
    df_test[lag_cols] = df_test[lag_cols].fillna(0.0)
    df_all_train[lag_cols] = df_all_train[lag_cols].fillna(0.0)

    # --- Filter to hours 0-8 + spring for training ---
    print("[3/5] Filtering to night hours (0-8) + spring (Apr-May)...")
    impossible = identify_impossible_rows(df_all_train)
    df_all_train["_is_impossible"] = impossible.values

    night_mask = df_all_train[TIMESTAMP_COL].dt.hour.isin(NIGHT_HOURS)
    spring_mask = df_all_train[TIMESTAMP_COL].dt.month.isin(SPRING_MONTHS)
    clean_mask = ~df_all_train["_is_impossible"]
    
    train_mask = night_mask & spring_mask & clean_mask
    print(f"  Night+Spring training rows: {train_mask.sum()}")

    # ES val: use May 2025 hours 0-8 as validation
    ts = df_all_train[TIMESTAMP_COL]
    val_mask = (night_mask & (ts >= "2025-04-01") & (ts < "2025-06-01") & clean_mask)
    tr_mask = train_mask & ~val_mask
    print(f"  Train: {tr_mask.sum()}  Val: {val_mask.sum()}")

    # --- Power curves + feature selection ---
    fit_data = df_all_train[tr_mask & clean_mask]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = add_power_curve_features(df_all_train[tr_mask].copy(), pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = add_power_curve_features(df_all_train[val_mask].copy(), pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)
    df_te = add_power_curve_features(df_test.copy(), pc_sector, pc_global)
    df_te = add_wake_features(df_te, wake)

    feat_cols_all = [c for c in feature_columns(df_tr)
                     if c not in ("_is_impossible", "_split", "_orig_row", TARGET_COL)]
    
    # Include power lag features explicitly
    for c in lag_cols:
        if c not in feat_cols_all and c in df_tr.columns:
            feat_cols_all.append(c)

    # Probe for top-K
    print("[4/5] Feature selection...")
    a_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), a_tr)
    a_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(df_va[TARGET_COL].to_numpy(dtype=np.float32), a_va)

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    dt_ = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt_, num_boost_round=3000,
                      valid_sets=[dv_], valid_names=["val"],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist()), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    
    # Report lag feature importance
    lag_in_top = [n for n in top_k if "power_lag" in n or "power_rmean" in n]
    print(f"  Top-{K} features selected")
    print(f"  Power-lag features in top-K: {lag_in_top}")
    if lag_in_top:
        for n in lag_in_top:
            print(f"    {n}: gain={dict(feat_imp)[n]:,.0f}")

    # --- Train night specialist ---
    print("[5/5] Training night specialist (CF + MW)...")
    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    # Test: only hours 0-8 of May 18
    te_mask_0_8 = df_te[TIMESTAMP_COL].dt.hour.between(0, 8)
    X_te = df_te.loc[te_mask_0_8, top_k].to_numpy(dtype=np.float32)
    for c in set(top_k) - set(df_te.columns):
        X_te_fix = np.zeros((te_mask_0_8.sum(), 1))  # placeholder

    ws_tr = df_tr["wind_speed_120m"].to_numpy()
    sw = np.ones(len(df_tr), dtype=np.float32)
    # Up-weight 2026 data
    is_2026 = df_tr[TIMESTAMP_COL].dt.year >= 2026
    sw[is_2026.to_numpy()] = 5.0

    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    # CF target
    print("  [CF target]")
    _, test_cf = train_specialist(X_tr, y_tr_cf, X_va, y_va_cf, X_te, top_k, ws_tr, sw, SEEDS)
    
    # MW target
    print("  [MW target]")
    _, test_mw = train_specialist(X_tr, y_tr_mw, X_va, y_va_mw, X_te, top_k, ws_tr, sw, SEEDS)

    # Convert and blend
    active_te = df_te.loc[te_mask_0_8, "active_turbines"].to_numpy(dtype=np.float32)
    pred_cf_mw = np.clip(from_cf(test_cf, active_te), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(test_mw, 0, CAPACITY_MW)
    night_pred = np.clip(0.5 * pred_cf_mw + 0.5 * pred_mw_mw, 0, CAPACITY_MW)

    # Val nMAE
    val_cf, _ = train_specialist(X_tr, y_tr_cf, X_va, y_va_cf, X_va, top_k, ws_tr, sw, SEEDS)
    val_cf_mw = np.clip(from_cf(val_cf, a_va), 0, CAPACITY_MW)
    print(f"  Val nMAE (CF on Apr-May 2025 nights): {normalized_mae(y_va_mw, val_cf_mw):.4f}%")

    # --- Combine with existing full-day forecast ---
    print("\n  Combining with existing forecast for hours 9-23...")
    existing = pd.read_csv(EXISTING_FORECAST)
    existing["datetime"] = pd.to_datetime(existing["datetime"])
    existing = existing.sort_values("hour").reset_index(drop=True)

    # Build final 24-hour output
    final_df = existing[["datetime", "hour", "forecast_mw"]].copy()
    
    # Replace hours 0-8 with night specialist
    test_hours = df_te.loc[te_mask_0_8, TIMESTAMP_COL].dt.hour.to_numpy()
    for i, h in enumerate(test_hours):
        final_df.loc[final_df["hour"] == h, "forecast_mw"] = night_pred[i]

    # Apply persistence correction for hours 0-2 (farm was OFF at 23:00)
    # Blend specialist with persistence: 50% specialist + 50% persistence (0.076 MW)
    MAY17_23_POWER = 0.076
    for h in [0, 1, 2]:
        old = final_df.loc[final_df["hour"] == h, "forecast_mw"].values[0]
        blended = 0.5 * old + 0.5 * MAY17_23_POWER
        final_df.loc[final_df["hour"] == h, "forecast_mw"] = blended

    final_df["forecast_mw"] = final_df["forecast_mw"].clip(0, CAPACITY_MW)

    # Print results
    print(f"\n{'Hour':>5}  {'Existing':>9}  {'Specialist':>11}  {'Final':>7}")
    print(f"  {'-'*42}")
    for _, r in final_df[final_df["hour"] <= 8].iterrows():
        h = int(r["hour"])
        ex = existing.loc[existing["hour"] == h, "forecast_mw"].values[0]
        sp = night_pred[test_hours == h][0] if h in test_hours else ex
        print(f"  {h:>2}h   {ex:>8.3f}  {sp:>10.3f}  {r['forecast_mw']:>6.3f}")

    print(f"\n  Hours 0-8 mean: {final_df[final_df['hour'] <= 8]['forecast_mw'].mean():.3f} MW")
    print(f"  Full 24h mean:  {final_df['forecast_mw'].mean():.3f} MW")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
