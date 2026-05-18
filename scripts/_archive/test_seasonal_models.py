"""Test seasonal separate LGBM models (per §6.1 of literature review).

Paper [26] shows separate winter/summer models substantially reduce RMSE.
Our test period is Q1 2026 (January-March = winter), so a winter-specific
model might generalize better than one trained on all seasons.

Strategies to test:
A) Train on ALL data + add "month_group" feature (current approach).
B) Train only on Q1 data (Jan-Mar from 2022-2025) → specialist model.
C) Train two models: winter (Dec-Feb) and rest; use winter for Q1 inference.
D) Soft-weighted training: Q1 rows get weight 3x, else 1x.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
SEEDS = [42, 123, 456, 789, 2026]


def _add_pc(df, pc_sector, pc_global):
    df = df.copy()
    v_eff = df["v_eff"].to_numpy()
    dir_deg = (df["wind_direction_120m"] * 1000.0).to_numpy()
    df["p_curve_sector"] = pc_sector.predict(v_eff, dir_deg)
    df["p_curve_global"] = pc_global.predict(v_eff)
    df["p_curve_rews"] = pc_global.predict(df["rews"].to_numpy())
    df["p_curve_x_active"] = df["p_curve_sector"] * df["active_turbines_ratio"]
    df["p_curve_global_x_active"] = df["p_curve_global"] * df["active_turbines_ratio"]
    df["p_curve_ratio"] = df["p_curve_sector"] / CAPACITY_MW
    df["p_curve_sector_minus_global"] = df["p_curve_sector"] - df["p_curve_global"]
    return df


def merge_era5(df, era5):
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")
    df["ws10_bias"] = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"] = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"] = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    era5_new = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def add_era5_rolling(df):
    df = df.copy()
    ws = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]
    for w in [3, 6, 12, 24]:
        roll = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = roll.mean()
        df[f"era5_ws100_roll_std_{w}h"] = roll.std().fillna(0)
    df["era5_ws100_diff1"] = ws.diff(1).fillna(0)
    df["era5_ws100_diff3"] = ws.diff(3).fillna(0)
    df["era5_ws100_diff6"] = ws.diff(6).fillna(0)
    df["era5_pressure_diff3"] = pres.diff(3).fillna(0)
    df["era5_pressure_diff6"] = pres.diff(6).fillna(0)
    df["era5_pressure_diff12"] = pres.diff(12).fillna(0)
    roll6 = ws.rolling(6, min_periods=1)
    df["era5_turb_intensity_6h"] = roll6.std().fillna(0) / (roll6.mean() + 1e-3)
    dir_sin = df["era5_dir100_sin"]
    dir_cos = df["era5_dir100_cos"]
    df["era5_dir_sin_diff1"] = dir_sin.diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = dir_cos.diff(1).fillna(0)
    return df


def train_seeds(X_tr, y_tr, X_va, y_va, feat_cols, w_tr, config):
    preds = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        if w_tr is not None:
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols, free_raw_data=False)
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        p = np.clip(b.predict(X_va, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        preds.append(p)
    return np.mean(preds, axis=0)


def main():
    set_global_seed(42)
    df_train = load_train(_ROOT / "data" / "raw" / "train_dataset.csv")
    df_valid = load_valid_features(_ROOT / "data" / "raw" / "valid_features.csv")
    era5 = pd.read_parquet(ERA5_PATH)

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    folds = default_folds()
    fold5 = folds[-1]  # Q1 2025
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(df_train.iloc[val_idx], pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]

    # Use v14 tuned config.
    config = LGBMConfig(
        num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
        feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
        lambda_l1=0.253, lambda_l2=0.00971,
        num_boost_round=5000, early_stopping_rounds=250, log_period=0,
    )

    # Probe for top-70.
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    cfg0 = LGBMConfig(**{**config.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_70 = [n for n, _ in feat_imp[:70]]

    X_tr = df_tr[top_70].to_numpy(dtype=np.float32)
    X_va = df_va[top_70].to_numpy(dtype=np.float32)

    # Month array for fold train.
    train_months = df_tr[TIMESTAMP_COL].dt.month.to_numpy()
    val_months = df_va[TIMESTAMP_COL].dt.month.to_numpy()

    # --- Strategy A: Baseline (all data, no weighting). ---
    print("=== Strategy A: Baseline v14 (all data, no seasonal weighting) ===")
    preds_a = train_seeds(X_tr, y_tr, X_va, y_va, top_70, None, config)
    nmae_a = normalized_mae(y_va, preds_a)
    print(f"  Fold-5 nMAE: {nmae_a:.4f}%")

    # --- Strategy B: Train only on Q1 (Jan-Mar) data. ---
    print("\n=== Strategy B: Train only on Q1 data ===")
    mask_q1 = np.isin(train_months, [1, 2, 3])
    X_tr_q1 = X_tr[mask_q1]
    y_tr_q1 = y_tr[mask_q1]
    print(f"  Q1-only training rows: {len(X_tr_q1)}")
    preds_b = train_seeds(X_tr_q1, y_tr_q1, X_va, y_va, top_70, None, config)
    nmae_b = normalized_mae(y_va, preds_b)
    print(f"  Fold-5 nMAE: {nmae_b:.4f}%")

    # --- Strategy C: Winter (Dec-Feb) vs rest. Winter for Q1 inference. ---
    print("\n=== Strategy C: Winter (Dec-Feb) only training ===")
    mask_winter = np.isin(train_months, [12, 1, 2])
    X_tr_w = X_tr[mask_winter]
    y_tr_w = y_tr[mask_winter]
    print(f"  Winter-only training rows: {len(X_tr_w)}")
    preds_c = train_seeds(X_tr_w, y_tr_w, X_va, y_va, top_70, None, config)
    nmae_c = normalized_mae(y_va, preds_c)
    print(f"  Fold-5 nMAE: {nmae_c:.4f}%")

    # --- Strategy D: Weighted (Q1 rows = 3x, others = 1x). ---
    print("\n=== Strategy D: Weighted training (Q1 rows 3x) ===")
    weights_d = np.where(mask_q1, 3.0, 1.0).astype(np.float32)
    preds_d = train_seeds(X_tr, y_tr, X_va, y_va, top_70, weights_d, config)
    nmae_d = normalized_mae(y_va, preds_d)
    print(f"  Fold-5 nMAE: {nmae_d:.4f}%")

    # --- Strategy E: Weighted (winter rows = 2x). ---
    print("\n=== Strategy E: Weighted (winter Dec-Feb = 2x) ===")
    weights_e = np.where(mask_winter, 2.0, 1.0).astype(np.float32)
    preds_e = train_seeds(X_tr, y_tr, X_va, y_va, top_70, weights_e, config)
    nmae_e = normalized_mae(y_va, preds_e)
    print(f"  Fold-5 nMAE: {nmae_e:.4f}%")

    # --- Strategy F: Blend of Strategy A and B. ---
    print("\n=== Strategy F: Blend A + B (50/50) ===")
    preds_f = (preds_a + preds_b) / 2.0
    nmae_f = normalized_mae(y_va, preds_f)
    print(f"  Fold-5 nMAE: {nmae_f:.4f}%")

    # Summary.
    print("\n=== SUMMARY ===")
    print(f"  A. All data:               {nmae_a:.4f}%")
    print(f"  B. Q1-only:                {nmae_b:.4f}%  (delta {nmae_b - nmae_a:+.4f})")
    print(f"  C. Winter-only:            {nmae_c:.4f}%  (delta {nmae_c - nmae_a:+.4f})")
    print(f"  D. Q1 weighted 3x:         {nmae_d:.4f}%  (delta {nmae_d - nmae_a:+.4f})")
    print(f"  E. Winter weighted 2x:     {nmae_e:.4f}%  (delta {nmae_e - nmae_a:+.4f})")
    print(f"  F. Blend A + B (50/50):    {nmae_f:.4f}%  (delta {nmae_f - nmae_a:+.4f})")


if __name__ == "__main__":
    main()
