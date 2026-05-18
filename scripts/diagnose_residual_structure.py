"""Diagnose whether the post-PC residual has learnable structure.

Tests three reformulations of the prediction target:

  R1.  target_mw (current approach — what v32 etc. predict).
  R2.  CF = target / (n_active × 3.465) (current "CF leg" target).
  R3.  residual_mfr = target - mfr_pc_pred (residual after manufacturer
       power curve).
  R4.  residual_isotonic = target - p_curve_global (residual after the
       fold-fitted isotonic on v_eff).

For each target, on Fold-5 OOF, report:

  - target std (variance scale)
  - what fraction of error a *zero-prediction* baseline would give
    (sanity: if std is small, even predicting 0 is decent)
  - what fraction of variance is correlated with hour, month, ws_120
    (linear regressors — do simple proxies still capture it?)
  - residual after fitting a one-feature LightGBM on each candidate
    feature in isolation (gain from a single split)

If R3/R4 residuals show high non-trivially-explainable variance, the
residual modelling switch is justified. If their residuals are noise,
LightGBM-on-target is already extracting everything.
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
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import (
    add_datasheet_power_features,
    per_turbine_power_kw,
)
from src.features.era5_v2 import merge_era5_v2
from src.features.extras import add_extra_features
from src.features.pipeline import build_features
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup

from src.training.train_v32_era5v2 import (
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling,
)

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"


def main():
    print("[1/3] Loading + building features...")
    df = _load_train_raw(TRAIN_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    df["_split"] = "train"
    df = build_features(df, sort_by_time=False)
    df = _merge_era5(df, era5)
    df = add_datasheet_power_features(df)
    df = add_extra_features(df)
    df = _add_era5_rolling(df)
    df = merge_era5_v2(df)

    impossible = identify_impossible_rows(df)
    df["_is_impossible"] = impossible.values

    folds = default_folds()
    train_idx, val_idx = split_indices(df, folds[-1])
    fold_train = df.iloc[train_idx]
    fold_val = df.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)
    df = _add_pc(df, pc_sector, pc_global)
    df = add_wake_features(df, wake)
    fold_val = df.iloc[val_idx].copy()

    # --- Compute manufacturer PC prediction as baseline ----------
    print("[2/3] Computing manufacturer PC residual...")
    rho = fold_val["air_density"].fillna(1.225).to_numpy()
    ws120 = fold_val["wind_speed_120m"].to_numpy()
    n_active = fold_val["active_turbines"].to_numpy()

    p_per_turbine_kw = per_turbine_power_kw(ws120, rho)
    fold_val["mfr_pc_pred_mw"] = (p_per_turbine_kw * n_active / 1000.0).clip(0, CAPACITY_MW)

    # Existing isotonic prediction (on v_eff, fold-fit).
    fold_val["iso_pc_pred_mw"] = (
        fold_val["p_curve_global"].to_numpy() * fold_val["active_turbines_ratio"].to_numpy()
    ).clip(0, CAPACITY_MW)

    y = fold_val[TARGET_COL].to_numpy()
    print(f"  Fold-5 rows: {len(y)}")
    print(f"  target_mw stats: mean={y.mean():.3f}, std={y.std():.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")

    # --- Residual diagnostics --------------------------------------
    print("\n[3/3] Residual structure...\n")
    print(f"{'target':<28}  {'std':>8}  {'naive_nMAE':>12}  {'iso_pc_nMAE':>12}  {'R²(ws120)':>10}")
    print("-" * 80)

    def report(name: str, target_arr: np.ndarray, naive_pred: np.ndarray | float = None):
        std = target_arr.std()
        if naive_pred is None:
            naive_pred = float(target_arr.mean())
        if np.isscalar(naive_pred):
            naive_pred_arr = np.full_like(target_arr, naive_pred, dtype=float)
        else:
            naive_pred_arr = naive_pred
        # Always evaluate nMAE in MW space (so numbers are comparable across targets).
        # If target is CF or residual, we need to map predictions back to MW for nMAE.
        # For diagnostic clarity, report std AND nMAE-equivalent.
        print(f"{name:<28}  {std:>8.3f}", end="")

        # nMAE of "predict the target's own mean" baseline, on MW.
        if name == "target_mw":
            mw_naive = naive_pred_arr
        elif name == "CF (= target/(n×3.465))":
            mw_naive = naive_pred_arr * n_active * 3.465
        elif name.startswith("residual_mfr"):
            mw_naive = fold_val["mfr_pc_pred_mw"].to_numpy() + naive_pred_arr
        elif name.startswith("residual_iso"):
            mw_naive = fold_val["iso_pc_pred_mw"].to_numpy() + naive_pred_arr
        else:
            mw_naive = naive_pred_arr
        mw_naive = np.clip(mw_naive, 0.0, CAPACITY_MW)
        nmae_naive = normalized_mae(y, mw_naive)

        # nMAE if you predict using the isotonic baseline as MW.
        nmae_iso = normalized_mae(y, fold_val["iso_pc_pred_mw"].to_numpy())

        # R²(ws120) on the target itself (linear): how much linear signal
        # remains in the target.
        ws_centered = ws120 - ws120.mean()
        target_centered = target_arr - target_arr.mean()
        r2_ws = (np.corrcoef(ws_centered, target_centered)[0, 1] ** 2
                 if target_arr.std() > 1e-6 else 0.0)

        print(f"  {nmae_naive:>10.4f}%   {nmae_iso:>10.4f}%   {r2_ws:>10.4f}")

    # R1: raw target
    report("target_mw", y)

    # R2: CF
    cf = y / (n_active * 3.465)
    report("CF (= target/(n×3.465))", cf)

    # R3: residual after manufacturer PC
    resid_mfr = y - fold_val["mfr_pc_pred_mw"].to_numpy()
    print()
    print(f"residual_mfr stats:  mean={resid_mfr.mean():.3f}, std={resid_mfr.std():.3f}, "
          f"min={resid_mfr.min():.2f}, max={resid_mfr.max():.2f}")
    report("residual_mfr (= y - mfr_PC)", resid_mfr)

    # R4: residual after fold-fitted isotonic
    resid_iso = y - fold_val["iso_pc_pred_mw"].to_numpy()
    print(f"residual_iso stats:  mean={resid_iso.mean():.3f}, std={resid_iso.std():.3f}, "
          f"min={resid_iso.min():.2f}, max={resid_iso.max():.2f}")
    report("residual_iso (= y - iso_PC)", resid_iso)

    # --- How much of the target std the manufacturer PC eats? -----
    print("\n--- Variance decomposition ---")
    print(f"  target std             : {y.std():.3f} MW")
    print(f"  iso_PC pred std        : {fold_val['iso_pc_pred_mw'].std():.3f} MW")
    print(f"  iso_PC residual std    : {resid_iso.std():.3f} MW   "
          f"(= {(1 - resid_iso.var() / y.var()) * 100:.1f} % of variance explained)")
    print(f"  mfr_PC pred std        : {fold_val['mfr_pc_pred_mw'].std():.3f} MW")
    print(f"  mfr_PC residual std    : {resid_mfr.std():.3f} MW   "
          f"(= {(1 - resid_mfr.var() / y.var()) * 100:.1f} % of variance explained)")

    # --- One-feature single-LGBM on the residual ----------------
    print("\n--- Can a 1-feature LGBM predict residual_iso? ---")
    candidate_cols = [
        "wind_speed_120m", "ws120_v", "ws120_u",
        "era5v2_blh", "era5v2_wind_speed_84m",
        "wind_dir_120m_sin", "wind_dir_120m_cos",
        "n_repair", "active_turbines_ratio",
        "ws_shear_80_180", "alpha_local",
        "era5v2_air_density", "air_density",
        "hour_sin", "hour_cos", "month_sin", "doy_sin",
    ]
    fold_train = df.iloc[train_idx]
    train_y = fold_train[TARGET_COL].to_numpy()
    train_iso_pc = (fold_train["p_curve_global"].to_numpy()
                    * fold_train["active_turbines_ratio"].to_numpy()).clip(0, CAPACITY_MW)
    train_resid_iso = train_y - train_iso_pc
    val_y = y
    val_resid_iso = resid_iso

    print(f"  {'feature':<28} {'gain':>14}  {'nMAE_eval':>12}")
    print("  " + "-" * 60)
    for col in candidate_cols:
        if col not in fold_train.columns or col not in fold_val.columns:
            continue
        x_tr = fold_train[col].fillna(0.0).to_numpy().astype(np.float32).reshape(-1, 1)
        x_va = fold_val[col].fillna(0.0).to_numpy().astype(np.float32).reshape(-1, 1)
        try:
            dtrain = lgb.Dataset(x_tr, label=train_resid_iso, free_raw_data=False)
            dval = lgb.Dataset(x_va, label=val_resid_iso, free_raw_data=False)
            params = dict(
                objective="regression_l1", metric="mae",
                learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
                feature_fraction=1.0, bagging_fraction=1.0,
                seed=42, verbose=-1, deterministic=True, force_col_wise=True,
            )
            b = lgb.train(
                params, dtrain, num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred_resid = b.predict(x_va, num_iteration=b.best_iteration)
            # Reconstruct MW prediction.
            pred_mw = np.clip(fold_val["iso_pc_pred_mw"].to_numpy() + pred_resid,
                              0.0, CAPACITY_MW)
            nmae = normalized_mae(val_y, pred_mw)
            gain = float(b.feature_importance(importance_type="gain")[0])
            print(f"  {col:<28} {gain:>14,.0f}  {nmae:>10.4f}%")
        except Exception as e:
            print(f"  {col:<28}  failed: {e}")

    # Reference: predicting raw 0 residual = isotonic baseline.
    nmae_iso_only = normalized_mae(val_y, fold_val["iso_pc_pred_mw"].to_numpy())
    print(f"\n  baseline (iso_PC alone, no residual): {nmae_iso_only:.4f}%")

    # Reference: actual v32 OOF Fold-5.
    print(f"  reference (v32 CF OOF Fold-5)       : 7.5367 %")
    print(f"  reference (v34 CF OOF Fold-5)       : 7.5706 %")


if __name__ == "__main__":
    main()
