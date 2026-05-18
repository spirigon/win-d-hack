"""Compare v7 vs v9 residuals to see where improvement/regression happens."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train
from src.data.outliers import identify_impossible_rows
from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.curtailment import add_curtailment_features, compute_curtail_rate_by_hour
from src.features.pipeline import build_features
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.models.lightgbm_model import LGBMConfig, predict_lgbm, train_lgbm

import lightgbm as lgb

TRAIN = _ROOT / "data" / "raw" / "train_dataset.csv"


def run(name: str, use_curtail: bool, use_outlier_fit: bool, n_sectors: int = 8):
    """Run a single config and return Fold-5 nMAE + residuals per regime."""
    df = load_train(TRAIN)
    impossible = identify_impossible_rows(df)
    df["_is_impossible"] = impossible.values
    df = build_features(df, sort_by_time=False)

    if use_curtail:
        clean = df[~df["_is_impossible"]]
        rates = compute_curtail_rate_by_hour(clean)
        df = add_curtailment_features(df, curtail_rates_by_hour=rates)

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df, fold5)

    fold_train = df.iloc[train_idx]
    if use_outlier_fit:
        fit_data = fold_train[~fold_train["_is_impossible"]]
    else:
        fit_data = fold_train

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=n_sectors)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = fold_train.copy()
    v_eff = df_tr["v_eff"].to_numpy()
    dir_deg = (df_tr["wind_direction_120m"] * 1000.0).to_numpy()
    df_tr["p_curve_sector"] = pc_sector.predict(v_eff, dir_deg)
    df_tr["p_curve_global"] = pc_global.predict(v_eff)
    df_tr["p_curve_x_active"] = df_tr["p_curve_sector"] * df_tr["active_turbines_ratio"]
    df_tr["p_curve_ratio"] = df_tr["p_curve_sector"] / 90.09
    df_tr["p_curve_sector_minus_global"] = df_tr["p_curve_sector"] - df_tr["p_curve_global"]

    df_va = df.iloc[val_idx].copy()
    v_eff_v = df_va["v_eff"].to_numpy()
    dir_deg_v = (df_va["wind_direction_120m"] * 1000.0).to_numpy()
    df_va["p_curve_sector"] = pc_sector.predict(v_eff_v, dir_deg_v)
    df_va["p_curve_global"] = pc_global.predict(v_eff_v)
    df_va["p_curve_x_active"] = df_va["p_curve_sector"] * df_va["active_turbines_ratio"]
    df_va["p_curve_ratio"] = df_va["p_curve_sector"] / 90.09
    df_va["p_curve_sector_minus_global"] = df_va["p_curve_sector"] - df_va["p_curve_global"]

    drop = {TIMESTAMP_COL, TARGET_COL, "_submission_row", "_is_impossible"}
    feat_cols = [c for c in df_tr.columns if c not in drop and pd.api.types.is_numeric_dtype(df_tr[c])]

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    config = LGBMConfig(
        num_leaves=207, min_data_in_leaf=235, learning_rate=0.02221,
        feature_fraction=0.947, bagging_fraction=0.714, bagging_freq=6,
        lambda_l1=0.00192, lambda_l2=2.059,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )
    booster = train_lgbm(X_tr, y_tr, X_va, y_va, feature_names=feat_cols, config=config)
    preds = predict_lgbm(booster, X_va)
    # Post-process.
    preds_pp = np.where(v_eff_v < 3.0, 0.0, preds)
    preds_pp = np.where(v_eff_v > 25.0, np.minimum(preds_pp, 5.0), preds_pp)
    preds_pp = np.clip(preds_pp, 0.0, 90.09)

    nmae = normalized_mae(y_va, preds_pp)
    abs_err = np.abs(y_va - preds_pp)
    residual = y_va - preds_pp

    # Residuals by wind speed bin.
    df_va_eval = pd.DataFrame({
        "ws": df_va["wind_speed_120m"].to_numpy(),
        "gust": df_va["wind_gusts_10m"].to_numpy(),
        "y": y_va,
        "pred": preds_pp,
        "abs_err": abs_err,
        "residual": residual,
    })
    df_va_eval["ws_bin"] = pd.cut(df_va_eval["ws"], [0, 3, 5, 7, 10, 14, 25])
    by_bin = df_va_eval.groupby("ws_bin", observed=True).agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        bias=("residual", "mean"),
    ).round(3)

    print(f"\n=== {name} (Fold-5 nMAE = {nmae:.4f}%, n_feat={len(feat_cols)}) ===")
    print(by_bin)
    return nmae, feat_cols


print("Comparing configurations on Fold-5...")

# Baseline v7: no curtail features, no outlier fit
n1, f1 = run("v7 (no curtail, full power curve fit)", use_curtail=False, use_outlier_fit=False)

# v7 but with outlier fit only
n2, f2 = run("v7+clean_fit (no curtail, outlier-filtered fit)", use_curtail=False, use_outlier_fit=True)

# v9: clean fit + curtail features
n3, f3 = run("v9 (clean fit + curtail features)", use_curtail=True, use_outlier_fit=True)

print("\n" + "=" * 60)
print("SUMMARY:")
print(f"  v7 (baseline)            : {n1:.4f}%  ({len(f1)} features)")
print(f"  v7 + clean power curve   : {n2:.4f}%  ({len(f2)} features)")
print(f"  v9 (clean + curtail)     : {n3:.4f}%  ({len(f3)} features)")
