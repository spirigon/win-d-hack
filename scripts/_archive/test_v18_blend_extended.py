"""Extended blend search for v18: test w_mlp up to 0.90 and multi-fold check.

If the optimum is near 0.50 → genuine diversity improvement.
If the optimum is near 1.00 → MLP actually dominates, not just a blend win.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
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
from src.training.train_v18_mlp_blend import (
    CONFIG, SEEDS_SPEC, MLP_SEEDS, K, TURBINE_RATED_MW,
    _add_pc, merge_era5, add_era5_rolling, to_cf, from_cf,
    train_mlp_ensemble, train_lgbm_specialist_valid, DEVICE,
)

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"


def eval_fold(df_train, fold, feat_imp, K_val, fold_name):
    train_idx, val_idx = split_indices(df_train, fold)
    fold_train = df_train.iloc[train_idx]
    fold_val = df_train.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    top_k = [n for n, _ in feat_imp[:K_val]]
    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(y_va_mw, active_va)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    # LGBM specialists.
    regime_preds = []
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        preds_cf, _ = train_lgbm_specialist_valid(X_tr, y_tr_cf, X_va, y_va_cf, top_k, SEEDS_SPEC, weights)
        regime_preds.append(preds_cf)
    lgbm_cf = np.mean(regime_preds, axis=0)

    # MLP.
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_tr_std = np.nan_to_num((X_tr - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_va_std = np.nan_to_num((X_va - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mlp_cf = train_mlp_ensemble(X_tr_std, y_tr_cf, X_va_std, y_va_cf, MLP_SEEDS)

    # Sweep fuller range.
    print(f"\n{fold_name}:")
    best = (0.0, float("inf"))
    for w in np.arange(0.0, 1.01, 0.05):
        blend_cf = (1 - w) * lgbm_cf + w * mlp_cf
        blend_mw = np.clip(from_cf(blend_cf, active_va), 0, CAPACITY_MW)
        nmae = normalized_mae(y_va_mw, blend_mw)
        if nmae < best[1]:
            best = (w, nmae)
    lgbm_mw = np.clip(from_cf(lgbm_cf, active_va), 0, CAPACITY_MW)
    mlp_mw = np.clip(from_cf(mlp_cf, active_va), 0, CAPACITY_MW)
    print(f"  LGBM alone: {normalized_mae(y_va_mw, lgbm_mw):.4f}%")
    print(f"  MLP alone:  {normalized_mae(y_va_mw, mlp_mw):.4f}%")
    print(f"  Best blend: w_mlp={best[0]:.2f} -> {best[1]:.4f}%")
    return best


def main():
    set_global_seed(42)
    print(f"Device: {DEVICE}")
    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
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

    # Get feature importance from Fold-5.
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fold_val = df_train.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)
    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)
    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]
    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(df_va[TARGET_COL].to_numpy(dtype=np.float32), active_va)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)

    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dt = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dv = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])

    # Test fold-1 (winter 2023) and fold-5 (winter 2025).
    f1_best = eval_fold(df_train, folds[0], feat_imp, K, "Fold-1 (winter 2023)")
    f5_best = eval_fold(df_train, folds[-1], feat_imp, K, "Fold-5 (winter 2025)")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Fold-1 best blend: w_mlp={f1_best[0]:.2f} -> {f1_best[1]:.4f}%")
    print(f"  Fold-5 best blend: w_mlp={f5_best[0]:.2f} -> {f5_best[1]:.4f}%")


if __name__ == "__main__":
    main()
