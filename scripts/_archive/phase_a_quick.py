"""Quick Phase-A only: does LGBM benefit from TRUE lags on Fold-5?

If even optimistic (perfect lag values) doesn't beat 8.75% meaningfully,
autoregressive rolling can't help either, so we stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.autoreg_lags import (
    PowerHistory,
    attach_lag_features,
    lag_feature_columns,
)
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed


def _add_power_curve_features(df, pc_sector, pc_global):
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


def _post_process(preds, v_eff):
    V_IN, V_OUT = 3.0, 25.0
    preds = np.asarray(preds, dtype=float)
    preds = np.where(v_eff < V_IN, 0.0, preds)
    preds = np.where(v_eff > V_OUT, np.minimum(preds, 5.0), preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


def main():
    set_global_seed(42)
    TRAIN = _ROOT / "data" / "raw" / "train_dataset.csv"
    df = load_train(TRAIN)
    impossible = identify_impossible_rows(df)
    df["_is_impossible"] = impossible.values
    df = build_features(df, sort_by_time=False)

    config = LGBMConfig(
        num_leaves=207, min_data_in_leaf=235, learning_rate=0.02221,
        feature_fraction=0.947, bagging_fraction=0.714, bagging_freq=6,
        lambda_l1=0.00192, lambda_l2=2.059,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df, fold5)

    fold_train = df.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_va = _add_power_curve_features(df.iloc[val_idx], pc_sector, pc_global)

    # Baseline: no lag features, multi-seed ensemble.
    print("=== Baseline (no lags) ===")
    feat_cols_base = [c for c in feature_columns(df_tr) if c != "_is_impossible"]
    X_tr = df_tr[feat_cols_base].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols_base].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()

    SEEDS = [42, 123, 456, 789, 2026]
    baseline_preds = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols_base, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols_base, free_raw_data=False)
        b = lgb.train(
            cfg.to_params(), dtrain, num_boost_round=4000,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        p = np.clip(b.predict(X_va, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        baseline_preds.append(_post_process(p, v_eff_va))
    base_ens = np.mean(baseline_preds, axis=0)
    print(f"  Baseline Fold-5 ensemble nMAE: {normalized_mae(y_va, base_ens):.4f} %")

    # === Phase A: Optimistic lags ===
    print("\n=== Phase A: TRUE lag values (optimistic upper bound) ===")
    df_all = pd.concat([df_tr, df_va], ignore_index=True).sort_values(TIMESTAMP_COL)
    fallback = pd.Series(df_all["p_curve_global"].to_numpy(), index=pd.to_datetime(df_all[TIMESTAMP_COL]))

    # Training history (train-only powers).
    hist_train = PowerHistory.from_frame(
        df_tr[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
        power_col="power", fallback_series=fallback,
    )
    # Optimistic val history (train+val true powers).
    hist_opt = PowerHistory.from_frame(
        df_all[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
        power_col="power", fallback_series=fallback,
    )

    df_tr_lag = attach_lag_features(df_tr, hist_train)
    df_va_lag = attach_lag_features(df_va, hist_opt)
    feat_cols_lag = [c for c in feature_columns(df_tr_lag) if c != "_is_impossible"]
    print(f"  Features: {len(feat_cols_lag)} (including {len(lag_feature_columns())} lag features)")

    X_tr_lag = df_tr_lag[feat_cols_lag].to_numpy(dtype=np.float32)
    X_va_lag = df_va_lag[feat_cols_lag].to_numpy(dtype=np.float32)

    lag_preds = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr_lag, label=y_tr, feature_name=feat_cols_lag, free_raw_data=False)
        dval = lgb.Dataset(X_va_lag, label=y_va, feature_name=feat_cols_lag, free_raw_data=False)
        b = lgb.train(
            cfg.to_params(), dtrain, num_boost_round=4000,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        p = np.clip(b.predict(X_va_lag, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        p_pp = _post_process(p, v_eff_va)
        lag_preds.append(p_pp)
        nmae = normalized_mae(y_va, p_pp)
        print(f"  Seed {s}: OPTIMISTIC nMAE = {nmae:.4f} %")

    lag_ens = np.mean(lag_preds, axis=0)
    print(f"\n  OPTIMISTIC Fold-5 ensemble nMAE: {normalized_mae(y_va, lag_ens):.4f} %")

    # Feature importance — where are the lags ranking?
    b_last = b  # last seed's booster
    imp = b_last.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_lag, imp), key=lambda x: -x[1])
    print("\n  Top 15 features (gain):")
    for name, score in feat_imp[:15]:
        marker = "  *" if name in lag_feature_columns() else "   "
        print(f"  {marker} {name:40s} {score:12.1f}")


if __name__ == "__main__":
    main()
