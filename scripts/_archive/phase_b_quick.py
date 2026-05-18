"""Phase B: rolling (autoregressive) evaluation on Fold-5.

For each hour in val, predict using accumulating history (own predictions).
Checks how badly error compounds over the Q1 quarter.
"""

from __future__ import annotations

import copy
import sys
import time
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
    LAG_HOURS,
    ROLL_WINDOWS,
    PowerHistory,
    attach_lag_features,
    lag_feature_columns,
)
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

V_CUT_IN = 3.0
V_CUT_OUT = 25.0


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
    preds = np.asarray(preds, dtype=float)
    preds = np.where(v_eff < V_CUT_IN, 0.0, preds)
    preds = np.where(v_eff > V_CUT_OUT, np.minimum(preds, 5.0), preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


def rolling_predict(booster, df_base, history, feat_cols):
    preds = np.empty(len(df_base), dtype=np.float32)
    ts_arr = pd.to_datetime(df_base[TIMESTAMP_COL]).to_numpy()
    v_eff_arr = df_base["v_eff"].to_numpy()

    lag_col_set = set(lag_feature_columns())
    non_lag_cols = [c for c in feat_cols if c not in lag_col_set]
    lag_cols_ordered = [c for c in feat_cols if c in lag_col_set]

    X_nonlag = df_base[non_lag_cols].to_numpy(dtype=np.float32)
    feat_idx = {c: i for i, c in enumerate(feat_cols)}
    nonlag_positions = [feat_idx[c] for c in non_lag_cols]
    lag_positions = [feat_idx[c] for c in lag_cols_ordered]

    x_row = np.zeros(len(feat_cols), dtype=np.float32)

    for i in range(len(df_base)):
        t = pd.Timestamp(ts_arr[i])
        for j, pos in enumerate(nonlag_positions):
            x_row[pos] = X_nonlag[i, j]

        lag_vals = {}
        for lag in LAG_HOURS:
            lag_vals[f"power_lag_{lag}h"] = history.lookup(t - pd.Timedelta(hours=lag))
        for w in ROLL_WINDOWS:
            window = [history.lookup(t - pd.Timedelta(hours=k)) for k in range(1, w + 1)]
            arr = np.asarray(window, dtype=np.float32)
            lag_vals[f"power_roll_mean_{w}h"] = float(arr.mean())
            lag_vals[f"power_roll_std_{w}h"] = float(arr.std())
            lag_vals[f"power_roll_max_{w}h"] = float(arr.max())
        lag_vals["power_diff_1h"] = lag_vals["power_lag_1h"] - lag_vals["power_lag_2h"]
        lag_vals["power_diff_3h"] = lag_vals["power_lag_1h"] - lag_vals["power_lag_3h"]
        for name, pos in zip(lag_cols_ordered, lag_positions, strict=True):
            x_row[pos] = lag_vals[name]

        pred = float(booster.predict(x_row.reshape(1, -1))[0])
        v_eff = float(v_eff_arr[i])
        if v_eff < V_CUT_IN:
            pred = 0.0
        elif v_eff > V_CUT_OUT:
            pred = min(pred, 5.0)
        pred = float(np.clip(pred, 0.0, CAPACITY_MW))
        preds[i] = pred
        history.add(t, pred)

    return preds


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
    df_va = _add_power_curve_features(df.iloc[val_idx], pc_sector, pc_global).sort_values(TIMESTAMP_COL).reset_index(drop=True)

    df_all_fb = pd.concat([df_tr, df_va], ignore_index=True).sort_values(TIMESTAMP_COL)
    fallback = pd.Series(df_all_fb["p_curve_global"].to_numpy(), index=pd.to_datetime(df_all_fb[TIMESTAMP_COL]))

    # Training history: train targets + fallback.
    hist_tr = PowerHistory.from_frame(
        df_tr[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
        power_col="power", fallback_series=fallback,
    )
    df_tr_lag = attach_lag_features(df_tr, hist_tr)

    feat_cols = [c for c in feature_columns(df_tr_lag) if c != "_is_impossible"]
    X_tr = df_tr_lag[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr_lag[TARGET_COL].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    # Optimistic val for training early stopping.
    hist_opt = PowerHistory.from_frame(
        df_all_fb[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
        power_col="power", fallback_series=fallback,
    )
    df_va_opt = attach_lag_features(df_va, hist_opt)
    X_va_opt = df_va_opt[feat_cols].to_numpy(dtype=np.float32)

    v_eff_va = df_va["v_eff"].to_numpy()

    # Rolling-history template (train targets only, no val).
    history_template = PowerHistory.from_frame(
        df_tr[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
        power_col="power", fallback_series=fallback,
    )

    SEEDS = [42, 123, 456, 789, 2026]

    # Train seed 42 first as a probe.
    print("=== Seed 42 probe ===")
    cfg = LGBMConfig(**{**config.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va_opt, label=y_va, feature_name=feat_cols, free_raw_data=False)
    b42 = lgb.train(
        cfg.to_params(), dtrain, num_boost_round=4000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    print(f"  best_iter: {b42.best_iteration}")

    print("  Rolling forward 2016 hours...")
    t0 = time.time()
    hist_roll_42 = copy.deepcopy(history_template)
    preds_roll_42 = rolling_predict(b42, df_va, hist_roll_42, feat_cols)
    elapsed = time.time() - t0
    nmae_roll_42 = normalized_mae(y_va, preds_roll_42)
    print(f"  Rolling took {elapsed:.1f} s")
    print(f"  Seed 42 rolling Fold-5 nMAE: {nmae_roll_42:.4f} %")

    # If probe is promising (< baseline 8.76), run full ensemble.
    if nmae_roll_42 < 8.76:
        print(f"\n  Rolling beats baseline. Running full ensemble ({len(SEEDS)} seeds)...")
        rolling_preds = [preds_roll_42]
        for s in SEEDS[1:]:
            print(f"  Seed {s}...")
            cfg_s = LGBMConfig(**{**config.__dict__, "seed": s})
            dtrain_s = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
            dval_s = lgb.Dataset(X_va_opt, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b_s = lgb.train(
                cfg_s.to_params(), dtrain_s, num_boost_round=4000,
                valid_sets=[dtrain_s, dval_s], valid_names=["train", "val"],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            hist_s = copy.deepcopy(history_template)
            p_s = rolling_predict(b_s, df_va, hist_s, feat_cols)
            nmae_s = normalized_mae(y_va, p_s)
            rolling_preds.append(p_s)
            print(f"    Seed {s} rolling nMAE: {nmae_s:.4f} %")
        ens = np.mean(rolling_preds, axis=0)
        ens_nmae = normalized_mae(y_va, ens)
        print(f"\n  >>> Rolling ensemble Fold-5 nMAE: {ens_nmae:.4f} % <<<")
    else:
        print(f"\n  Rolling ({nmae_roll_42:.4f}%) worse than baseline (8.76%). Stop.")

    # Per-hour diagnostics: how does error grow over the quarter?
    print("\n  Rolling error drift over time (every ~400 hours):")
    n = len(df_va)
    chunks = [(0, 400), (400, 800), (800, 1200), (1200, 1600), (1600, n)]
    for lo, hi in chunks:
        chunk_nmae = normalized_mae(y_va[lo:hi], preds_roll_42[lo:hi])
        print(f"    hours [{lo:4d}-{hi:4d}): nMAE = {chunk_nmae:.4f} %")


if __name__ == "__main__":
    main()
