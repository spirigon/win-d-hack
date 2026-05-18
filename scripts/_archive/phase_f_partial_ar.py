"""Phase F: Hybrid — use autoregressive lags for first N hours, then switch to no-lag baseline.

The first hours of Q1-2026 have TRUE lag values from training. If we can
exploit them for the first 24-48 hours, we might shave some error even if
the later hours just use the baseline model.
"""

from __future__ import annotations

import copy
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
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

V_CUT_IN = 3.0
V_CUT_OUT = 25.0


class MiniHistory:
    def __init__(self, fallback):
        self._known = {}
        self._fallback = fallback

    def add(self, ts, p):
        self._known[pd.Timestamp(ts)] = float(np.clip(p, 0, CAPACITY_MW))

    def lookup(self, ts):
        ts = pd.Timestamp(ts)
        if ts in self._known:
            return self._known[ts]
        if ts in self._fallback.index:
            return float(self._fallback.loc[ts])
        return 0.0


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


def attach_short_lags(df, history, max_lag=3):
    df = df.copy()
    ts_arr = pd.to_datetime(df[TIMESTAMP_COL]).to_numpy()
    for lag in range(1, max_lag + 1):
        vals = np.array([history.lookup(pd.Timestamp(t) - pd.Timedelta(hours=lag)) for t in ts_arr], dtype=np.float32)
        df[f"power_lag_{lag}h"] = vals
    return df


def short_lag_cols(max_lag=3):
    return [f"power_lag_{l}h" for l in range(1, max_lag + 1)]


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

    df_all = pd.concat([df_tr, df_va], ignore_index=True).sort_values(TIMESTAMP_COL)
    fallback = pd.Series(df_all["p_curve_global"].to_numpy(), index=pd.to_datetime(df_all[TIMESTAMP_COL]))

    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()
    ts_va = pd.to_datetime(df_va[TIMESTAMP_COL]).to_numpy()

    # === BASELINE: no-lag model ===
    print("=== No-lag baseline (reference) ===")
    feat_cols_base = [c for c in feature_columns(df_tr) if c != "_is_impossible"]
    X_tr_base = df_tr[feat_cols_base].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_base = df_va[feat_cols_base].to_numpy(dtype=np.float32)

    cfg = LGBMConfig(**{**config.__dict__, "seed": 42})
    dtrain_b = lgb.Dataset(X_tr_base, label=y_tr, feature_name=feat_cols_base, free_raw_data=False)
    dval_b = lgb.Dataset(X_va_base, label=y_va, feature_name=feat_cols_base, free_raw_data=False)
    b_base = lgb.train(
        cfg.to_params(), dtrain_b, num_boost_round=4000,
        valid_sets=[dtrain_b, dval_b], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    preds_base = _post_process(np.clip(b_base.predict(X_va_base, num_iteration=b_base.best_iteration), 0, CAPACITY_MW), v_eff_va)
    nmae_base = normalized_mae(y_va, preds_base)
    print(f"  Baseline nMAE: {nmae_base:.4f} %")

    # === LAG MODEL: with short lags ===
    print("\n=== Lag model (max_lag=3) ===")
    hist_tr = MiniHistory(fallback)
    for ts, p in zip(df_tr[TIMESTAMP_COL].to_numpy(), df_tr[TARGET_COL].to_numpy(), strict=True):
        hist_tr.add(ts, p)

    df_tr_lag = attach_short_lags(df_tr, hist_tr, max_lag=3)
    hist_opt = MiniHistory(fallback)
    for ts, p in zip(df_all[TIMESTAMP_COL].to_numpy(), df_all[TARGET_COL].to_numpy(), strict=True):
        hist_opt.add(ts, p)
    df_va_opt = attach_short_lags(df_va, hist_opt, max_lag=3)

    feat_cols = [c for c in feature_columns(df_tr_lag) if c != "_is_impossible"]
    X_tr = df_tr_lag[feat_cols].to_numpy(dtype=np.float32)
    X_va_opt = df_va_opt[feat_cols].to_numpy(dtype=np.float32)

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va_opt, label=y_va, feature_name=feat_cols, free_raw_data=False)
    b_lag = lgb.train(
        cfg.to_params(), dtrain, num_boost_round=4000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    preds_opt = _post_process(np.clip(b_lag.predict(X_va_opt, num_iteration=b_lag.best_iteration), 0, CAPACITY_MW), v_eff_va)
    print(f"  Optimistic nMAE: {normalized_mae(y_va, preds_opt):.4f} %")

    # === HYBRID: autoregressive for first N hours, then baseline ===
    # For each first N hours, compute rolling lags; for the rest, use baseline predictions.
    print("\n=== HYBRID: AR for first N hours, then baseline ===")
    for n_ar in [1, 3, 6, 12, 24, 48, 72, 168]:
        hist_roll = MiniHistory(fallback)
        for ts, p in zip(df_tr[TIMESTAMP_COL].to_numpy(), df_tr[TARGET_COL].to_numpy(), strict=True):
            hist_roll.add(ts, p)

        lag_set = set(short_lag_cols(3))
        non_lag = [c for c in feat_cols if c not in lag_set]
        lag_ord = [c for c in feat_cols if c in lag_set]
        X_nonlag = df_va[non_lag].to_numpy(dtype=np.float32)
        feat_idx = {c: i for i, c in enumerate(feat_cols)}
        nonlag_pos = [feat_idx[c] for c in non_lag]
        lag_pos = [feat_idx[c] for c in lag_ord]
        x_row = np.zeros(len(feat_cols), dtype=np.float32)

        preds_hybrid = np.empty(len(df_va), dtype=np.float32)
        for i in range(len(df_va)):
            if i < n_ar:
                # Use lag model.
                t = pd.Timestamp(ts_va[i])
                for j, pos in enumerate(nonlag_pos):
                    x_row[pos] = X_nonlag[i, j]
                for lag in range(1, 4):
                    lag_val = hist_roll.lookup(t - pd.Timedelta(hours=lag))
                    x_row[feat_idx[f"power_lag_{lag}h"]] = lag_val
                pred = float(b_lag.predict(x_row.reshape(1, -1))[0])
                pred = _post_process(np.array([pred]), np.array([v_eff_va[i]]))[0]
                hist_roll.add(t, pred)
                preds_hybrid[i] = pred
            else:
                # Use baseline.
                preds_hybrid[i] = preds_base[i]

        nmae_hybrid = normalized_mae(y_va, preds_hybrid)
        # Where does the improvement come from?
        delta = nmae_hybrid - nmae_base
        print(f"  n_ar={n_ar:3d}: nMAE = {nmae_hybrid:.4f} %  (delta vs baseline: {delta:+.4f}pp)")


if __name__ == "__main__":
    main()
