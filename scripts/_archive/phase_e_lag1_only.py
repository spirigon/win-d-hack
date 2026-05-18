"""Phase E: Use only lag_1h and lag_2h (very short memory).

Shorter lags minimize compounding depth.
Also test blending rolling predictions with power-curve floor.
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


def attach_short_lags(df, history, max_lag=2):
    """Only compute power_lag_1h and power_lag_2h."""
    df = df.copy()
    ts_arr = pd.to_datetime(df[TIMESTAMP_COL]).to_numpy()
    for lag in range(1, max_lag + 1):
        vals = np.array([history.lookup(pd.Timestamp(t) - pd.Timedelta(hours=lag)) for t in ts_arr], dtype=np.float32)
        df[f"power_lag_{lag}h"] = vals
    return df


def short_lag_cols(max_lag=2):
    return [f"power_lag_{l}h" for l in range(1, max_lag + 1)]


def rolling_predict_short(booster, df_base, history, feat_cols, max_lag, blend=0.0):
    """Rolling with short lags + optional blend with p_curve_global."""
    preds = np.empty(len(df_base), dtype=np.float32)
    ts_arr = pd.to_datetime(df_base[TIMESTAMP_COL]).to_numpy()
    v_eff_arr = df_base["v_eff"].to_numpy()
    lag_names = short_lag_cols(max_lag)
    lag_set = set(lag_names)
    non_lag = [c for c in feat_cols if c not in lag_set]
    lag_ord = [c for c in feat_cols if c in lag_set]
    X_nonlag = df_base[non_lag].to_numpy(dtype=np.float32)
    p_curve_global = df_base["p_curve_global"].to_numpy()

    feat_idx = {c: i for i, c in enumerate(feat_cols)}
    nonlag_pos = [feat_idx[c] for c in non_lag]
    lag_pos = [feat_idx[c] for c in lag_ord]
    x_row = np.zeros(len(feat_cols), dtype=np.float32)

    for i in range(len(df_base)):
        t = pd.Timestamp(ts_arr[i])
        for j, pos in enumerate(nonlag_pos):
            x_row[pos] = X_nonlag[i, j]
        lag_vals = {}
        for lag in range(1, max_lag + 1):
            lag_vals[f"power_lag_{lag}h"] = history.lookup(t - pd.Timedelta(hours=lag))
        for name, pos in zip(lag_ord, lag_pos, strict=True):
            x_row[pos] = lag_vals[name]

        pred = float(booster.predict(x_row.reshape(1, -1))[0])
        v_eff = float(v_eff_arr[i])
        if v_eff < V_CUT_IN:
            pred = 0.0
        elif v_eff > V_CUT_OUT:
            pred = min(pred, 5.0)
        pred = float(np.clip(pred, 0.0, CAPACITY_MW))
        # Blend with power curve to dampen drift.
        if blend > 0:
            pred = (1 - blend) * pred + blend * p_curve_global[i]
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

    df_all = pd.concat([df_tr, df_va], ignore_index=True).sort_values(TIMESTAMP_COL)
    fallback = pd.Series(df_all["p_curve_global"].to_numpy(), index=pd.to_datetime(df_all[TIMESTAMP_COL]))

    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    # === Test: only lag-1h ===
    for max_lag in [1, 2, 3]:
        hist_tr = MiniHistory(fallback)
        for ts, p in zip(df_tr[TIMESTAMP_COL].to_numpy(), df_tr[TARGET_COL].to_numpy(), strict=True):
            hist_tr.add(ts, p)

        df_tr_lag = attach_short_lags(df_tr, hist_tr, max_lag=max_lag)
        hist_opt = MiniHistory(fallback)
        for ts, p in zip(df_all[TIMESTAMP_COL].to_numpy(), df_all[TARGET_COL].to_numpy(), strict=True):
            hist_opt.add(ts, p)
        df_va_opt = attach_short_lags(df_va, hist_opt, max_lag=max_lag)

        feat_cols = [c for c in feature_columns(df_tr_lag) if c != "_is_impossible"]
        X_tr = df_tr_lag[feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_tr_lag[TARGET_COL].to_numpy(dtype=np.float32)
        X_va_opt = df_va_opt[feat_cols].to_numpy(dtype=np.float32)

        cfg = LGBMConfig(**{**config.__dict__, "seed": 42})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va_opt, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(
            cfg.to_params(), dtrain, num_boost_round=4000,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        # Optimistic
        preds_opt = _post_process(
            np.clip(b.predict(X_va_opt, num_iteration=b.best_iteration), 0, CAPACITY_MW),
            df_va["v_eff"].to_numpy(),
        )
        nmae_opt = normalized_mae(y_va, preds_opt)

        # Rolling with different blend levels.
        for blend in [0.0, 0.2, 0.5, 0.8]:
            hist_roll = MiniHistory(fallback)
            for ts, p in zip(df_tr[TIMESTAMP_COL].to_numpy(), df_tr[TARGET_COL].to_numpy(), strict=True):
                hist_roll.add(ts, p)
            preds_roll = rolling_predict_short(b, df_va, hist_roll, feat_cols, max_lag, blend=blend)
            nmae_roll = normalized_mae(y_va, preds_roll)
            print(f"  max_lag={max_lag}, blend={blend:.2f}: optimistic={nmae_opt:.4f}%, rolling={nmae_roll:.4f}%")
        print()


if __name__ == "__main__":
    main()
