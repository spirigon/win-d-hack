"""V13: Autoregressive lag features with both optimistic and rolling CV.

Critical experiment: does autoregressive feedback help despite error
compounding over 2000+ hours? We evaluate two protocols:

Phase A (OPTIMISTIC): True lag values from training data. Gives upper bound.
Phase B (ROLLING): Simulated inference - iteratively predict each hour,
    feed prediction forward. Matches production conditions.

Decision:
- A ≥ 8.75%: STOP, lag features don't help even in the best case.
- B - A > 0.3pp: compounding wins, cut long lags, retry.
- B < 8.85%: viable for submission.

Usage:
    python -m src.training.train_autoreg
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
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
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v1.3_autoreg.csv"

V_CUT_IN = 3.0
V_CUT_OUT = 25.0
SEEDS = [42, 123, 456, 789, 2026]


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


def _train_single(X_tr, y_tr, X_va, y_va, feat_cols, seed, config_base):
    cfg = LGBMConfig(**{**config_base.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(
        cfg.to_params(),
        dtrain,
        num_boost_round=4000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )


def _train_full(X, y, feat_cols, seed, n_rounds, config_base):
    cfg = LGBMConfig(**{**config_base.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(cfg.to_params(), dtrain, num_boost_round=n_rounds)


def rolling_predict(
    booster: lgb.Booster,
    df_base: pd.DataFrame,
    history: PowerHistory,
    feat_cols: list[str],
) -> np.ndarray:
    """Predict hour-by-hour, feeding each prediction back as lag for the next.

    ``df_base`` must contain non-lag features (weather, calendar, power curve, ...)
    ALREADY COMPUTED. Lags are built on-the-fly from ``history``.

    Optimised: pre-extract non-lag columns once, only recompute lag columns
    per iteration.
    """
    from src.features.autoreg_lags import LAG_HOURS, ROLL_WINDOWS

    preds = np.empty(len(df_base), dtype=np.float32)
    ts_arr = pd.to_datetime(df_base[TIMESTAMP_COL]).to_numpy()
    v_eff_arr = df_base["v_eff"].to_numpy()

    # Pre-materialize the full feature matrix with placeholder zeros for lag cols.
    lag_col_names = set(lag_feature_columns())
    non_lag_cols = [c for c in feat_cols if c not in lag_col_names]
    lag_cols_ordered = [c for c in feat_cols if c in lag_col_names]

    X_nonlag = df_base[non_lag_cols].to_numpy(dtype=np.float32)
    # Final feature matrix will be non-lag + lag interleaved; we need to
    # reconstruct in feat_cols order. Build an index map.
    feat_idx = {c: i for i, c in enumerate(feat_cols)}
    nonlag_positions = [feat_idx[c] for c in non_lag_cols]
    lag_positions = [feat_idx[c] for c in lag_cols_ordered]

    x_row = np.zeros(len(feat_cols), dtype=np.float32)

    for i in range(len(df_base)):
        t = pd.Timestamp(ts_arr[i])
        # Non-lag values.
        for j, pos in enumerate(nonlag_positions):
            x_row[pos] = X_nonlag[i, j]
        # Lag values built from history.
        lag_vals: dict[str, float] = {}
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


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    # Fit full-data power curves once (we'll refit per-fold during CV).
    df_train = build_features(df_train, sort_by_time=False)
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)

    config_base = LGBMConfig(
        num_leaves=207,
        min_data_in_leaf=235,
        learning_rate=0.02221,
        feature_fraction=0.947,
        bagging_fraction=0.714,
        bagging_freq=6,
        lambda_l1=0.00192,
        lambda_l2=2.059,
        num_boost_round=4000,
        early_stopping_rounds=200,
        log_period=0,
    )

    # === Phase A: OPTIMISTIC evaluation (true lags on val) ===
    print("\n" + "=" * 70)
    print("PHASE A: OPTIMISTIC (true lag values on validation)")
    print("=" * 70)

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)

    # Build a history for training: use ONLY training rows' true power.
    # Fallback series from the global power curve at each timestamp (train+val).
    df_all_for_fallback = pd.concat([df_tr, df_va], ignore_index=True).sort_values(TIMESTAMP_COL)
    fallback = pd.Series(
        df_all_for_fallback["p_curve_global"].to_numpy(),
        index=pd.to_datetime(df_all_for_fallback[TIMESTAMP_COL]),
    )

    # Training history: only train targets.
    history_train_only = PowerHistory.from_frame(
        df_tr[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
        power_col="power",
        fallback_series=fallback,
    )
    # Optimistic val history: train targets + val targets (for upper bound).
    history_true_val = PowerHistory.from_frame(
        df_all_for_fallback[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
        power_col="power",
        fallback_series=fallback,
    )

    # Attach lags: training uses training-only history; val (optimistic) uses full-truth history.
    df_tr = attach_lag_features(df_tr, history_train_only)
    df_va_optimistic = attach_lag_features(df_va, history_true_val)

    feat_cols = [c for c in feature_columns(df_tr) if c not in ("_is_impossible",)]
    lag_cols = lag_feature_columns()
    print(f"  Total features: {len(feat_cols)} (including {len(lag_cols)} lag features)")

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_opt = df_va_optimistic[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va_optimistic[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va_optimistic["v_eff"].to_numpy()

    preds_opt_ensemble = []
    best_iters = []
    for seed in SEEDS:
        booster = _train_single(X_tr, y_tr, X_va_opt, y_va, feat_cols, seed, config_base)
        p = np.clip(booster.predict(X_va_opt, num_iteration=booster.best_iteration), 0, CAPACITY_MW)
        p_pp = _post_process(p, v_eff_va)
        nmae = normalized_mae(y_va, p_pp)
        preds_opt_ensemble.append(p_pp)
        best_iters.append(booster.best_iteration)
        print(f"  Seed {seed}: OPTIMISTIC Fold-5 nMAE = {nmae:.4f} %  (best_iter={booster.best_iteration})")

    preds_opt_ens = np.mean(preds_opt_ensemble, axis=0)
    nmae_opt = normalized_mae(y_va, preds_opt_ens)
    print(f"\n  >>> PHASE A (optimistic) ensemble nMAE: {nmae_opt:.4f} % <<<")

    # Gate: if optimistic isn't even better than 8.75%, don't bother with rolling.
    if nmae_opt > 8.75:
        print(f"  WARNING: Even optimistic is >= 8.75%. Lag features unlikely to help.")
        print("  (Continuing anyway for completeness.)")

    # === Phase B: ROLLING evaluation (autoregressive) ===
    print("\n" + "=" * 70)
    print("PHASE B: ROLLING (autoregressive, simulates inference)")
    print("=" * 70)

    # Retrain each seed but we need the booster objects — train once into list already available.
    # Actually we need a fresh training loop with PROPER lag construction for training too.
    # For training, we use TRUE prior values (leak-free because target is known in training).
    # Boosters from Phase A are already trained with that setup — reuse them.

    # For rolling evaluation, create a history with ONLY training values (no val targets).
    train_only_df = df_tr[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"})
    history_roll_template = PowerHistory.from_frame(
        train_only_df,
        power_col="power",
        fallback_series=fallback,
    )

    # Retrain boosters using the Phase-A history but using val lag values built
    # the same way they were in training. For the rolling evaluation we need
    # seeds x rolling passes — expensive. Do a single-seed rolling first as a
    # check, then ensemble if it's not too slow.
    print("  Rolling forward seed 42...")
    booster_42 = _train_single(X_tr, y_tr, X_va_opt, y_va, feat_cols, 42, config_base)

    # Copy the history so we don't pollute for subsequent seeds.
    import copy
    history_for_42 = copy.deepcopy(history_roll_template)
    # df_va WITHOUT lag columns (we rebuild on the fly).
    # Make sure row order is chronological for rolling.
    df_va_rolling = df_va.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    preds_rolling_42 = rolling_predict(booster_42, df_va_rolling, history_for_42, feat_cols)
    # Align back to df_va's original order.
    y_va_rolling = df_va_rolling[TARGET_COL].to_numpy(dtype=np.float32)
    nmae_roll_42 = normalized_mae(y_va_rolling, preds_rolling_42)
    print(f"  Rolling Fold-5 nMAE (seed 42): {nmae_roll_42:.4f} %")

    # If seed 42 rolling is close to optimistic, run the full ensemble.
    if nmae_roll_42 - nmae_opt < 0.5:
        print("  Rolling close to optimistic — running full ensemble rolling...")
        rolling_preds_all = [preds_rolling_42]
        for seed in SEEDS[1:]:
            booster_s = _train_single(X_tr, y_tr, X_va_opt, y_va, feat_cols, seed, config_base)
            history_s = copy.deepcopy(history_roll_template)
            p_roll = rolling_predict(booster_s, df_va_rolling, history_s, feat_cols)
            nmae_s = normalized_mae(y_va_rolling, p_roll)
            rolling_preds_all.append(p_roll)
            print(f"  Seed {seed}: rolling nMAE = {nmae_s:.4f} %")
        preds_roll_ens = np.mean(rolling_preds_all, axis=0)
        nmae_roll_ens = normalized_mae(y_va_rolling, preds_roll_ens)
        print(f"\n  >>> PHASE B (rolling) ensemble nMAE: {nmae_roll_ens:.4f} % <<<")
    else:
        print(f"  Gap too large ({nmae_roll_42 - nmae_opt:.2f}pp), autoregression compounds badly.")
        nmae_roll_ens = nmae_roll_42

    # === Decision ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Non-lag baseline (v1.2 final): ~8.76%")
    print(f"  PHASE A optimistic ensemble:   {nmae_opt:.4f}%  (gain: {8.76 - nmae_opt:+.3f}pp)")
    print(f"  PHASE B rolling ensemble:      {nmae_roll_ens:.4f}%  (vs baseline: {nmae_roll_ens - 8.76:+.3f}pp)")

    if nmae_roll_ens < 8.76:
        decision = "SHIP - rolling AR is better than baseline"
    elif nmae_opt < 8.76 and nmae_roll_ens > nmae_opt + 0.3:
        decision = "DO NOT SHIP - optimistic helps but rolling compounds"
    else:
        decision = "DO NOT SHIP - autoregressive lags not useful"
    print(f"\n  Decision: {decision}")

    # === Only generate submission if rolling ensemble beats baseline ===
    if nmae_roll_ens < 8.76:
        print("\n=== Generating autoregressive submission ===")
        # Full-fit with all data + autoregressive lags during training.
        df_train_clean = df_train[~df_train["_is_impossible"]]
        pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
        pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
        df_train_full = _add_power_curve_features(df_train, pc_sector_full, pc_global_full)

        # Full training history using true values.
        fallback_full = pd.Series(
            df_train_full["p_curve_global"].to_numpy(),
            index=pd.to_datetime(df_train_full[TIMESTAMP_COL]),
        )
        history_full = PowerHistory.from_frame(
            df_train_full[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
            power_col="power",
            fallback_series=fallback_full,
        )
        df_train_full = attach_lag_features(df_train_full, history_full)

        feat_cols_full = [c for c in feature_columns(df_train_full) if c not in ("_is_impossible",)]
        X_full = df_train_full[feat_cols_full].to_numpy(dtype=np.float32)
        y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

        n_rounds = max(int(np.median(best_iters) * 1.2), 1000)
        print(f"  Training {len(SEEDS)} models, n_rounds={n_rounds}")
        boosters_full = [_train_full(X_full, y_full, feat_cols_full, s, n_rounds, config_base) for s in SEEDS]

        # Inference: rolling forward on valid with history starting from TRAIN.
        df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
        fallback_valid = pd.Series(
            df_valid_pred["p_curve_global"].to_numpy(),
            index=pd.to_datetime(df_valid_pred[TIMESTAMP_COL]),
        )
        # Start with training history + valid fallback.
        def make_init_history():
            h = PowerHistory.from_frame(
                df_train_full[[TIMESTAMP_COL, TARGET_COL]].rename(columns={TARGET_COL: "power"}),
                power_col="power",
                fallback_series=pd.concat([fallback_full, fallback_valid]),
            )
            return h

        valid_rolling_all = []
        for i, b in enumerate(boosters_full):
            hist_i = make_init_history()
            p = rolling_predict(b, df_valid_pred, hist_i, feat_cols_full)
            valid_rolling_all.append(p)
            print(f"  Seed {SEEDS[i]}: rolling predictions done")

        preds_final = np.mean(valid_rolling_all, axis=0)
        preds_final = np.clip(preds_final, 0, CAPACITY_MW)

        order = df_valid_pred["_submission_row"].to_numpy().astype(int)
        preds_ordered = np.empty_like(preds_final)
        preds_ordered[order] = preds_final
        write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))
        print(f"  Submission saved: {SUBMISSION_PATH}")
    else:
        print("\n  Skipping submission generation (decision: do not ship).")


if __name__ == "__main__":
    main()
