"""V6: Multi-seed LightGBM ensemble + optimized power curve + post-processing.

Strategy to close the gap to 7.8%:
1. Train multiple LightGBM models with different seeds for diversity.
2. Use a finer power curve (0.25 m/s bins, 16 sectors).
3. Post-process: isotonic regression to enforce monotonicity in wind speed.
4. Blend multi-seed predictions (reduces variance without adding bias).

Usage:
    python -m src.training.train_v6
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.features.power_curve import fit_power_curve
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v0.6_multi_seed.csv"

N_SEEDS = 5  # Number of different seeds for the ensemble.


def _add_power_curve_features(df: pd.DataFrame, pc8, pc16) -> pd.DataFrame:
    ws = df["wind_speed_80m"].to_numpy(dtype=float)
    ws120 = df["wind_speed_120m"].to_numpy(dtype=float)
    dir_deg = df["wind_direction_80m"].to_numpy(dtype=float) * 1000.0

    df = df.copy()

    # 8-sector.
    ds8 = (dir_deg // 45).astype(int) % 8
    df["p_theoretical"] = pc8.predict(ws, ds8)
    df["p_theo_ratio"] = df["p_theoretical"] / CAPACITY_MW
    df["p_theo_x_active"] = df["p_theoretical"] * df["active_turbines"] / 26.0

    # 16-sector.
    ds16 = (dir_deg // 22.5).astype(int) % 16
    df["p_theoretical_16s"] = pc16.predict(ws, ds16)
    df["p_theo_16s_x_active"] = df["p_theoretical_16s"] * df["active_turbines"] / 26.0

    # 120 m.
    df["p_theoretical_120m"] = pc8.predict(ws120, ds8)
    df["p_theo_diff_80_120"] = df["p_theoretical"] - df["p_theoretical_120m"]

    return df


def _train_single_seed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    feat_cols: list[str],
    seed: int,
) -> lgb.Booster:
    """Train a single LightGBM model with a specific seed."""
    config = LGBMConfig(
        num_leaves=434,
        min_data_in_leaf=124,
        learning_rate=0.0128,
        feature_fraction=0.644,
        bagging_fraction=0.747,
        bagging_freq=2,
        lambda_l1=0.00736,
        lambda_l2=0.00108,
        num_boost_round=4000,
        early_stopping_rounds=200,
        log_period=0,
        seed=seed,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
    callbacks = [lgb.early_stopping(200, verbose=False)]
    booster = lgb.train(
        config.to_params(),
        dtrain,
        num_boost_round=4000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return booster


def _train_full_seed(
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: list[str],
    seed: int,
    n_rounds: int = 2500,
) -> lgb.Booster:
    """Full-fit a single model."""
    config = LGBMConfig(
        num_leaves=434,
        min_data_in_leaf=124,
        learning_rate=0.0128,
        feature_fraction=0.644,
        bagging_fraction=0.747,
        bagging_freq=2,
        lambda_l1=0.00736,
        lambda_l2=0.00108,
        num_boost_round=n_rounds,
        log_period=0,
        seed=seed,
    )
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
    booster = lgb.train(config.to_params(), dtrain, num_boost_round=n_rounds)
    return booster


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    print(f"  Train: {len(df_train)} rows, Valid: {len(df_valid)} rows")

    print("Building features...")
    df_train = build_features(df_train, sort_by_time=False)
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)

    # Fold-5 for evaluation.
    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    # Power curves.
    pc8 = fit_power_curve(df_train.iloc[train_idx], n_sectors=8, ws_bin_width=0.25)
    pc16 = fit_power_curve(df_train.iloc[train_idx], n_sectors=16, ws_bin_width=0.25)

    df_tr = _add_power_curve_features(df_train.iloc[train_idx], pc8, pc16)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc8, pc16)
    feat_cols = feature_columns(df_tr)

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    print(f"\n  Features: {len(feat_cols)}")
    print(f"  Train: {len(X_tr)}, Val: {len(X_va)}")

    # === Multi-seed CV ===
    print(f"\n=== Multi-seed ensemble ({N_SEEDS} seeds) on Fold-5 ===")
    seeds = [42, 123, 456, 789, 2026]
    val_preds_all = []

    for i, seed in enumerate(seeds[:N_SEEDS]):
        booster = _train_single_seed(X_tr, y_tr, X_va, y_va, feat_cols, seed)
        preds = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0.0, CAPACITY_MW)
        nmae = normalized_mae(y_va, preds)
        val_preds_all.append(preds)
        print(f"  Seed {seed}: nMAE = {nmae:.4f} %  (best_iter={booster.best_iteration})")

    # Average ensemble.
    preds_avg = np.mean(val_preds_all, axis=0)
    preds_avg = np.clip(preds_avg, 0.0, CAPACITY_MW)
    nmae_avg = normalized_mae(y_va, preds_avg)
    print(f"\n  Multi-seed average nMAE: {nmae_avg:.4f} %")

    # === Isotonic calibration ===
    # Fit isotonic regression: ws_80m -> residual correction.
    ws_va = df_va["wind_speed_80m"].to_numpy()
    residuals = y_va - preds_avg
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ws_va, residuals)
    correction = iso.predict(ws_va)
    preds_iso = np.clip(preds_avg + correction, 0.0, CAPACITY_MW)
    nmae_iso = normalized_mae(y_va, preds_iso)
    print(f"  After isotonic correction: nMAE = {nmae_iso:.4f} %")

    # === Full-fit all seeds ===
    print(f"\n=== Full-fit ({N_SEEDS} seeds) ===")
    pc8_full = fit_power_curve(df_train, n_sectors=8, ws_bin_width=0.25)
    pc16_full = fit_power_curve(df_train, n_sectors=16, ws_bin_width=0.25)
    df_train_full = _add_power_curve_features(df_train, pc8_full, pc16_full)
    feat_cols = feature_columns(df_train_full)

    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

    boosters_full = []
    for seed in seeds[:N_SEEDS]:
        b = _train_full_seed(X_full, y_full, feat_cols, seed, n_rounds=2500)
        boosters_full.append(b)
    print(f"  Trained {len(boosters_full)} models")

    # === Predict Q1-2026 ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc8_full, pc16_full)
    missing = set(feat_cols) - set(df_valid_pred.columns)
    for col in missing:
        df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)

    preds_list = []
    for b in boosters_full:
        p = np.clip(b.predict(X_valid), 0.0, CAPACITY_MW)
        preds_list.append(p)

    preds_final = np.mean(preds_list, axis=0)
    preds_final = np.clip(preds_final, 0.0, CAPACITY_MW)

    # Restore original row order.
    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    preds_ordered = np.empty_like(preds_final)
    preds_ordered[order] = preds_final

    write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))

    # Stats.
    print(f"\n  Prediction stats: mean={preds_final.mean():.2f}, std={preds_final.std():.2f}")
    print(f"  P10={np.percentile(preds_final, 10):.2f}, P50={np.median(preds_final):.2f}, P90={np.percentile(preds_final, 90):.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
