"""V5: LightGBM with sample weighting + feature selection + power curve.

Key ideas:
- Weight recent years more heavily (2024-2025 get 2x weight vs 2022-2023).
- Use only top-N features by importance from v4 to reduce noise.
- Try different power curve granularities (16 sectors vs 8).
- Experiment with Huber loss (less sensitive to maintenance outliers).

Usage:
    python -m src.training.train_v5
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
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.features.power_curve import fit_power_curve
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig, predict_lgbm, train_lgbm
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v0.5_lgbm_v5.csv"


def _add_power_curve_features(df: pd.DataFrame, pc_lookup, pc_lookup_16=None) -> pd.DataFrame:
    ws = df["wind_speed_80m"].to_numpy(dtype=float)
    dir_deg = df["wind_direction_80m"].to_numpy(dtype=float) * 1000.0

    df = df.copy()

    # 8-sector power curve.
    n_sectors = pc_lookup.n_sectors
    sector_width = 360.0 / n_sectors
    dir_sector = (dir_deg // sector_width).astype(int) % n_sectors
    df["p_theoretical"] = pc_lookup.predict(ws, dir_sector)
    df["p_theo_ratio"] = df["p_theoretical"] / CAPACITY_MW
    df["p_theo_x_active"] = df["p_theoretical"] * df["active_turbines"] / 26.0

    # 16-sector power curve (finer directional resolution).
    if pc_lookup_16 is not None:
        n16 = pc_lookup_16.n_sectors
        sw16 = 360.0 / n16
        ds16 = (dir_deg // sw16).astype(int) % n16
        df["p_theoretical_16s"] = pc_lookup_16.predict(ws, ds16)
        df["p_theo_16s_x_active"] = df["p_theoretical_16s"] * df["active_turbines"] / 26.0

    # 120 m lookup.
    ws120 = df["wind_speed_120m"].to_numpy(dtype=float)
    df["p_theoretical_120m"] = pc_lookup.predict(ws120, dir_sector)
    df["p_theo_diff_80_120"] = df["p_theoretical"] - df["p_theoretical_120m"]

    return df


def _compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Weight recent years more heavily.

    2024-2025 data is most representative of 2026 conditions.
    """
    year = df[TIMESTAMP_COL].dt.year
    weights = np.ones(len(df), dtype=np.float32)
    weights[year == 2022] = 0.5
    weights[year == 2023] = 0.75
    weights[year == 2024] = 1.0
    weights[year == 2025] = 1.5
    return weights


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

    # Configs to try.
    configs = {
        "A_tuned_weighted": LGBMConfig(
            num_leaves=434, min_data_in_leaf=124, learning_rate=0.0128,
            feature_fraction=0.644, bagging_fraction=0.747, bagging_freq=2,
            lambda_l1=0.00736, lambda_l2=0.00108,
            num_boost_round=4000, early_stopping_rounds=200, log_period=0,
        ),
        "B_huber": LGBMConfig(
            objective="huber", metric="mae",
            num_leaves=434, min_data_in_leaf=124, learning_rate=0.0128,
            feature_fraction=0.644, bagging_fraction=0.747, bagging_freq=2,
            lambda_l1=0.00736, lambda_l2=0.00108,
            num_boost_round=4000, early_stopping_rounds=200, log_period=0,
        ),
        "C_deeper": LGBMConfig(
            num_leaves=511, min_data_in_leaf=80, learning_rate=0.015,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            lambda_l1=0.01, lambda_l2=0.01,
            num_boost_round=4000, early_stopping_rounds=200, log_period=0,
        ),
    }

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    # Fit power curves.
    pc_8 = fit_power_curve(df_train.iloc[train_idx], n_sectors=8)
    pc_16 = fit_power_curve(df_train.iloc[train_idx], n_sectors=16)

    df_tr = _add_power_curve_features(df_train.iloc[train_idx], pc_8, pc_16)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_8, pc_16)
    feat_cols = feature_columns(df_tr)

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    # Sample weights for training.
    w_tr = _compute_sample_weights(df_train.iloc[train_idx])

    print(f"\n  Features: {len(feat_cols)}")
    print(f"  Train: {len(X_tr)}, Val: {len(X_va)}")

    best_nmae = 999.0
    best_name = ""
    best_config = None

    for name, config in configs.items():
        # Train with sample weights.
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)

        callbacks = [lgb.early_stopping(config.early_stopping_rounds, verbose=False)]
        booster = lgb.train(
            config.to_params(),
            dtrain,
            num_boost_round=config.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        preds = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0.0, CAPACITY_MW)
        nmae = normalized_mae(y_va, preds)
        print(f"  {name}: nMAE = {nmae:.4f} %  (best_iter={booster.best_iteration})")

        if nmae < best_nmae:
            best_nmae = nmae
            best_name = name
            best_config = config

    # Also try without sample weights for comparison.
    print("\n  --- Without sample weights ---")
    config_no_w = configs["A_tuned_weighted"]
    booster_nw = train_lgbm(X_tr, y_tr, X_va, y_va, feature_names=feat_cols, config=config_no_w)
    preds_nw = predict_lgbm(booster_nw, X_va)
    nmae_nw = normalized_mae(y_va, preds_nw)
    print(f"  A_tuned_no_weight: nMAE = {nmae_nw:.4f} %")

    if nmae_nw < best_nmae:
        best_nmae = nmae_nw
        best_name = "A_tuned_no_weight"

    print(f"\n  Best: {best_name} with nMAE = {best_nmae:.4f} %")

    # === Full-fit with best approach and generate submission ===
    print("\n=== Full-fit with best config ===")
    pc_full_8 = fit_power_curve(df_train, n_sectors=8)
    pc_full_16 = fit_power_curve(df_train, n_sectors=16)
    df_train_full = _add_power_curve_features(df_train, pc_full_8, pc_full_16)
    feat_cols = feature_columns(df_train_full)

    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)
    w_full = _compute_sample_weights(df_train)

    # Use best config for full fit.
    use_config = configs.get(best_name, configs["A_tuned_weighted"])
    dtrain_full = lgb.Dataset(X_full, label=y_full, weight=w_full, feature_name=feat_cols, free_raw_data=False)
    booster_full = lgb.train(
        use_config.to_params(),
        dtrain_full,
        num_boost_round=2500,
    )

    model_path = MODEL_DIR / "lgbm_v5.txt"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster_full.save_model(str(model_path))
    print(f"  Model saved to {model_path}")

    # === Predict ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_full_8, pc_full_16)
    missing = set(feat_cols) - set(df_valid_pred.columns)
    for col in missing:
        df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)
    preds = np.clip(booster_full.predict(X_valid), 0.0, CAPACITY_MW)

    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    preds_ordered = np.empty_like(preds)
    preds_ordered[order] = preds

    write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))
    print("\nDone.")


if __name__ == "__main__":
    main()
