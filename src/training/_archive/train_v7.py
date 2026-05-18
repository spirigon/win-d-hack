"""V7: Physics-aware features + isotonic power curve + post-processing.

Incorporates all recommendations from the research guide:
1. REWS + v_eff + Hellmann shear + air density (physics-informed features).
2. Per-sector isotonic power curve (smoother than binned).
3. Cut-in (v_eff < 3.5) and cut-out (v_eff > 25) post-processing.
4. Multi-seed LightGBM ensemble (5 seeds).
5. Full 5-fold CV for calibration.

Usage:
    python -m src.training.train_v7
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
from src.features.physics import IsotonicPowerCurve, SectorIsotonicPowerCurve, fit_sector_isotonic
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v0.7_physics_ensemble.csv"

# Cut-in / cut-out thresholds (Siemens Gamesa typical, from research doc).
V_CUT_IN = 3.0  # m/s at hub height — below this, turbine doesn't produce.
V_CUT_OUT = 25.0  # m/s — above this, turbine shuts down.

N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2026]


def _add_power_curve_features(
    df: pd.DataFrame,
    pc_sector: SectorIsotonicPowerCurve,
    pc_global: IsotonicPowerCurve,
) -> pd.DataFrame:
    """Add isotonic power curve predictions as features."""
    df = df.copy()
    v_eff = df["v_eff"].to_numpy()
    dir_deg = (df["wind_direction_120m"] * 1000.0).to_numpy()

    # Per-sector curve on v_eff.
    df["p_curve_sector"] = pc_sector.predict(v_eff, dir_deg)

    # Global curve on v_eff.
    df["p_curve_global"] = pc_global.predict(v_eff)

    # Also on REWS.
    df["p_curve_rews"] = pc_global.predict(df["rews"].to_numpy())

    # Capacity-scaled.
    df["p_curve_x_active"] = df["p_curve_sector"] * df["active_turbines_ratio"]
    df["p_curve_global_x_active"] = df["p_curve_global"] * df["active_turbines_ratio"]

    # Ratio features.
    df["p_curve_ratio"] = df["p_curve_sector"] / CAPACITY_MW

    # Difference between sector and global (captures anomalous sectors).
    df["p_curve_sector_minus_global"] = df["p_curve_sector"] - df["p_curve_global"]

    return df


def _post_process(
    preds: np.ndarray,
    v_eff: np.ndarray,
    active_turbines_ratio: np.ndarray,
    apply_capacity: bool = False,
) -> np.ndarray:
    """Apply physics-informed post-processing.

    1. Cut-in: v_eff < 3 m/s -> 0 MW (turbine idle).
    2. Cut-out: v_eff > 25 m/s -> predictions damped.
    3. Clip to [0, 90.09].
    4. (Optional) Scale by active turbines ratio.
    """
    preds = np.asarray(preds, dtype=float)
    # Cut-in.
    preds = np.where(v_eff < V_CUT_IN, 0.0, preds)
    # Cut-out (soft — clip rather than zero to avoid over-penalizing).
    preds = np.where(v_eff > V_CUT_OUT, np.minimum(preds, 5.0), preds)
    # Optional capacity scaling.
    if apply_capacity:
        preds = preds * active_turbines_ratio
    return np.clip(preds, 0.0, CAPACITY_MW)


def _train_single(
    X_tr, y_tr, X_va, y_va, feat_cols, seed, config_base
):
    cfg_kwargs = {**config_base.__dict__, "seed": seed}
    config = LGBMConfig(**cfg_kwargs)
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


def _train_full(X, y, feat_cols, seed, n_rounds, config_base):
    cfg_kwargs = {**config_base.__dict__, "seed": seed}
    config = LGBMConfig(**cfg_kwargs)
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
    booster = lgb.train(config.to_params(), dtrain, num_boost_round=n_rounds)
    return booster


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    print(f"  Train: {len(df_train)} rows, Valid: {len(df_valid)} rows")

    print("Building features (physics-aware)...")
    df_train = build_features(df_train, sort_by_time=False)
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)

    # Tuned LGBM config.
    config_base = LGBMConfig(
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
    )

    # === Full 5-fold CV for honest evaluation ===
    print("\n=== 5-fold Walk-forward CV ===")
    folds = default_folds()
    fold_results: list[float] = []

    for fold in folds:
        train_idx, val_idx = split_indices(df_train, fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        # Fit power curves on this fold's training data only.
        pc_sector = fit_sector_isotonic(df_train.iloc[train_idx], n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(
            df_train.iloc[train_idx]["v_eff"], df_train.iloc[train_idx][TARGET_COL]
        )

        df_tr = _add_power_curve_features(df_train.iloc[train_idx], pc_sector, pc_global)
        df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
        feat_cols = feature_columns(df_tr)

        X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
        y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

        # Train single-seed model for CV.
        booster = _train_single(X_tr, y_tr, X_va, y_va, feat_cols, 42, config_base)
        preds = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW)

        # Post-processing.
        v_eff_va = df_va["v_eff"].to_numpy()
        active_ratio_va = df_va["active_turbines_ratio"].to_numpy()
        preds_pp = _post_process(preds, v_eff_va, active_ratio_va, apply_capacity=False)

        nmae_raw = normalized_mae(y_va, preds)
        nmae_pp = normalized_mae(y_va, preds_pp)
        fold_results.append(nmae_pp)
        print(f"  {fold.name}: raw nMAE = {nmae_raw:.4f} %, after post-proc = {nmae_pp:.4f} %  (best_iter={booster.best_iteration})")

    print(f"\n  Mean 5-fold nMAE: {np.mean(fold_results):.4f} %")
    print(f"  Fold-5 (Q1 surrogate): {fold_results[-1]:.4f} %")

    # === Fold-5 multi-seed ensemble for final evaluation ===
    print(f"\n=== Fold-5 multi-seed ensemble ({N_SEEDS} seeds) ===")
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    pc_sector = fit_sector_isotonic(df_train.iloc[train_idx], n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(
        df_train.iloc[train_idx]["v_eff"], df_train.iloc[train_idx][TARGET_COL]
    )

    df_tr = _add_power_curve_features(df_train.iloc[train_idx], pc_sector, pc_global)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    feat_cols = feature_columns(df_tr)

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()
    active_ratio_va = df_va["active_turbines_ratio"].to_numpy()

    print(f"  Features: {len(feat_cols)}")
    preds_all = []
    for seed in SEEDS[:N_SEEDS]:
        booster = _train_single(X_tr, y_tr, X_va, y_va, feat_cols, seed, config_base)
        p = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW)
        p_pp = _post_process(p, v_eff_va, active_ratio_va, apply_capacity=False)
        nmae_seed = normalized_mae(y_va, p_pp)
        preds_all.append(p_pp)
        print(f"  Seed {seed}: Fold-5 nMAE = {nmae_seed:.4f} %")

    preds_ens = np.mean(preds_all, axis=0)
    preds_ens = np.clip(preds_ens, 0, CAPACITY_MW)
    nmae_ens = normalized_mae(y_va, preds_ens)
    print(f"  Multi-seed ensemble nMAE: {nmae_ens:.4f} %")

    # === Full-fit on all data ===
    print(f"\n=== Full-fit ({N_SEEDS} seeds) ===")
    pc_sector_full = fit_sector_isotonic(df_train, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train["v_eff"], df_train[TARGET_COL])

    df_train_full = _add_power_curve_features(df_train, pc_sector_full, pc_global_full)
    feat_cols = feature_columns(df_train_full)
    print(f"  Features: {len(feat_cols)}")

    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

    boosters_full = []
    for seed in SEEDS[:N_SEEDS]:
        b = _train_full(X_full, y_full, feat_cols, seed, 2500, config_base)
        boosters_full.append(b)
    print(f"  Trained {len(boosters_full)} models")

    # === Predict Q1-2026 ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
    missing = set(feat_cols) - set(df_valid_pred.columns)
    for col in missing:
        df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)
    v_eff_valid = df_valid_pred["v_eff"].to_numpy()
    active_ratio_valid = df_valid_pred["active_turbines_ratio"].to_numpy()

    valid_preds_list = []
    for b in boosters_full:
        p = np.clip(b.predict(X_valid), 0, CAPACITY_MW)
        p_pp = _post_process(p, v_eff_valid, active_ratio_valid, apply_capacity=False)
        valid_preds_list.append(p_pp)

    preds_final = np.mean(valid_preds_list, axis=0)
    preds_final = np.clip(preds_final, 0, CAPACITY_MW)

    # Save feature importance from seed-42 model.
    imp = boosters_full[0].feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    print("\n=== Top 20 features by importance (gain, seed 42) ===")
    for name, score in feat_imp[:20]:
        print(f"  {name:40s} {score:12.1f}")

    # Restore original row order.
    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    preds_ordered = np.empty_like(preds_final)
    preds_ordered[order] = preds_final

    write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))

    # Stats.
    print(f"\n  Prediction stats: mean={preds_final.mean():.2f}, std={preds_final.std():.2f}")
    print(f"  P10={np.percentile(preds_final, 10):.2f}, P50={np.median(preds_final):.2f}, P90={np.percentile(preds_final, 90):.2f}")

    # Save models.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(boosters_full):
        b.save_model(str(MODEL_DIR / f"lgbm_v7_seed{SEEDS[i]}.txt"))
    print(f"\n  Models saved to {MODEL_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
