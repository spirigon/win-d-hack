"""V10: Clean power curve + bias correction + multi-seed ensemble.

Learning from v7/v8/v9 experiments:
- Outlier-filtered power curve fit helped (8.82% -> 8.74% Fold-5).
- Curtailment features HURT Fold-5 (removed).
- There is a systematic under-prediction bias in the 10-14 m/s band.

This version:
1. Outlier filter applied ONLY to power curve fit.
2. Drop curtailment features.
3. Learn bias correction from within-CV OOF, apply at inference.
4. 5-seed LGBM ensemble.

Usage:
    python -m src.training.train_v10
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
from src.features.bias_correction import BiasCalibrator
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v1.0_clean_bias.csv"

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


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    n_impossible = int(impossible.sum())
    print(f"  Outliers (excluded from power curve fit only): {n_impossible}")

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

    # === 5-fold CV: collect OOF predictions for bias calibration ===
    print("\n=== 5-fold Walk-forward CV ===")
    folds = default_folds()
    fold_results = []
    oof_records = []  # for training the bias calibrator

    for fold in folds:
        train_idx, val_idx = split_indices(df_train, fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        fold_train = df_train.iloc[train_idx]
        fit_data = fold_train[~fold_train["_is_impossible"]]
        pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

        df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
        df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
        feat_cols = [c for c in feature_columns(df_tr) if c != "_is_impossible"]

        X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
        y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        ws_va = df_va["wind_speed_120m"].to_numpy()
        v_eff_va = df_va["v_eff"].to_numpy()

        booster = _train_single(X_tr, y_tr, X_va, y_va, feat_cols, 42, config_base)
        preds = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW)
        preds_pp = _post_process(preds, v_eff_va)
        nmae = normalized_mae(y_va, preds_pp)
        fold_results.append(nmae)

        # Record OOF (for bias calibration).
        for i in range(len(y_va)):
            oof_records.append({
                "fold": fold.name,
                "ws": ws_va[i],
                "y_true": float(y_va[i]),
                "y_pred": float(preds_pp[i]),
            })
        print(f"  {fold.name}: nMAE = {nmae:.4f} %  (best_iter={booster.best_iteration})")

    print(f"\n  Mean 5-fold nMAE: {np.mean(fold_results):.4f} %")

    # === Fit bias calibrator on OOF (ALL folds) ===
    print("\n=== Bias calibrator on OOF ===")
    oof_df = pd.DataFrame(oof_records)
    calibrator = BiasCalibrator(n_knots=20)
    calibrator.fit(
        oof_df["ws"].to_numpy(),
        oof_df["y_true"].to_numpy(),
        oof_df["y_pred"].to_numpy(),
    )

    # Check improvement on Fold-5 alone.
    f5_df = oof_df[oof_df["fold"] == "fold5_2025Q1"]
    corrected = np.clip(
        calibrator.apply(f5_df["y_pred"].to_numpy(), f5_df["ws"].to_numpy()),
        0.0, CAPACITY_MW,
    )
    nmae_raw = normalized_mae(f5_df["y_true"], f5_df["y_pred"])
    nmae_cal = normalized_mae(f5_df["y_true"], corrected)
    print(f"  Fold-5 raw:       {nmae_raw:.4f} %")
    print(f"  Fold-5 calibrated: {nmae_cal:.4f} %  (delta {nmae_cal - nmae_raw:+.4f})")

    # Print knots for inspection.
    print("  Bias calibration knots (ws -> bias):")
    for ws, bias in zip(calibrator._ws_knots, calibrator._bias_knots, strict=True):
        print(f"    ws={ws:5.2f} -> bias={bias:+.3f}")

    # === Fold-5 multi-seed ensemble with calibration ===
    print(f"\n=== Fold-5 multi-seed ensemble + calibration ===")
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    feat_cols = [c for c in feature_columns(df_tr) if c != "_is_impossible"]

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()
    ws_va = df_va["wind_speed_120m"].to_numpy()

    preds_all = []
    preds_all_cal = []
    best_iters = []
    for seed in SEEDS:
        booster = _train_single(X_tr, y_tr, X_va, y_va, feat_cols, seed, config_base)
        p = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW)
        p_pp = _post_process(p, v_eff_va)
        # Also apply bias calibration — but use a per-fold calibrator fit on OTHER folds to avoid leak.
        # For Fold-5 eval, use a calibrator fit on folds 1-4 only.
        preds_all.append(p_pp)
        best_iters.append(booster.best_iteration)
        print(f"  Seed {seed}: raw nMAE = {normalized_mae(y_va, p_pp):.4f} %  (best_iter={booster.best_iteration})")

    preds_ens = np.mean(preds_all, axis=0)
    nmae_ens = normalized_mae(y_va, preds_ens)
    print(f"\n  Multi-seed ensemble RAW:        {nmae_ens:.4f} %")

    # Calibrator fit on Folds 1-4 only (to avoid Fold-5 leakage).
    cal_14 = BiasCalibrator(n_knots=20)
    f14 = oof_df[oof_df["fold"] != "fold5_2025Q1"]
    cal_14.fit(f14["ws"].to_numpy(), f14["y_true"].to_numpy(), f14["y_pred"].to_numpy())
    preds_ens_cal = np.clip(cal_14.apply(preds_ens, ws_va), 0, CAPACITY_MW)
    # Re-apply cut-in / cut-out after calibration.
    preds_ens_cal = _post_process(preds_ens_cal, v_eff_va)
    nmae_ens_cal = normalized_mae(y_va, preds_ens_cal)
    print(f"  Multi-seed ensemble CALIBRATED: {nmae_ens_cal:.4f} %")

    # === Full-fit ===
    print(f"\n=== Full-fit ===")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    df_train_full = _add_power_curve_features(df_train, pc_sector_full, pc_global_full)
    feat_cols = [c for c in feature_columns(df_train_full) if c != "_is_impossible"]

    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

    n_rounds = max(int(np.median(best_iters) * 1.2), 1000)
    print(f"  n_rounds per seed: {n_rounds}")
    boosters_full = [_train_full(X_full, y_full, feat_cols, s, n_rounds, config_base) for s in SEEDS]

    # === Predict Q1-2026 (with full-data calibrator) ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
    for col in set(feat_cols) - set(df_valid_pred.columns):
        df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)
    v_eff_valid = df_valid_pred["v_eff"].to_numpy()
    ws_valid = df_valid_pred["wind_speed_120m"].to_numpy()

    valid_preds_list = []
    for b in boosters_full:
        p = np.clip(b.predict(X_valid), 0, CAPACITY_MW)
        p_pp = _post_process(p, v_eff_valid)
        valid_preds_list.append(p_pp)

    preds_ens = np.mean(valid_preds_list, axis=0)

    # Use the all-folds calibrator (calibrator) for the submission since it's the
    # most information-rich estimate of the bias.
    preds_final = calibrator.apply(preds_ens, ws_valid)
    preds_final = _post_process(preds_final, v_eff_valid)
    preds_final = np.clip(preds_final, 0, CAPACITY_MW)

    # Also save non-calibrated version to compare.
    preds_raw = _post_process(preds_ens, v_eff_valid)
    preds_raw = np.clip(preds_raw, 0, CAPACITY_MW)

    # Restore row order.
    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    preds_final_ordered = np.empty_like(preds_final)
    preds_final_ordered[order] = preds_final
    preds_raw_ordered = np.empty_like(preds_raw)
    preds_raw_ordered[order] = preds_raw

    write_submission(preds_final_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))
    raw_path = _ROOT / "submissions" / "archive" / "v1.0_clean_nocal.csv"
    write_submission(preds_raw_ordered, raw_path, expected_rows=len(df_valid))

    print(f"\n  Calibrated stats: mean={preds_final.mean():.2f}, std={preds_final.std():.2f}")
    print(f"  Raw stats:        mean={preds_raw.mean():.2f}, std={preds_raw.std():.2f}")
    print(f"  Diff (calibrated - raw):  mean={np.mean(preds_final - preds_raw):.2f}, max abs diff={np.max(np.abs(preds_final - preds_raw)):.2f}")

    # Save.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(boosters_full):
        b.save_model(str(MODEL_DIR / f"lgbm_v10_seed{SEEDS[i]}.txt"))
    print(f"\n  Models saved to {MODEL_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
