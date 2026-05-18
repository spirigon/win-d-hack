"""Adversarial Validation: quantify train/test distribution shift.

Builds a binary classifier to distinguish training rows (label 0) from
Q1 2026 valid rows (label 1). If AUC is close to 0.5, distributions are
similar. If AUC >> 0.5, there's a genuine shift worth correcting.

Also identifies:
- Which features differ most (importance in the AV classifier)
- Which training rows look most like the Q1 2026 test set

Usage:
    python -m src.training.adversarial_validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_OUT = _ROOT / "data" / "processed" / "adversarial_oof.parquet"


def merge_era5(df, era5):
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")
    df["ws10_bias"] = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"] = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"] = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    era5_new = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def add_era5_rolling(df):
    df = df.copy()
    ws = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]
    for w in [3, 6, 12, 24]:
        roll = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = roll.mean()
        df[f"era5_ws100_roll_std_{w}h"] = roll.std().fillna(0)
    df["era5_ws100_diff1"] = ws.diff(1).fillna(0)
    df["era5_ws100_diff3"] = ws.diff(3).fillna(0)
    df["era5_ws100_diff6"] = ws.diff(6).fillna(0)
    df["era5_pressure_diff3"] = pres.diff(3).fillna(0)
    df["era5_pressure_diff6"] = pres.diff(6).fillna(0)
    df["era5_pressure_diff12"] = pres.diff(12).fillna(0)
    roll6 = ws.rolling(6, min_periods=1)
    df["era5_turb_intensity_6h"] = roll6.std().fillna(0) / (roll6.mean() + 1e-3)
    dir_sin = df["era5_dir100_sin"]
    dir_cos = df["era5_dir100_cos"]
    df["era5_dir_sin_diff1"] = dir_sin.diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = dir_cos.diff(1).fillna(0)
    return df


def main():
    set_global_seed(42)
    print("=" * 60)
    print("Adversarial Validation: Train vs Q1 2026")
    print("=" * 60)

    print("\nLoading data...")
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

    df_train_feat = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_feat = combined[combined["_split"] == "valid"].reset_index(drop=True)

    print(f"  Train rows: {len(df_train_feat)}")
    print(f"  Valid rows: {len(df_valid_feat)}")

    # Drop the target column and bookkeeping to get features.
    feat_cols = [c for c in feature_columns(df_train_feat) if c not in ("_split", "_submission_row")]
    # Remove features that trivially separate train/test by calendar/config,
    # not by genuine weather-distribution shift:
    # - calendar: year, month, day-of-year encodings, weekend, q1/winter flags
    # - operational: active_turbines / maintenance counts (valid set constrained to 3)
    # - hour_of_day is kept because Q1 2026 covers all hours just like training
    from src.data.schema import TURBINES_IN_MAINTENANCE_COL
    leaky_cols = {
        "year_index", "is_winter", "is_q1", "is_weekend",
        "month", "month_sin", "month_cos",
        "doy_sin", "doy_cos",
        "dow_sin", "dow_cos",
        TURBINES_IN_MAINTENANCE_COL,
        "active_turbines", "active_turbines_ratio", "maintenance_ratio",
        "rews_cube_x_active", "v_eff_cube_x_active",
        "p_curve_x_active", "p_curve_global_x_active",
        "era5_ws100_x_active", "era5_v_eff_cube_x_active", "era5_wpd_x_active",
        "era5_p_curve_x_active", "wpd_x_active",
    }
    feat_cols = [c for c in feat_cols if c not in leaky_cols]
    print(f"  Features: {len(feat_cols)} (excluded {len(leaky_cols)} calendar/config)")

    # Build AV dataset.
    X_train_av = df_train_feat[feat_cols].to_numpy(dtype=np.float32)
    X_valid_av = df_valid_feat[feat_cols].to_numpy(dtype=np.float32)
    X_av = np.vstack([X_train_av, X_valid_av])
    y_av = np.concatenate([
        np.zeros(len(X_train_av), dtype=np.int32),
        np.ones(len(X_valid_av), dtype=np.int32),
    ])
    print(f"  AV dataset: {len(X_av)} rows, positive ratio = {y_av.mean():.4f}")

    # --- Stratified K-fold AV classifier ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X_av), dtype=np.float32)
    fold_aucs = []
    fold_models = []

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    print("\n--- 5-fold AV training ---")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_av, y_av)):
        dtrain = lgb.Dataset(X_av[tr_idx], label=y_av[tr_idx], feature_name=feat_cols)
        dval = lgb.Dataset(X_av[va_idx], label=y_av[va_idx], feature_name=feat_cols)
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        va_preds = model.predict(X_av[va_idx], num_iteration=model.best_iteration)
        oof_probs[va_idx] = va_preds
        auc = roc_auc_score(y_av[va_idx], va_preds)
        fold_aucs.append(auc)
        fold_models.append(model)
        print(f"  Fold {fold+1}: AUC = {auc:.4f} (best_iter={model.best_iteration})")

    mean_auc = np.mean(fold_aucs)
    print(f"\nMean OOF AUC: {mean_auc:.4f}")
    if mean_auc < 0.55:
        print("  -> Distributions are very similar. AV reweighting unlikely to help.")
    elif mean_auc < 0.70:
        print("  -> Mild distribution shift. AV reweighting may give marginal gains.")
    elif mean_auc < 0.85:
        print("  -> Moderate distribution shift. AV reweighting should help.")
    else:
        print("  -> Strong distribution shift. AV reweighting likely valuable.")

    # --- Feature importance: which features separate train from Q1 2026? ---
    print("\n--- Top features separating train vs Q1 2026 ---")
    avg_imp = np.mean([m.feature_importance(importance_type="gain") for m in fold_models], axis=0)
    feat_imp = sorted(zip(feat_cols, avg_imp), key=lambda x: -x[1])
    for i, (name, imp) in enumerate(feat_imp[:25]):
        print(f"  {i+1:2d}. {name:40s} {imp:12.0f}")

    # --- Look at OOF prob distribution for training rows ---
    train_oof = oof_probs[: len(X_train_av)]
    valid_oof = oof_probs[len(X_train_av) :]
    print(f"\nTrain OOF probs: mean={train_oof.mean():.4f}, median={np.median(train_oof):.4f}, p90={np.quantile(train_oof, 0.9):.4f}, p99={np.quantile(train_oof, 0.99):.4f}")
    print(f"Valid OOF probs: mean={valid_oof.mean():.4f}, median={np.median(valid_oof):.4f}, p10={np.quantile(valid_oof, 0.1):.4f}")

    # Temporal pattern: what years/months are training rows that look most like Q1 2026?
    df_train_feat["av_prob"] = train_oof
    yearly = df_train_feat.groupby(df_train_feat[TIMESTAMP_COL].dt.year)["av_prob"].agg(["mean", "median", "count"])
    print("\nAV prob by train year (higher = more Q1-2026-like):")
    print(yearly.to_string())
    monthly = df_train_feat.groupby(df_train_feat[TIMESTAMP_COL].dt.month)["av_prob"].agg(["mean", "median", "count"])
    print("\nAV prob by train month (higher = more Q1-2026-like):")
    print(monthly.to_string())

    # Cross: year x month for the most recent years.
    recent = df_train_feat[df_train_feat[TIMESTAMP_COL].dt.year >= 2024].copy()
    recent["month"] = recent[TIMESTAMP_COL].dt.month
    recent["year"] = recent[TIMESTAMP_COL].dt.year
    recent_pivot = recent.groupby(["year", "month"])["av_prob"].mean().unstack()
    print("\nAV prob mean by year × month (recent years):")
    print(recent_pivot.to_string())

    # --- Save OOF for later use ---
    OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df = df_train_feat[[TIMESTAMP_COL]].copy()
    out_df["av_prob"] = train_oof
    out_df.to_parquet(OOF_OUT, index=False)
    print(f"\nTrain OOF probs saved: {OOF_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
