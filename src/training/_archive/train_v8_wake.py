"""V8: v7 refined + wake correction features.

Per-direction × wind-speed wake correction lookup learned from clean
training data. Applied to datasheet predictions, gives "actual-turbine-farm"
estimates that already bake in wake physics.

Usage:
    python -m src.training.train_v8_wake
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
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v8.0_wake.csv"

SEEDS = [42, 123, 456, 789, 2026, 3141, 1618, 2718, 7777, 12345]


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
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    era5_new = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def _train_single(X_tr, y_tr, X_va, y_va, feat_cols, seed, config):
    cfg = LGBMConfig(**{**config.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(
        cfg.to_params(), dtrain, num_boost_round=4000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )


def _train_full(X, y, feat_cols, seed, n_rounds, config):
    cfg = LGBMConfig(**{**config.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(cfg.to_params(), dtrain, num_boost_round=n_rounds)


def main():
    set_global_seed(42)
    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    df_train = build_features(df_train, sort_by_time=False)
    df_train = merge_era5(df_train, era5)
    df_train = add_datasheet_power_features(df_train)
    df_train = add_extra_features(df_train)

    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)
    df_valid_sorted = merge_era5(df_valid_sorted, era5)
    df_valid_sorted = add_datasheet_power_features(df_valid_sorted)
    df_valid_sorted = add_extra_features(df_valid_sorted)

    config = LGBMConfig(
        num_leaves=453, min_data_in_leaf=121, learning_rate=0.01555,
        feature_fraction=0.695, bagging_fraction=0.509, bagging_freq=5,
        lambda_l1=0.00294, lambda_l2=0.342,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    # Physics power curves.
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    # Wake correction lookup from clean training rows.
    wake_lookup = fit_wake_lookup(fit_data, n_sectors=16)

    # Apply all features.
    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake_lookup)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake_lookup)
    feat_cols_all = [c for c in feature_columns(df_tr) if c != "_is_impossible"]
    print(f"  Total features: {len(feat_cols_all)}")

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    probe = _train_single(X_tr_all, y_tr, X_va_all, y_va, feat_cols_all, 42, config)
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])

    print("\n  Wake features in top 30:")
    for rank, (name, score) in enumerate(feat_imp[:30], 1):
        if "wake" in name:
            print(f"    rank {rank:2d}: {name:40s} {score:10.1f}")

    # K ablation.
    print("\n  === K ablation ===")
    ks = [50, 60, 70, 80, 90, 100, len(feat_cols_all)]
    best_nmae = 999
    best_k = 60
    for k in ks:
        top_k = [n for n, _ in feat_imp[:k]]
        X_tr_k = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va_k = df_va[top_k].to_numpy(dtype=np.float32)
        preds_all = []
        for seed in SEEDS[:5]:
            b = _train_single(X_tr_k, y_tr, X_va_k, y_va, top_k, seed, config)
            p = np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW)
            preds_all.append(p)
        ens = np.mean(preds_all, axis=0)
        nmae = normalized_mae(y_va, ens)
        print(f"    K={k:3d}: {nmae:.4f} %")
        if nmae < best_nmae:
            best_nmae = nmae
            best_k = k

    print(f"\n  Best K={best_k} nMAE={best_nmae:.4f}")

    # Full 10-seed ensemble at best K.
    print(f"\n  === Final K={best_k}, 10 seeds ===")
    top_k_final = [n for n, _ in feat_imp[:best_k]]
    X_tr_k = df_tr[top_k_final].to_numpy(dtype=np.float32)
    X_va_k = df_va[top_k_final].to_numpy(dtype=np.float32)

    preds_all = []
    best_iters = []
    for seed in SEEDS:
        b = _train_single(X_tr_k, y_tr, X_va_k, y_va, top_k_final, seed, config)
        p = np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        nmae = normalized_mae(y_va, p)
        preds_all.append(p)
        best_iters.append(b.best_iteration)
        print(f"    Seed {seed}: {nmae:.4f}%  iter={b.best_iteration}")
    ens = np.mean(preds_all, axis=0)
    nmae_ens = normalized_mae(y_va, ens)
    print(f"\n    Ensemble Fold-5: {nmae_ens:.4f}%")

    # Full-fit.
    print("\n  === Full-fit ===")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    wake_full = fit_wake_lookup(df_train_clean, n_sectors=16)

    df_train_full = _add_power_curve_features(df_train, pc_sector_full, pc_global_full)
    df_train_full = add_wake_features(df_train_full, wake_full)

    X_full = df_train_full[top_k_final].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)
    n_rounds = max(int(np.median(best_iters) * 1.2), 1000)
    boosters = [_train_full(X_full, y_full, top_k_final, s, n_rounds, config) for s in SEEDS]

    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
    df_valid_pred = add_wake_features(df_valid_pred, wake_full)
    for col in set(top_k_final) - set(df_valid_pred.columns):
        df_valid_pred[col] = 0.0
    X_valid = df_valid_pred[top_k_final].to_numpy(dtype=np.float32)
    valid_preds = [np.clip(b.predict(X_valid), 0, CAPACITY_MW) for b in boosters]
    preds_final = np.clip(np.mean(valid_preds, axis=0), 0, CAPACITY_MW)

    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_final)
    po[order] = preds_final
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))

    # Blend with v7.0.
    v70 = pd.read_csv(_ROOT / "submissions" / "archive" / "v7.0_refined.csv", header=None)[0].to_numpy()
    blend = np.clip((po + v70) / 2.0, 0, CAPACITY_MW)
    write_submission(blend, _ROOT / "submissions" / "archive" / "v8.1_v7_blend.csv", expected_rows=len(df_valid))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(boosters):
        b.save_model(str(MODEL_DIR / f"lgbm_v8_seed{SEEDS[i]}.txt"))
    print("\nDone.")


if __name__ == "__main__":
    main()
