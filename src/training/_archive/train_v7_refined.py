"""V7: v6 datasheet + extra features + soft cut-in + LGBM structural diversity.

Adds:
- Extra shear pairs + wind vector components + wind regime (low-risk)
- Soft sigmoid cut-in instead of hard zero
- 3 structurally different LGBM configs (default, deep, wide) × 5 seeds each

Usage:
    python -m src.training.train_v7_refined
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
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v7.0_refined.csv"

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


def _soft_cutin(preds, v_eff, center=2.6, width=1.2):
    """Sigmoid ramp: smoothly damps predictions below cut-in.

    Sigmoid = 1 / (1 + exp(-(v_eff - center)/width*slope)).
    At v_eff = center - width, sigmoid ≈ 0.12.
    At v_eff = center, sigmoid = 0.5.
    At v_eff = center + width, sigmoid ≈ 0.88.
    """
    # Scale factor: how quickly the sigmoid transitions.
    slope = 4.0 / width
    factor = 1.0 / (1.0 + np.exp(-(v_eff - center) * slope))
    return preds * factor


def _post_process(preds, v_eff, soft=True):
    preds = np.asarray(preds, dtype=float)
    if soft:
        preds = _soft_cutin(preds, v_eff)
    else:
        preds = np.where(v_eff < 3.0, 0.0, preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


# Three structurally diverse LGBM configs.
CONFIGS = {
    "tuned": LGBMConfig(
        num_leaves=453, min_data_in_leaf=121, learning_rate=0.01555,
        feature_fraction=0.695, bagging_fraction=0.509, bagging_freq=5,
        lambda_l1=0.00294, lambda_l2=0.342,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    ),
    "deep": LGBMConfig(
        num_leaves=127, min_data_in_leaf=50, learning_rate=0.02,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
        lambda_l1=0.1, lambda_l2=0.5,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    ),
    "wide": LGBMConfig(
        num_leaves=511, min_data_in_leaf=200, learning_rate=0.01,
        feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=7,
        lambda_l1=0.001, lambda_l2=1.0,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    ),
}


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

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    feat_cols_all = [c for c in feature_columns(df_tr) if c != "_is_impossible"]
    print(f"  Total features: {len(feat_cols_all)}")

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()

    # Probe with tuned config for importance.
    probe = _train_single(X_tr_all, y_tr, X_va_all, y_va, feat_cols_all, 42, CONFIGS["tuned"])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])

    print("\n  Extra features in top 30:")
    for rank, (name, score) in enumerate(feat_imp[:30], 1):
        if any(x in name for x in ("shear_10_120", "shear_10_180", "shear_80_180", "ws120_u", "ws120_v", "ws80_u", "ws80_v", "regime", "era5_ws100_u", "era5_ws100_v")):
            print(f"    rank {rank:2d}: {name:40s} {score:10.1f}")

    # Ablation K with hard cut-in.
    print("\n  === K ablation (hard cut-in, tuned config, 5 seeds) ===")
    ks = [60, 70, 80, 90, 100, len(feat_cols_all)]
    best_nmae_hard = 999
    best_k_hard = 0
    for k in ks:
        top_k = [n for n, _ in feat_imp[:k]]
        X_tr_k = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va_k = df_va[top_k].to_numpy(dtype=np.float32)
        preds_all = []
        for seed in SEEDS:
            b = _train_single(X_tr_k, y_tr, X_va_k, y_va, top_k, seed, CONFIGS["tuned"])
            p = _post_process(np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va, soft=False)
            preds_all.append(p)
        ens = np.mean(preds_all, axis=0)
        nmae = normalized_mae(y_va, ens)
        print(f"    K={k:3d} (hard cut-in):  {nmae:.4f} %")
        if nmae < best_nmae_hard:
            best_nmae_hard = nmae
            best_k_hard = k

    # Same with soft cut-in at the best K.
    print(f"\n  === Soft cut-in at best K={best_k_hard} ===")
    top_k_best = [n for n, _ in feat_imp[:best_k_hard]]
    X_tr_k = df_tr[top_k_best].to_numpy(dtype=np.float32)
    X_va_k = df_va[top_k_best].to_numpy(dtype=np.float32)
    preds_all_soft = []
    for seed in SEEDS:
        b = _train_single(X_tr_k, y_tr, X_va_k, y_va, top_k_best, seed, CONFIGS["tuned"])
        p = _post_process(np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va, soft=True)
        preds_all_soft.append(p)
    ens_soft = np.mean(preds_all_soft, axis=0)
    nmae_soft = normalized_mae(y_va, ens_soft)
    print(f"    Soft cut-in: {nmae_soft:.4f} %  (delta {nmae_soft - best_nmae_hard:+.4f})")

    # Also try NO cut-in (let model decide).
    preds_all_none = []
    for seed in SEEDS:
        b = _train_single(X_tr_k, y_tr, X_va_k, y_va, top_k_best, seed, CONFIGS["tuned"])
        p = np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        preds_all_none.append(p)
    ens_none = np.mean(preds_all_none, axis=0)
    nmae_none = normalized_mae(y_va, ens_none)
    print(f"    No cut-in:   {nmae_none:.4f} %  (delta {nmae_none - best_nmae_hard:+.4f})")

    best_cutin = "hard" if best_nmae_hard <= min(nmae_soft, nmae_none) else ("soft" if nmae_soft <= nmae_none else "none")
    best_overall = min(best_nmae_hard, nmae_soft, nmae_none)
    print(f"\n  Best: {best_cutin} cut-in @ K={best_k_hard}, nMAE={best_overall:.4f} %")

    # Structural diversity: train with tuned / deep / wide configs at best K.
    print(f"\n  === Structural diversity (3 configs × {len(SEEDS)} seeds) ===")
    config_preds = {}
    config_iters = {}
    for name, cfg in CONFIGS.items():
        preds_cfg = []
        iters_cfg = []
        for seed in SEEDS:
            b = _train_single(X_tr_k, y_tr, X_va_k, y_va, top_k_best, seed, cfg)
            p = _post_process(np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va, soft=(best_cutin == "soft"))
            if best_cutin == "none":
                p = np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW)
            preds_cfg.append(p)
            iters_cfg.append(b.best_iteration)
        config_preds[name] = np.mean(preds_cfg, axis=0)
        config_iters[name] = iters_cfg
        print(f"    Config {name}: Fold-5 = {normalized_mae(y_va, config_preds[name]):.4f} %")

    struct_ens = np.mean(list(config_preds.values()), axis=0)
    nmae_struct = normalized_mae(y_va, struct_ens)
    print(f"\n  Structural ensemble Fold-5: {nmae_struct:.4f} %")

    # Pick the best approach for final submission.
    if nmae_struct < best_overall:
        print(f"  Structural ensemble wins (+{best_overall - nmae_struct:.4f} pp)")
        use_struct = True
    else:
        print(f"  Single-config ensemble wins")
        use_struct = False

    # === Full-fit ===
    print("\n  === Full-fit ===")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    df_train_full = _add_power_curve_features(df_train, pc_sector_full, pc_global_full)

    X_full = df_train_full[top_k_best].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

    if use_struct:
        valid_preds_list = []
        for name, cfg in CONFIGS.items():
            n_rounds = max(int(np.median(config_iters[name]) * 1.2), 1000)
            for seed in SEEDS:
                b = _train_full(X_full, y_full, top_k_best, seed, n_rounds, cfg)
                # Predict later; save booster + config.
                df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
                for col in set(top_k_best) - set(df_valid_pred.columns):
                    df_valid_pred[col] = 0.0
                X_valid = df_valid_pred[top_k_best].to_numpy(dtype=np.float32)
                p = np.clip(b.predict(X_valid), 0, CAPACITY_MW)
                valid_preds_list.append(p)
    else:
        n_rounds = max(int(np.median(config_iters["tuned"]) * 1.2), 1000)
        boosters = [_train_full(X_full, y_full, top_k_best, s, n_rounds, CONFIGS["tuned"]) for s in SEEDS]
        df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
        for col in set(top_k_best) - set(df_valid_pred.columns):
            df_valid_pred[col] = 0.0
        X_valid = df_valid_pred[top_k_best].to_numpy(dtype=np.float32)
        valid_preds_list = [np.clip(b.predict(X_valid), 0, CAPACITY_MW) for b in boosters]

    v_eff_valid = df_valid_pred["v_eff"].to_numpy()
    preds_final_raw = np.mean(valid_preds_list, axis=0)
    if best_cutin == "hard":
        preds_final = _post_process(preds_final_raw, v_eff_valid, soft=False)
    elif best_cutin == "soft":
        preds_final = _post_process(preds_final_raw, v_eff_valid, soft=True)
    else:
        preds_final = np.clip(preds_final_raw, 0, CAPACITY_MW)

    preds_final = np.clip(preds_final, 0, CAPACITY_MW)
    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_final)
    po[order] = preds_final
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))

    # Blend with v6.0 for insurance.
    v60 = pd.read_csv(_ROOT / "submissions" / "archive" / "v6.0_datasheet.csv", header=None)[0].to_numpy()
    blend = np.clip((po + v60) / 2.0, 0, CAPACITY_MW)
    write_submission(blend, _ROOT / "submissions" / "archive" / "v7.1_v6_blend.csv", expected_rows=len(df_valid))

    print("\nDone.")


if __name__ == "__main__":
    main()
