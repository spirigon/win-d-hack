"""Quick multi-fold check: K=70 vs K=80 on Folds 1 and 5 (winter folds)."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TOTAL_TURBINES
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"

SEEDS_SPEC = [42, 123, 456, 789, 2026]
TURBINE_RATED_MW = 3.465

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)


def _add_pc(df, pc_sector, pc_global):
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


def to_cf(y_mw, active_turbines):
    p_available = active_turbines * TURBINE_RATED_MW
    return y_mw / np.maximum(p_available, 1e-3)


def from_cf(cf, active_turbines):
    p_available = active_turbines * TURBINE_RATED_MW
    return cf * p_available


def eval_fold(df_train, fold, feat_imp, K_val):
    """Evaluate a single fold with given K."""
    train_idx, val_idx = split_indices(df_train, fold)
    fold_train = df_train.iloc[train_idx]
    fold_val = df_train.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    top_k = [n for n, _ in feat_imp[:K_val]]
    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)

    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(y_va_mw, active_va)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    # 3 specialists averaged.
    regime_preds = []
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        preds_cf_list = []
        for s in SEEDS_SPEC:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr_cf, weight=weights, feature_name=top_k, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(250, verbose=False)])
            preds_cf_list.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds.append(np.mean(preds_cf_list, axis=0))

    avg3_cf = np.mean(regime_preds, axis=0)
    avg3_mw = np.clip(from_cf(avg3_cf, active_va), 0, CAPACITY_MW)
    return normalized_mae(y_va_mw, avg3_mw)


def main():
    set_global_seed(42)
    print("Loading data...")
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

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    folds = default_folds()

    # Use Fold-5 for probe (same as training script).
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fold_val = df_train.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]
    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(df_va[TARGET_COL].to_numpy(dtype=np.float32), active_va)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)

    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])

    # Evaluate Fold-1 (winter 2023) and Fold-5 (winter 2025) with K=70 and K=80.
    print("\nMulti-fold comparison (K=70 vs K=80):")
    print("-" * 50)
    for fold_idx in [0, 4]:  # Fold-1 and Fold-5 (winter folds)
        fold = folds[fold_idx]
        for K_val in [70, 80]:
            nmae = eval_fold(df_train, fold, feat_imp, K_val)
            print(f"  Fold-{fold_idx+1} K={K_val}: {nmae:.4f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
