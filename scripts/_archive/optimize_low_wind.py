"""Test different low-wind strategies on Fold-5.

Specifically: boost predictions for rows where v_eff < 3 (cut-in region).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Load trained v8 models.
from src.data.loaders import load_train, load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed
import lightgbm as lgb


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


def main():
    set_global_seed(42)
    df_train = load_train(_ROOT / "data" / "raw" / "train_dataset.csv")
    era5 = pd.read_parquet(_ROOT / "data" / "external" / "era5_reanalysis.parquet")
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    df_train = build_features(df_train, sort_by_time=False)
    df_train = merge_era5(df_train, era5)
    df_train = add_datasheet_power_features(df_train)
    df_train = add_extra_features(df_train)

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)
    feat_cols_all = [c for c in feature_columns(df_tr) if c != "_is_impossible"]

    config = LGBMConfig(
        num_leaves=453, min_data_in_leaf=121, learning_rate=0.01555,
        feature_fraction=0.695, bagging_fraction=0.509, bagging_freq=5,
        lambda_l1=0.00294, lambda_l2=0.342,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()

    # Probe + select top 60.
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    dtrain = lgb.Dataset(X_tr_all, label=y_tr, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train({**config.to_params(), "seed": 42}, dtrain, num_boost_round=4000,
                       valid_sets=[dval], valid_names=["val"],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:60]]

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)

    # Train 5 models.
    SEEDS = [42, 123, 456, 789, 2026]
    preds_all = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=top_k, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=top_k, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = np.clip(b.predict(X_va, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        preds_all.append(p)
    ens = np.mean(preds_all, axis=0)
    print(f"Baseline ensemble: {normalized_mae(y_va, ens):.4f} %")

    # Test different low-wind strategies.
    strategies = {
        "no change": ens,
        "zero < 3": np.where(v_eff_va < 3.0, 0.0, ens),
        "zero < 2.5": np.where(v_eff_va < 2.5, 0.0, ens),
        "zero < 2.0": np.where(v_eff_va < 2.0, 0.0, ens),
        "boost in [1,3] by 2 MW": np.where(
            (v_eff_va >= 1.0) & (v_eff_va < 3.0), ens + 2.0, ens
        ),
        "boost by factor 1.3 in [1,4]": np.where(
            (v_eff_va >= 1.0) & (v_eff_va < 4.0), ens * 1.3, ens
        ),
    }

    print("\nLow-wind strategy comparison (Fold-5 nMAE):")
    for name, p in strategies.items():
        p = np.clip(p, 0, CAPACITY_MW)
        n = normalized_mae(y_va, p)
        print(f"  {name}: {n:.4f} %")

    # What IS the truth in that regime?
    mask_low = v_eff_va < 3.0
    if mask_low.sum() > 0:
        print(f"\nIn low-wind regime (v_eff<3.0): n={mask_low.sum()}")
        print(f"  True power mean: {y_va[mask_low].mean():.2f} MW, median: {np.median(y_va[mask_low]):.2f}")
        print(f"  v8 ensemble mean: {ens[mask_low].mean():.2f} MW")
        print(f"  MAE in that regime: {np.mean(np.abs(y_va[mask_low] - ens[mask_low])):.2f} MW")
        print(f"  Zero-out MAE: {np.mean(np.abs(y_va[mask_low] - 0)):.2f} MW")


if __name__ == "__main__":
    main()
