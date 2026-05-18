"""V25 submission: quantile alpha=0.48 with 3-fold CV-bagging (v20 methodology).

Based on v25 experiment: quantile 0.48 gave -0.016 pp on Fold-5.
Apply same CV-bag approach as v20.1 (LB 7.630) but with quantile objective.

Also produce a blend of v20.1 (MAE) + v25 (quantile) for diversity.
"""
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
AI_WEATHER_PATH = _ROOT / "data" / "external" / "ai_weather_models.parquet"

SEEDS_SPEC = [42, 123, 456, 789, 2026]
K = 80
FOLD_IDS = [2, 3, 4]  # Folds 3, 4, 5
TURBINE_RATED_MW = 3.465
ALPHA = 0.48

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


def merge_ai_weather(df, ai_df):
    df = df.copy()
    ai = ai_df.copy().rename(columns={"time": TIMESTAMP_COL})
    usable = [c for c in ai.columns if c != TIMESTAMP_COL and ai[c].notna().sum() > 1000]
    ai = ai[[TIMESTAMP_COL] + usable]
    df = df.merge(ai, on=TIMESTAMP_COL, how="left")
    if "graphcast_wind_speed_10m" in df.columns:
        df["gc_ws10_bias"] = df["wind_speed_10m"] - df["graphcast_wind_speed_10m"]
        df["gc_pressure_bias"] = df["pressure_msl"] - df["graphcast_pressure_msl"]
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


def main():
    set_global_seed(42)
    print("=" * 60)
    print(f"V25: Quantile alpha={ALPHA} + 3-fold CV-bag")
    print("=" * 60)

    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    ai_weather = pd.read_parquet(AI_WEATHER_PATH) if AI_WEATHER_PATH.exists() else None

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)
    if ai_weather is not None:
        combined = merge_ai_weather(combined, ai_weather)

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    folds = default_folds()

    # Feature probe from Fold-5.
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train_ = df_train.iloc[train_idx]
    fit_data_ = fold_train_[~fold_train_["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)
    df_t_ = _add_pc(fold_train_, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = _add_pc(df_train.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)
    feat_cols_all = [c for c in feature_columns(df_t_) if c not in ("_is_impossible", "_split")]
    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    X_t_all_ = df_t_[feat_cols_all].to_numpy(dtype=np.float32)
    X_v_all_ = df_v_[feat_cols_all].to_numpy(dtype=np.float32)

    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dt_ = lgb.Dataset(X_t_all_, label=y_t_cf_, feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_v_all_, label=y_v_cf_, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt_, num_boost_round=5000,
                      valid_sets=[dv_], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    # 3-fold CV-bag with quantile objective.
    test_preds_per_fold = []
    for fold_idx in FOLD_IDS:
        fold = folds[fold_idx]
        print(f"\nFold {fold_idx+1}...")
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
        df_te = _add_pc(df_valid_sorted, pc_sector, pc_global)
        df_te = add_wake_features(df_te, wake)
        for c in set(top_k) - set(df_te.columns):
            df_te[c] = 0.0

        active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
        active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
        y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
        y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_cf = to_cf(y_va_mw, active_va)
        X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr = df_tr["wind_speed_120m"].to_numpy()

        regime_test_cf = {}
        for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
            mask = (ws_tr >= lo) & (ws_tr < hi)
            weights = np.where(mask, 2.0, 0.3).astype(np.float32)
            test_preds = []
            for s in SEEDS_SPEC:
                cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
                params = cfg.to_params()
                params["objective"] = "quantile"
                params["alpha"] = ALPHA
                del params["metric"]
                dt = lgb.Dataset(X_tr, label=y_tr_cf, weight=weights, feature_name=top_k, free_raw_data=False)
                dv = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
                b = lgb.train(params, dt, num_boost_round=5000,
                              valid_sets=[dv], valid_names=["val"],
                              callbacks=[lgb.early_stopping(250, verbose=False)])
                test_preds.append(b.predict(X_test, num_iteration=b.best_iteration))
            regime_test_cf[name] = np.mean(test_preds, axis=0)

        fold_test_cf = np.mean(list(regime_test_cf.values()), axis=0)
        test_preds_per_fold.append(fold_test_cf)
        # Fold-level validation.
        val_preds = []
        for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
            mask = (ws_tr >= lo) & (ws_tr < hi)
            weights = np.where(mask, 2.0, 0.3).astype(np.float32)
            vp = []
            for s in SEEDS_SPEC:
                cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
                params = cfg.to_params()
                params["objective"] = "quantile"
                params["alpha"] = ALPHA
                del params["metric"]
                dt = lgb.Dataset(X_tr, label=y_tr_cf, weight=weights, feature_name=top_k, free_raw_data=False)
                dv = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
                b = lgb.train(params, dt, num_boost_round=5000,
                              valid_sets=[dv], valid_names=["val"],
                              callbacks=[lgb.early_stopping(250, verbose=False)])
                vp.append(b.predict(X_va, num_iteration=b.best_iteration))
            val_preds.append(np.mean(vp, axis=0))
        val_cf = np.mean(val_preds, axis=0)
        val_mw = np.clip(from_cf(val_cf, active_va), 0, CAPACITY_MW)
        print(f"  Fold {fold_idx+1} val nMAE: {normalized_mae(y_va_mw, val_mw):.4f}%")

    # Average test predictions across folds.
    test_cf = np.mean(test_preds_per_fold, axis=0)
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    test_mw = np.clip(from_cf(test_cf, active_valid), 0, CAPACITY_MW)

    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(test_mw)
    po[order] = test_mw
    out_path = _ROOT / "submissions" / "archive" / f"v25.0_quantile_{ALPHA}.csv"
    write_submission(po, out_path, expected_rows=len(df_valid))
    print(f"\nSaved: {out_path.name}  mean={test_mw.mean():.2f} MW")

    # Also blend with v20.1 (MAE-based, LB 7.630).
    v20_path = _ROOT / "submissions" / "archive" / "v20.1_lgbm_cv.csv"
    if v20_path.exists():
        v20_df = pd.read_csv(v20_path)
        v20_preds = v20_df[TARGET_COL].to_numpy()
        for w in [0.3, 0.5, 0.7]:
            blend = w * po + (1 - w) * v20_preds
            blend_path = _ROOT / "submissions" / "archive" / f"v25.1_blend_q{int(w*100)}_mae{int((1-w)*100)}.csv"
            write_submission(blend, blend_path, expected_rows=len(df_valid),
                             timestamps=v20_df[TIMESTAMP_COL].to_numpy())
            print(f"  Blend q{int(w*100)}/mae{int((1-w)*100)}: mean={blend.mean():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
