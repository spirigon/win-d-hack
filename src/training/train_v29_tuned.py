"""V29: CF-target regime-specialist ensemble with verified Optuna HPs.

Optuna trial 23 params (lr=0.026, leaves=55, mdl=43) verified at 5 seeds:
Fold-5 = 7.5734% — best confirmed result so far (+0.031 pp vs v15 baseline).

Usage:
    python -m src.training.train_v29_tuned
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

from src.data.loaders import load_valid_features
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
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v29.0_tuned_cf.csv"

TURBINE_RATED_MW = 3.465
K = 70
SEEDS_SPEC = [42, 123, 456, 789, 2026]

# Optuna trial 23 — verified 5-seed Fold-5: 7.5734%
# (trial 38 had lr=0.047 which was 3-seed noise; trial 23 is more robust)
CONFIG = LGBMConfig(
    num_leaves=55,
    min_data_in_leaf=43,
    learning_rate=0.0259476307435388,
    feature_fraction=0.49455788842926784,
    bagging_fraction=0.4229521099231461,
    bagging_freq=3,
    lambda_l1=1.599611931632972,
    lambda_l2=0.0003476780244211917,
    num_boost_round=5000,
    early_stopping_rounds=250,
    log_period=0,
)


def _load_train_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")
    return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)


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


def _merge_era5(df, era5):
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
    era5_new = [
        c for c in df.columns
        if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"
    ]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def _add_era5_rolling(df):
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
    df["era5_dir_sin_diff1"] = df["era5_dir100_sin"].diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = df["era5_dir100_cos"].diff(1).fillna(0)
    return df


def _to_cf(y_mw, active_turbines):
    denom = np.maximum(active_turbines.astype(np.float32) * TURBINE_RATED_MW, 1e-3)
    return (y_mw / denom).astype(np.float32)


def _from_cf(cf, active_turbines):
    cf = np.clip(cf, 0.0, 1.0)
    return (cf * active_turbines.astype(np.float32) * TURBINE_RATED_MW).astype(np.float32)


def _train_specialists_valid(X_tr, y_tr_cf, X_va, y_va_cf, feat_cols, ws_tr, sample_weight):
    regime_preds = {}
    regime_iters = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        seed_preds = []
        iters = []
        for s in SEEDS_SPEC:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dtrain = lgb.Dataset(
                X_tr, label=y_tr_cf, weight=weights,
                feature_name=feat_cols, free_raw_data=False,
            )
            dval = lgb.Dataset(X_va, label=y_va_cf, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dtrain, num_boost_round=5000,
                valid_sets=[dval], valid_names=["val"],
                callbacks=[lgb.early_stopping(250, verbose=False)],
            )
            seed_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
            iters.append(b.best_iteration)
        regime_preds[name] = np.mean(seed_preds, axis=0)
        regime_iters[name] = iters
    return regime_preds, regime_iters


def _train_specialists_full(X, y_cf, feat_cols, ws, sample_weight, n_rounds_per_spec, X_test):
    regime_preds = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws >= lo) & (ws < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        n_rounds = n_rounds_per_spec[name]
        seed_preds = []
        for s in SEEDS_SPEC:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dtrain = lgb.Dataset(X, label=y_cf, weight=weights, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dtrain, num_boost_round=n_rounds)
            seed_preds.append(b.predict(X_test))
        regime_preds[name] = np.mean(seed_preds, axis=0)
    return regime_preds


def main() -> None:
    set_global_seed(42)
    print("V29: CF-target + Optuna trial-23 HPs (verified 5-seed Fold-5: 7.5734%)")
    print("=" * 70)

    print("\nLoading and building features...")
    df_train_raw = _load_train_raw(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train_raw["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train_raw, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = _merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = _add_era5_rolling(combined)

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    sample_weight_full = (~impossible.to_numpy()).astype(np.float32)

    print("\n=== Fold-5 evaluation ===")
    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(df_train.iloc[val_idx], pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [
        c for c in feature_columns(df_tr)
        if c not in ("_is_impossible", "_split", TARGET_COL)
    ]
    active_tr = df_tr["active_turbines"].to_numpy()
    active_va = df_va["active_turbines"].to_numpy()
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf = _to_cf(y_tr_mw, active_tr)
    y_va_cf = _to_cf(y_va_mw, active_va)
    sample_weight_fold = sample_weight_full[train_idx]

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    probe_cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain_probe = lgb.Dataset(
        X_tr_all, label=y_tr_cf, weight=sample_weight_fold,
        feature_name=feat_cols_all, free_raw_data=False,
    )
    dval_probe = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        probe_cfg.to_params(), dtrain_probe, num_boost_round=5000,
        valid_sets=[dval_probe], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp, strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    print(f"  Top-{K} features selected from {len(feat_cols_all)} total")

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    print("  Training 3 specialists × 5 seeds...")
    regime_preds, regime_iters = _train_specialists_valid(
        X_tr, y_tr_cf, X_va, y_va_cf, top_k, ws_tr, sample_weight_fold,
    )

    for name, preds_cf in regime_preds.items():
        preds_mw = np.clip(_from_cf(preds_cf, active_va), 0.0, CAPACITY_MW)
        print(f"    {name}: {normalized_mae(y_va_mw, preds_mw):.4f}%")

    avg_cf = np.mean(list(regime_preds.values()), axis=0)
    avg_mw = np.clip(_from_cf(avg_cf, active_va), 0.0, CAPACITY_MW)
    nmae_fold5 = normalized_mae(y_va_mw, avg_mw)
    print(f"\n  Fold-5 nMAE (avg 3 specialists): {nmae_fold5:.4f}%")

    print("\n=== Full-fit ===")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    wake_full = fit_wake_lookup(df_train_clean, n_sectors=16)

    df_train_full = _add_pc(df_train, pc_sector_full, pc_global_full)
    df_train_full = add_wake_features(df_train_full, wake_full)
    X_full = df_train_full[top_k].to_numpy(dtype=np.float32)
    y_full_mw = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)
    active_full = df_train_full["active_turbines"].to_numpy()
    y_full_cf = _to_cf(y_full_mw, active_full)
    ws_full = df_train_full["wind_speed_120m"].to_numpy()

    df_vp = _add_pc(df_valid_sorted, pc_sector_full, pc_global_full)
    df_vp = add_wake_features(df_vp, wake_full)
    for c in set(top_k) - set(df_vp.columns):
        df_vp[c] = 0.0
    X_valid = df_vp[top_k].to_numpy(dtype=np.float32)

    n_rounds_per_spec = {}
    for name, iters in regime_iters.items():
        n_rounds_per_spec[name] = max(int(np.median(iters) * 1.2), 500)
        print(f"  {name}: {len(SEEDS_SPEC)} seeds, {n_rounds_per_spec[name]} rounds")

    valid_regime_preds = _train_specialists_full(
        X_full, y_full_cf, top_k, ws_full, sample_weight_full, n_rounds_per_spec, X_valid,
    )

    valid_avg_cf = np.mean(list(valid_regime_preds.values()), axis=0)
    valid_avg_mw = np.clip(_from_cf(valid_avg_cf, df_vp["active_turbines"].to_numpy()), 0.0, CAPACITY_MW)

    order = df_vp["_submission_row"].to_numpy().astype(int)
    po = np.empty(len(valid_avg_mw), dtype=np.float64)
    po[order] = valid_avg_mw.astype(np.float64)

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))

    print(f"\n  Submission saved: {SUBMISSION_PATH}")
    print(f"  Mean: {valid_avg_mw.mean():.2f} MW")
    print(f"  Fold-5 nMAE: {nmae_fold5:.4f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
