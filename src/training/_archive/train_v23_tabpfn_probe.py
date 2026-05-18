"""V23 probe: TabPFN-TS style evaluation on Fold-4 & Fold-5.

Goals:
- Measure TabPFN base performance with random subsampling (3 bags of 10k)
- Compare against LGBM specialists on the same folds
- Decide if TabPFN is worth integrating into v22 ensemble

TabPFN-TS approach (Hoo et al., 2025):
- Treat as tabular regression (row-level, no sequence assumption)
- Use existing calendar/weather features directly
- Subsample with up to 10k rows per model instance; bag N instances

Extra: try "winter-biased" subsampling (favor Nov-Mar rows from AV analysis).

Usage:
    python -m src.training.train_v23_tabpfn_probe
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

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
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"

K = 80
TURBINE_RATED_MW = 3.465
TABPFN_SUBSAMPLE = 8000  # keep well under 10k for speed
TABPFN_BAGS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def tabpfn_predict_bag(X_tr, y_tr, X_va, n_samples, n_bags, seeds, winter_bias_months=None, timestamps=None):
    """Train TabPFN on `n_bags` random subsamples, average predictions.

    winter_bias_months: if provided, oversample rows whose month is in this set.
    """
    from tabpfn import TabPFNRegressor

    # Drop features with near-zero variance and standardize — TabPFN's internal
    # TruncatedSVD is numerically unstable on raw wind-speed/pressure scales.
    var = X_tr.var(axis=0)
    keep_feat = var > 1e-6
    X_tr_f = X_tr[:, keep_feat]
    X_va_f = X_va[:, keep_feat]
    if keep_feat.sum() < X_tr.shape[1]:
        print(f"    Dropped {(~keep_feat).sum()} low-variance features (kept {keep_feat.sum()})")

    mu = X_tr_f.mean(axis=0)
    sigma = X_tr_f.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_tr_f = ((X_tr_f - mu) / sigma).astype(np.float32)
    X_va_f = ((X_va_f - mu) / sigma).astype(np.float32)
    # Clip extreme outliers that can destabilize SVD.
    X_tr_f = np.clip(X_tr_f, -10, 10)
    X_va_f = np.clip(X_va_f, -10, 10)

    preds = []
    N = len(X_tr_f)
    for b, seed in enumerate(seeds[:n_bags]):
        rng = np.random.default_rng(seed)
        if winter_bias_months is not None and timestamps is not None:
            months = pd.Series(timestamps).dt.month.to_numpy()
            is_winter = np.isin(months, list(winter_bias_months))
            w = np.where(is_winter, 3.0, 1.0)
            w = w / w.sum()
            idx = rng.choice(N, size=min(n_samples, N), replace=False, p=w)
        else:
            idx = rng.choice(N, size=min(n_samples, N), replace=False)

        X_sub = X_tr_f[idx]
        y_sub = y_tr[idx]
        # Safety: no NaN/inf.
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
        X_va_clean = np.nan_to_num(X_va_f, nan=0.0, posinf=0.0, neginf=0.0)

        t0 = time.time()
        model = TabPFNRegressor(
            device=DEVICE,
            random_state=int(seed),
            n_estimators=4,
            ignore_pretraining_limits=True,
        )
        model.fit(X_sub, y_sub)
        p = model.predict(X_va_clean)
        preds.append(p)
        print(f"    Bag {b+1}/{n_bags} seed {seed}: n_train={len(idx)}, pred_time={time.time()-t0:.1f}s")
    return np.mean(preds, axis=0)


def train_lgbm_specialist(X_tr, y_tr, X_va, y_va, feat_cols, seeds, weights):
    preds = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        dt = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
        dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        preds.append(b.predict(X_va, num_iteration=b.best_iteration))
    return np.mean(preds, axis=0)


def eval_fold(df_train, fold, top_k, label):
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

    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(y_va_mw, active_va)
    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    print(f"\n{'='*60}\n{label}  train={len(X_tr)}  val={len(X_va)}\n{'='*60}")

    # LGBM specialists avg3 (reference).
    print(f"\n  LGBM specialists (3 seeds):")
    regime = {}
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        w = np.where(mask, 2.0, 0.3).astype(np.float32)
        p = train_lgbm_specialist(X_tr, y_tr_cf, X_va, y_va_cf, top_k, [42, 123, 456], w)
        regime[name] = p
    lgbm_cf = np.mean(list(regime.values()), axis=0)
    lgbm_mw = np.clip(from_cf(lgbm_cf, active_va), 0, CAPACITY_MW)
    lgbm_nmae = normalized_mae(y_va_mw, lgbm_mw)
    print(f"    LGBM avg3: {lgbm_nmae:.4f}%")

    # TabPFN random subsample.
    print(f"\n  TabPFN random subsample ({TABPFN_BAGS} bags x {TABPFN_SUBSAMPLE} rows):")
    tabpfn_cf = tabpfn_predict_bag(X_tr, y_tr_cf, X_va, TABPFN_SUBSAMPLE, TABPFN_BAGS, [42, 123, 456])
    tabpfn_mw = np.clip(from_cf(tabpfn_cf, active_va), 0, CAPACITY_MW)
    tabpfn_nmae = normalized_mae(y_va_mw, tabpfn_mw)
    print(f"    TabPFN random: {tabpfn_nmae:.4f}%")

    # TabPFN winter-biased subsample.
    print(f"\n  TabPFN winter-biased ({TABPFN_BAGS} bags x {TABPFN_SUBSAMPLE} rows):")
    tabpfn_w_cf = tabpfn_predict_bag(
        X_tr, y_tr_cf, X_va, TABPFN_SUBSAMPLE, TABPFN_BAGS, [42, 123, 456],
        winter_bias_months={11, 12, 1, 2, 3},
        timestamps=df_tr[TIMESTAMP_COL].to_numpy(),
    )
    tabpfn_w_mw = np.clip(from_cf(tabpfn_w_cf, active_va), 0, CAPACITY_MW)
    tabpfn_w_nmae = normalized_mae(y_va_mw, tabpfn_w_mw)
    print(f"    TabPFN winter-biased: {tabpfn_w_nmae:.4f}%")

    # Blend search.
    print(f"\n  Blend search (LGBM + TabPFN random):")
    best_b = (0.0, lgbm_nmae)
    for w in np.arange(0.0, 0.81, 0.05):
        blend_cf = (1 - w) * lgbm_cf + w * tabpfn_cf
        blend_mw = np.clip(from_cf(blend_cf, active_va), 0, CAPACITY_MW)
        nmae = normalized_mae(y_va_mw, blend_mw)
        if nmae < best_b[1]:
            best_b = (w, nmae)
    print(f"    Best LGBM+TabPFN(rand): w_tabpfn={best_b[0]:.2f} -> {best_b[1]:.4f}%")

    # Blend with winter-biased.
    best_bw = (0.0, lgbm_nmae)
    for w in np.arange(0.0, 0.81, 0.05):
        blend_cf = (1 - w) * lgbm_cf + w * tabpfn_w_cf
        blend_mw = np.clip(from_cf(blend_cf, active_va), 0, CAPACITY_MW)
        nmae = normalized_mae(y_va_mw, blend_mw)
        if nmae < best_bw[1]:
            best_bw = (w, nmae)
    print(f"    Best LGBM+TabPFN(winter): w_tabpfn={best_bw[0]:.2f} -> {best_bw[1]:.4f}%")

    return {
        "lgbm": lgbm_nmae,
        "tabpfn_rand": tabpfn_nmae,
        "tabpfn_winter": tabpfn_w_nmae,
        "blend_rand": best_b,
        "blend_winter": best_bw,
    }


def main():
    set_global_seed(42)
    print("=" * 60)
    print(f"V23 probe: TabPFN-TS ({TABPFN_BAGS} bags x {TABPFN_SUBSAMPLE} rows)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nPreparing data...")
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

    # Select top-K features via Fold-5 probe.
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
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
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
    print(f"  Selected {len(top_k)} features\n")

    # Evaluate Fold-4 (2024 Q4) and Fold-5 (2025 Q1).
    results = {}
    results["F4"] = eval_fold(df_train, folds[3], top_k, "Fold 4 (2024 Q4)")
    results["F5"] = eval_fold(df_train, folds[4], top_k, "Fold 5 (2025 Q1)")

    # Summary.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Fold':<4} {'LGBM':>9} {'TabPFN-rand':>12} {'TabPFN-winter':>14} {'Best LGBM+rand':>18} {'Best LGBM+winter':>18}")
    for fold, r in results.items():
        print(f"{fold:<4} {r['lgbm']:>8.4f}% {r['tabpfn_rand']:>11.4f}% {r['tabpfn_winter']:>13.4f}% {r['blend_rand'][1]:>10.4f}%@{r['blend_rand'][0]:.2f} {r['blend_winter'][1]:>10.4f}%@{r['blend_winter'][0]:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
