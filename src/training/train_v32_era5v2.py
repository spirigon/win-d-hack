"""V32: v27.1 mw50/cf50 architecture + ERA5v2 features.

This is the v27.1 production recipe (3-fold CV-bag × 3 regime specialists ×
N seeds × {CF target, MW target}, 50/50 blend) with a single change:
``data/interim/era5_features_v2.parquet`` is merged into the feature frame
to expose ERA5 boundary-layer height, 925/850 hPa winds, low-level-jet
strength, CAPE, total-column water vapour, sensible/latent heat fluxes,
soil temperature & moisture, etc.

Ablation result (Fold-5, 3 seeds, scripts/ablation_era5_v2.py):

    arm  blend    cf      mw       Δblend
    A0   7.6413   7.5900  7.7325
    A1   7.5806   7.5540  7.6595   −0.0607 pp   (this script)

29 of the 37 added ``era5v2_*`` columns ranked in the top-80 importance.

Reproduction:

    python -m src.training.train_v32_era5v2

Outputs:

    submissions/archive/v32.0_era5v2_mw50_cf50.csv

Notes:
- Hyperparameters are unchanged from v27.1 (Optuna trial that won LB 7.605).
- The ``--seeds`` flag selects {3, 5} seed bags. The historical LB 7.605
  used 5 seeds; 3 is used for fast iteration.
- The Open-Meteo NWP ensemble (A2 in the ablation) is *not* included here.
  Its added value (-0.013 pp on Fold-5) was within seed noise and the
  features were partial substitutes for the ERA5v2 winds. If we want it,
  we'll port it in a separate v33.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from src.features.era5_v2 import era5v2_columns, merge_era5_v2
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

# --- Paths (mirrors src/training/train_best.py) ------------------------
TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v32.0_era5v2_mw50_cf50.csv"

# --- Hyperparameters (v27.1 / train_best.py — Optuna trial that won LB 7.605) ---
LGBM_PARAMS = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)

# --- Constants ---------------------------------------------------------
SEEDS_3 = [42, 123, 456]
SEEDS_5 = [42, 123, 456, 789, 2026]
K = 80                       # v27.1 used K=80
FOLD_IDS = [2, 3, 4]         # CV-bag over folds 3, 4, 5 (most recent)
TURBINE_RATED_MW = 3.465
BLEND_WEIGHT_MW = 0.50       # 50% MW + 50% CF


# ------------------------------------------------------------------------
# Helpers — copied from train_best.py to keep this script self-contained
# (the original is left untouched).
# ------------------------------------------------------------------------

def _load_train_raw(path: str | Path) -> pd.DataFrame:
    """Read the training CSV without going through the schema-filter loader.

    ``load_train`` strips the weather columns due to
    ``GenerationSchema(strict='filter')``, which leaves ``v_eff`` NaN on
    every train row and breaks the power-curve fit. ``train_v29_tuned``
    works around this with the same direct ``read_csv`` path.
    """
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


# ------------------------------------------------------------------------
# Inner training loop — copied verbatim from train_best.py
# ------------------------------------------------------------------------

def _train_fold_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, seeds, config):
    """3 regime specialists × N seeds with early stopping. Return (val_preds, test_preds)."""
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        val_preds, test_preds = [], []
        for s in seeds:
            cfg = LGBMConfig(**{**config.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=config.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
            )
            val_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
            test_preds.append(b.predict(X_test, num_iteration=b.best_iteration))
        regime_val[name] = np.mean(val_preds, axis=0)
        regime_test[name] = np.mean(test_preds, axis=0)
    return np.mean(list(regime_val.values()), axis=0), np.mean(list(regime_test.values()), axis=0)


def _run_cv_bag(df_train, df_valid_sorted, top_k, folds, target_mode, seeds):
    """3-fold CV-bag with given target mode. Returns averaged test predictions."""
    test_preds_per_fold = []
    fold_nmaes = []

    for fold_idx in FOLD_IDS:
        fold = folds[fold_idx]
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
        y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr = df_tr["wind_speed_120m"].to_numpy()

        if target_mode == "cf":
            y_tr = _to_cf(y_tr_mw, active_tr)
            y_va = _to_cf(y_va_mw, active_va)
        else:  # "mw"
            y_tr = y_tr_mw
            y_va = y_va_mw

        val_pred, test_pred = _train_fold_ensemble(
            X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, seeds, LGBM_PARAMS,
        )
        test_preds_per_fold.append(test_pred)

        if target_mode == "cf":
            val_mw = np.clip(_from_cf(val_pred, active_va), 0, CAPACITY_MW)
        else:
            val_mw = np.clip(val_pred, 0, CAPACITY_MW)
        fold_nmae = float(normalized_mae(y_va_mw, val_mw))
        fold_nmaes.append(fold_nmae)
        print(f"    Fold {fold_idx + 1} ({target_mode}): {fold_nmae:.4f}%")

    return np.mean(test_preds_per_fold, axis=0), fold_nmaes


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["3", "5"], default="5",
                    help="seeds per specialist (3=fast, 5=production / LB 7.605 baseline). Default: 5")
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH,
                    help=f"submission output path (default: {OUTPUT_PATH.name})")
    args = ap.parse_args()

    seeds = SEEDS_5 if args.seeds == "5" else SEEDS_3
    set_global_seed(42)

    print("=" * 72)
    print(f"V32: v27.1 mw50/cf50 + ERA5v2 features  ({len(seeds)} seeds)")
    print("=" * 72)

    # --- Load and build features ----------------------------------------
    print("\n[1/4] Loading data...")
    df_train = _load_train_raw(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    print("[2/4] Building features...")
    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = _merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = _add_era5_rolling(combined)
    combined = merge_era5_v2(combined)   # ★ the V32 change

    df_train_full = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full)
    df_train_full["_is_impossible"] = impossible.values

    # --- Feature selection (probe on Fold-5) ----------------------------
    print("[3/4] Feature selection (probe on Fold-5)...")
    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train_full, fold5)
    fold_train_ = df_train_full.iloc[train_idx]
    fit_data_ = fold_train_[~fold_train_["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)

    df_t_ = _add_pc(fold_train_, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = _add_pc(df_train_full.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)

    feat_cols_all = [
        c for c in feature_columns(df_t_)
        if c not in ("_is_impossible", "_split", TARGET_COL)
    ]
    print(f"  Feature pool: {len(feat_cols_all)}  ({len(era5v2_columns(df_t_))} era5v2_*)")

    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = _to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = _to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    X_t_ = df_t_[feat_cols_all].to_numpy(dtype=np.float32)
    X_v_ = df_v_[feat_cols_all].to_numpy(dtype=np.float32)

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(X_t_, label=y_t_cf_, feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_v_, label=y_v_cf_, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    n_era5v2_top = sum(1 for n in top_k if n.startswith("era5v2_"))
    print(f"  Top-{K} features: {n_era5v2_top} are era5v2_*")

    # --- Train both target variants on the CV-bag -----------------------
    print(f"\n[4/4] Training ensembles (CV-bag over folds {[i + 1 for i in FOLD_IDS]}, "
          f"seeds={seeds})...")

    t_cf = time.time()
    print("\n  --- CF target ---")
    test_cf, cf_fold_nmaes = _run_cv_bag(
        df_train_full, df_valid_sorted, top_k, folds, target_mode="cf", seeds=seeds,
    )
    print(f"  CF mean fold nMAE: {np.mean(cf_fold_nmaes):.4f}%   ({time.time() - t_cf:.0f}s)")

    t_mw = time.time()
    print("\n  --- MW target ---")
    test_mw_raw, mw_fold_nmaes = _run_cv_bag(
        df_train_full, df_valid_sorted, top_k, folds, target_mode="mw", seeds=seeds,
    )
    print(f"  MW mean fold nMAE: {np.mean(mw_fold_nmaes):.4f}%   ({time.time() - t_mw:.0f}s)")

    # --- Convert and blend ----------------------------------------------
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    pred_cf_mw = np.clip(_from_cf(test_cf, active_valid), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(test_mw_raw, 0, CAPACITY_MW)
    final_mw = BLEND_WEIGHT_MW * pred_mw_mw + (1 - BLEND_WEIGHT_MW) * pred_cf_mw
    final_mw = np.clip(final_mw, 0, CAPACITY_MW)

    # --- Restore original (descending) submission row order -------------
    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw

    # Restore timestamps in submission row order so write_submission can
    # build the (timestamp, prediction) two-column CSV correctly.
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    ts_po = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, args.output, expected_rows=len(df_valid), timestamps=ts_po)

    print(f"\n{'=' * 72}")
    print(f"  Submission saved: {args.output}")
    print(f"  Mean: {final_mw.mean():.2f} MW   Std: {final_mw.std():.2f} MW   "
          f"Range: [{final_mw.min():.2f}, {final_mw.max():.2f}]")
    print(f"  CF mean fold nMAE: {np.mean(cf_fold_nmaes):.4f}%")
    print(f"  MW mean fold nMAE: {np.mean(mw_fold_nmaes):.4f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
