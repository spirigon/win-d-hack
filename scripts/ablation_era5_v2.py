"""Ablation: ERA5 v2 + Open-Meteo NWP ensemble, on the v27.1 mw50/cf50 architecture.

v27.1 (LB 7.605) is the actual best submission and the one we want to
compare against. This harness mirrors its architecture on Fold-5 only
(2025-01-01 → 2025-03-31, the Q1-2026 surrogate window):

   3 regime specialists × 3 seeds × {CF target, MW target} → 50/50 blend.

Three arms, identical except for which extra feature group is merged in:

  A0  baseline               — current pipeline, unchanged.
  A1  +ERA5v2                — adds era5v2_* (BLH, 925/850 hPa, CAPE, LLJ,
                                soil, fluxes, ...) from era5_features_v2.parquet.
  A2  +ERA5v2 + OM ensemble  — A1 plus om_ens_* (4-NWP mean / std / 84m / dir / nwp_diff).

Output: per-arm Fold-5 nMAE for CF, MW, and the 50/50 blend, plus how many
of the new columns ended up in the top-K probe selection.

Usage:

    python scripts/ablation_era5_v2.py
    python scripts/ablation_era5_v2.py --arms A0 A1
    python scripts/ablation_era5_v2.py --top-k 80

Run time: ~3 × the cost of a single v27.1 Fold-5 mw50/cf50 evaluation
(both targets × 3 specialists × 3 seeds = 18 LGBMs per arm). About
8-15 min per arm on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.era5_v2 import (
    era5v2_columns,
    merge_era5_v2,
    merge_om_ensemble,
    om_ensemble_columns,
)
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
RESULTS_PATH = _ROOT / "data" / "processed" / "ablation_era5_v2.json"

TURBINE_RATED_MW = 3.465
DEFAULT_K = 80                          # v27.1 used K=80
DEFAULT_SEEDS = [42, 123, 456]          # v27.1 fast-mode seeds (LB 7.605 used 5; 3 is enough for ablation)
BLEND_WEIGHT_MW = 0.50                  # v27.1 = 50/50 mw + cf

# v27.1 LGBMConfig — Optuna trial that won LB 7.605.
CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)


# -----------------------------------------------------------------------
# v27.1 helpers — copied verbatim from src/training/train_best.py so this
# script stays self-contained and the original is left untouched.
#
# NOTE: ``_load_train_raw`` mirrors what ``train_v29_tuned`` does. The
# loader-level ``load_train`` would silently drop the weather columns due
# to ``GenerationSchema(strict="filter")``, leaving ``v_eff`` NaN
# everywhere on the train side and breaking power-curve fitting.
# -----------------------------------------------------------------------

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


def _train_specialists(
    X_tr, y_tr, X_va, y_va, feat_cols, ws_tr, sample_weight, seeds, config,
):
    """3 regime specialists × N seeds, return averaged val predictions."""
    regime_preds = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        seed_preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**config.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights,
                             feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=config.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
            )
            seed_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds[name] = np.mean(seed_preds, axis=0)
    return np.mean(list(regime_preds.values()), axis=0), regime_preds


# -----------------------------------------------------------------------
# Ablation arm
# -----------------------------------------------------------------------

def _build_combined(arm: str) -> pd.DataFrame:
    """Build the time-sorted combined frame for one ablation arm."""
    df_train = _load_train_raw(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    combined = build_features(combined, sort_by_time=False)
    combined = _merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = _add_era5_rolling(combined)

    if arm in ("A1", "A2"):
        combined = merge_era5_v2(combined)
    if arm == "A2":
        combined = merge_om_ensemble(combined)

    return combined


def run_arm(arm: str, top_k: int, seeds: list[int]) -> dict:
    """Run one ablation arm and return the result dict (Fold-5, mw50/cf50)."""
    print(f"\n{'=' * 72}\n  Arm {arm}  (v27.1 mw50/cf50, Fold-5 only, seeds={seeds})\n{'=' * 72}")
    set_global_seed(42)

    t0 = time.time()
    combined = _build_combined(arm)

    df_train_full = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full)
    df_train_full["_is_impossible"] = impossible.values
    sample_weight_full = (~impossible.to_numpy()).astype(np.float32)

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train_full, fold5)
    fold_train = df_train_full.iloc[train_idx]
    fold_val   = df_train_full.iloc[val_idx]
    fit_data   = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [
        c for c in feature_columns(df_tr)
        if c not in ("_is_impossible", "_split", TARGET_COL)
    ]
    print(f"  Feature pool: {len(feat_cols_all)} columns")
    n_era5v2 = len([c for c in feat_cols_all if c in era5v2_columns(df_tr)])
    n_omens  = len([c for c in feat_cols_all if c in om_ensemble_columns(df_tr)])
    if n_era5v2 or n_omens:
        print(f"    era5v2_* in pool: {n_era5v2}    om_ens_* in pool: {n_omens}")

    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf = _to_cf(y_tr_mw, active_tr)
    y_va_cf = _to_cf(y_va_mw, active_va)
    sample_weight_fold = sample_weight_full[train_idx]

    # ---- Probe for top-K (CF target, single seed = 42) ----
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
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_features = [n for n, _ in feat_imp[:top_k]]
    imp_lookup = dict(feat_imp)

    top_era5v2 = [n for n in top_features if n.startswith("era5v2_")]
    top_omens  = [n for n in top_features if n.startswith("om_ens_")]
    if top_era5v2:
        print(f"  Top-{top_k} contains {len(top_era5v2)} era5v2_* features:")
        for n in top_era5v2:
            print(f"    {n}  (gain={imp_lookup[n]:,.0f})")
    if top_omens:
        print(f"  Top-{top_k} contains {len(top_omens)} om_ens_* features:")
        for n in top_omens:
            print(f"    {n}  (gain={imp_lookup[n]:,.0f})")

    # ---- Train two ensembles: CF target and MW target ----
    X_tr = df_tr[top_features].to_numpy(dtype=np.float32)
    X_va = df_va[top_features].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    print(f"  Training CF specialists (3 × {len(seeds)} seeds)...")
    cf_pred, _ = _train_specialists(
        X_tr, y_tr_cf, X_va, y_va_cf, top_features, ws_tr, sample_weight_fold, seeds, CONFIG,
    )
    cf_mw = np.clip(_from_cf(cf_pred, active_va), 0.0, CAPACITY_MW)
    nmae_cf = float(normalized_mae(y_va_mw, cf_mw))
    print(f"    CF Fold-5 nMAE: {nmae_cf:.4f}%")

    print(f"  Training MW specialists (3 × {len(seeds)} seeds)...")
    mw_pred, _ = _train_specialists(
        X_tr, y_tr_mw, X_va, y_va_mw, top_features, ws_tr, sample_weight_fold, seeds, CONFIG,
    )
    mw_mw = np.clip(mw_pred, 0.0, CAPACITY_MW).astype(np.float32)
    nmae_mw = float(normalized_mae(y_va_mw, mw_mw))
    print(f"    MW Fold-5 nMAE: {nmae_mw:.4f}%")

    # ---- 50/50 blend ----
    blend_mw = BLEND_WEIGHT_MW * mw_mw + (1 - BLEND_WEIGHT_MW) * cf_mw
    blend_mw = np.clip(blend_mw, 0.0, CAPACITY_MW)
    nmae_blend = float(normalized_mae(y_va_mw, blend_mw))

    elapsed = time.time() - t0
    print(f"  BLEND  Fold-5 nMAE: {nmae_blend:.4f}%   ({elapsed:.0f}s)")

    return {
        "arm": arm,
        "fold5_nmae_blend_pct": nmae_blend,
        "fold5_nmae_cf_pct": nmae_cf,
        "fold5_nmae_mw_pct": nmae_mw,
        "n_features_pool": len(feat_cols_all),
        "n_era5v2_in_pool": n_era5v2,
        "n_omens_in_pool": n_omens,
        "top_era5v2": top_era5v2,
        "top_omens": top_omens,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A0", "A1", "A2"],
                    choices=["A0", "A1", "A2"],
                    help="which arms to run (default: all three)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_K,
                    help=f"top-K features to keep after probe (default: {DEFAULT_K})")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help=f"seeds per specialist (default: {DEFAULT_SEEDS})")
    args = ap.parse_args()

    results = []
    for arm in args.arms:
        results.append(run_arm(arm, args.top_k, args.seeds))

    print("\n" + "=" * 72)
    print("  SUMMARY  (v27.1 mw50/cf50 architecture, Fold-5 only)")
    print("=" * 72)
    print(f"  {'arm':<4} {'blend':>8}  {'cf':>8}  {'mw':>8}  {'pool':>5}   notes")
    base = next((r for r in results if r["arm"] == "A0"), None)
    for r in results:
        delta = ""
        if base is not None and r["arm"] != "A0":
            d = r["fold5_nmae_blend_pct"] - base["fold5_nmae_blend_pct"]
            delta = f"  Δblend={d:+.4f} pp"
        notes = f"era5v2_top={len(r['top_era5v2'])}  om_ens_top={len(r['top_omens'])}"
        print(f"  {r['arm']:<4} "
              f"{r['fold5_nmae_blend_pct']:>8.4f}  "
              f"{r['fold5_nmae_cf_pct']:>8.4f}  "
              f"{r['fold5_nmae_mw_pct']:>8.4f}  "
              f"{r['n_features_pool']:>5}   {notes}{delta}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
