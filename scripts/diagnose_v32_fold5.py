"""Diagnose the v32 Fold-5 residuals to find concrete error bottlenecks.

Re-runs the v32 pipeline up to Fold-5 OOF predictions only (no full CV-bag,
no full-fit) and breaks down the absolute error by:

  - wind-speed regime (sub-cut-in / cut-in→rated / rated / above-cut-out)
  - hour-of-day
  - air-density regime
  - magnitude of NWP/ERA5 disagreement
  - operating-fraction (n_repair) effect
  - large vs small wind-ramp hours

Prints how many MW of nMAE budget each bucket is spending so we can rank
fixes by hard-evidence, not gut.

Usage:

    python scripts/diagnose_v32_fold5.py
"""

from __future__ import annotations

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
from src.features.era5_v2 import era5v2_columns, merge_era5_v2
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

# Reuse the same config and helpers as the v32 production script.
from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, K, BLEND_WEIGHT_MW, SEEDS_3,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"


def _train_specialists(X_tr, y_tr, X_va, y_va, feat_cols, ws_tr, sample_weight, seeds):
    """3 regime specialists × N seeds, returns averaged val predictions."""
    regime_preds = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        seed_preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights,
                             feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=LGBM_PARAMS.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(LGBM_PARAMS.early_stopping_rounds, verbose=False)],
            )
            seed_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds[name] = np.mean(seed_preds, axis=0)
    return np.mean(list(regime_preds.values()), axis=0)


def main():
    set_global_seed(42)
    print("Diagnosing v32 Fold-5 residuals...")
    t0 = time.time()

    # --- Build features (same path as train_v32_era5v2) --------------------
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
    combined = merge_era5_v2(combined)

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
    print(f"  Feature pool: {len(feat_cols_all)}  ({len(era5v2_columns(df_tr))} era5v2_*)")

    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf = _to_cf(y_tr_mw, active_tr)
    y_va_cf = _to_cf(y_va_mw, active_va)
    sample_weight_fold = sample_weight_full[train_idx]

    # --- Probe → top-K -----------------------------------------------------
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(X_tr_all, label=y_tr_cf, weight=sample_weight_fold,
                      feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    # --- Train CF + MW specialists on Fold-5 (3 seeds for speed) ----------
    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    print("  Training CF specialists (3 × 3 seeds)...")
    cf_pred = _train_specialists(
        X_tr, y_tr_cf, X_va, y_va_cf, top_k, ws_tr, sample_weight_fold, SEEDS_3,
    )
    cf_mw = np.clip(_from_cf(cf_pred, active_va), 0.0, CAPACITY_MW)

    print("  Training MW specialists (3 × 3 seeds)...")
    mw_pred = _train_specialists(
        X_tr, y_tr_mw, X_va, y_va_mw, top_k, ws_tr, sample_weight_fold, SEEDS_3,
    )
    mw_mw = np.clip(mw_pred, 0.0, CAPACITY_MW).astype(np.float32)

    blend_mw = (BLEND_WEIGHT_MW * mw_mw + (1 - BLEND_WEIGHT_MW) * cf_mw).clip(0, CAPACITY_MW)
    nmae_blend = float(normalized_mae(y_va_mw, blend_mw))
    nmae_cf    = float(normalized_mae(y_va_mw, cf_mw))
    nmae_mw    = float(normalized_mae(y_va_mw, mw_mw))
    print(f"\n  Fold-5 nMAE — blend: {nmae_blend:.4f}%  cf: {nmae_cf:.4f}%  mw: {nmae_mw:.4f}%   "
          f"({time.time() - t0:.0f}s)")

    # --- Residual diagnostics on the BLEND (which is what we ship) -------
    abs_err = np.abs(y_va_mw - blend_mw)
    signed_err = blend_mw - y_va_mw   # positive = over-prediction
    n = len(y_va_mw)
    total_err = abs_err.sum()

    def report(label: str, mask: np.ndarray) -> None:
        if mask.sum() == 0:
            print(f"  {label:<42s}  (empty)")
            return
        share_rows = mask.mean() * 100
        contrib_pct = abs_err[mask].sum() / total_err * 100
        local_nmae = abs_err[mask].mean() / CAPACITY_MW * 100
        bias_mw = signed_err[mask].mean()
        print(f"  {label:<42s}  rows={mask.sum():>4} ({share_rows:5.1f}%)  "
              f"local_nMAE={local_nmae:6.3f}%  share_of_err={contrib_pct:5.1f}%  "
              f"bias={bias_mw:+6.2f} MW")

    print()
    print("─" * 72)
    print(" Wind-speed regime (NWP wind_speed_120m at hub):")
    print("─" * 72)
    ws = df_va["wind_speed_120m"].to_numpy()
    report("ws < 3 m/s        (sub-cut-in)",        ws < 3.0)
    report("3-7 m/s           (cut-in → mid-curve)",  (ws >= 3.0) & (ws < 7.0))
    report("7-12 m/s          (steepest slope)",     (ws >= 7.0) & (ws < 12.0))
    report("12-17 m/s         (rated, flat)",        (ws >= 12.0) & (ws < 17.0))
    report("17-25 m/s         (storm regulation)",   (ws >= 17.0) & (ws < 25.0))
    report("ws >= 25 m/s      (above cut-out)",      ws >= 25.0)

    print()
    print("─" * 72)
    print(" ERA5-vs-NWP wind disagreement (|era5_wind_speed_120m - wind_speed_120m|):")
    print("─" * 72)
    if "era5v2_wind_speed_120m" in df_va.columns:
        nwp = df_va["wind_speed_120m"].to_numpy()
        e5  = df_va["era5v2_wind_speed_120m"].to_numpy()
        disagree = np.abs(nwp - e5)
        q = np.nanquantile(disagree, [0.5, 0.75, 0.9])
        print(f"  median disagree: {q[0]:.2f}  p75: {q[1]:.2f}  p90: {q[2]:.2f}  m/s")
        report("|disagree| <= p50",                disagree <= q[0])
        report("|disagree| in (p50, p75]",         (disagree > q[0]) & (disagree <= q[1]))
        report("|disagree| in (p75, p90]",         (disagree > q[1]) & (disagree <= q[2]))
        report("|disagree|  > p90",                disagree > q[2])

    print()
    print("─" * 72)
    print(" Wind ramps (|Δ wind_speed_120m| over 3 h):")
    print("─" * 72)
    ts_va = df_va[TIMESTAMP_COL].to_numpy()
    df_va_sorted = df_va.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    ramp_3h = df_va_sorted["wind_speed_120m"].diff(3).to_numpy()
    abs_ramp = np.abs(ramp_3h)
    # Map back to the original df_va order to align with abs_err
    order_back = df_va.reset_index().merge(
        df_va_sorted.reset_index().rename(columns={"index": "sorted_idx"}),
        on=TIMESTAMP_COL,
        how="left",
        suffixes=("_orig", "_sorted"),
    )["sorted_idx"].to_numpy()
    abs_ramp_va = abs_ramp[order_back]
    rq = np.nanquantile(abs_ramp_va[~np.isnan(abs_ramp_va)], [0.9, 0.95])
    print(f"  3h ramp p90: {rq[0]:.2f}  p95: {rq[1]:.2f}  m/s")
    report("|ramp_3h|  <= p90 (calm)",      ~np.isnan(abs_ramp_va) & (abs_ramp_va <= rq[0]))
    report("|ramp_3h|  in (p90, p95]",       ~np.isnan(abs_ramp_va) & (abs_ramp_va > rq[0]) & (abs_ramp_va <= rq[1]))
    report("|ramp_3h|  > p95 (severe)",      ~np.isnan(abs_ramp_va) & (abs_ramp_va > rq[1]))

    print()
    print("─" * 72)
    print(" Hour of day:")
    print("─" * 72)
    hr = pd.to_datetime(ts_va).hour if hasattr(pd.to_datetime(ts_va), "hour") else None
    hours = pd.Series(pd.to_datetime(ts_va)).dt.hour.to_numpy()
    for buckets in ((0, 6), (6, 12), (12, 18), (18, 24)):
        report(f"hour {buckets[0]:02d}-{buckets[1]:02d}",
               (hours >= buckets[0]) & (hours < buckets[1]))

    print()
    print("─" * 72)
    print(" Air density regime:")
    print("─" * 72)
    if "air_density" in df_va.columns:
        rho = df_va["air_density"].to_numpy()
        rq = np.nanquantile(rho, [0.25, 0.75])
        report(f"rho < {rq[0]:.3f} (warm/low)",    rho < rq[0])
        report(f"rho in [{rq[0]:.3f}, {rq[1]:.3f}]",
               (rho >= rq[0]) & (rho <= rq[1]))
        report(f"rho > {rq[1]:.3f} (cold/high)",  rho > rq[1])

    print()
    print("─" * 72)
    print(" Operating fraction (n_repair):")
    print("─" * 72)
    nrep = df_va["n_repair"].to_numpy() if "n_repair" in df_va.columns else \
           df_va["Кол-во_ВЭУ_в_ремонте"].to_numpy()
    report("n_repair = 0 (full fleet)", nrep == 0)
    report("n_repair = 1",              nrep == 1)
    report("n_repair = 2",              nrep == 2)
    report("n_repair >= 3",             nrep >= 3)

    print()
    print("─" * 72)
    print(" Top-20 worst hours (largest |error|):")
    print("─" * 72)
    worst_idx = np.argsort(-abs_err)[:20]
    for i in worst_idx:
        ts_i = pd.Timestamp(ts_va[i])
        print(f"  {ts_i}  ws={ws[i]:5.2f}  rho={df_va['air_density'].to_numpy()[i]:5.3f}  "
              f"y_true={y_va_mw[i]:6.2f}  pred={blend_mw[i]:6.2f}  "
              f"err={(blend_mw[i]-y_va_mw[i]):+6.2f}  n_rep={int(nrep[i])}")


if __name__ == "__main__":
    main()
