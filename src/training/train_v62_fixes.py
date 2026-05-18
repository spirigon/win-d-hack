"""V62: Three targeted fixes on top of V34.1 (LB 7.56).

Fix #1 — Availability leak (intra-fold)
    ``add_walk_forward_availability`` now receives ``freeze_after_ts=fold.train_end``
    so validation rows can't write underprod observations into the (month,
    hour) table that later validation rows of the same fold read.
    Expected effect: small Fold-5 *worsening* of OOF (it was optimistically
    leaky before) but a *truer* optimization target → better LB alignment.

Fix #2 — Targeted 7-12 m/s specialist
    Adds a fourth regime booster trained on the residual after the isotonic
    PC baseline (same framing as v50) but ONLY for the 7-12 m/s band where
    56% of the error lives.  Outside 7-12 m/s this booster contributes with
    weight 0.0 (not 0.1) — it's strictly the mid-wind residual.
    The final prediction = v34.1-style blend + alpha * (mid-wind residual).

Fix #3 — Per-regime affine calibration on Fold-5 OOF
    Five wind-speed regimes.  Fit (a, b) via least-squares:
        a * pred + b ≈ target
    on the Fold-5 OOF rows belonging to each regime, then apply at inference.
    Risk is low — only 10 parameters, fit on the same Q1 surrogate.

Outputs:
    submissions/archive/v62.0_three_fixes.csv

Usage:
    python -m src.training.train_v62_fixes --seeds 3
    python -m src.training.train_v62_fixes --seeds 5
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
from src.features.availability import add_walk_forward_availability
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.era5_v2 import era5v2_columns, merge_era5_v2
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, K, BLEND_WEIGHT_MW, FOLD_IDS, SEEDS_3, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH  = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v62.0_three_fixes.csv"


# ─── Fix #2: mid-wind residual specialist params ──────────────────────────
MID_WIND_LO = 7.0
MID_WIND_HI = 12.0
MID_WIND_ALPHA = 0.25      # blend weight: final = base + alpha * mid_resid
MID_WIND_WEIGHT = 3.0      # inside [7,12]; 0.0 outside


def _iso_pc_mw(df: pd.DataFrame) -> np.ndarray:
    """Farm-level iso-PC prediction in MW (already includes availability)."""
    return np.clip(df["p_curve_global"].to_numpy(), 0.0, CAPACITY_MW)


def _train_fold_ensemble(
    X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sample_weight, seeds,
):
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [
        ("low_0_7",   (0,   7)),
        ("mid_4_12",  (4,  12)),
        ("high_8_25", (8,  25)),
    ]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        vp, tp = [], []
        for s in seeds:
            cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights,
                             feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va,
                             feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=LGBM_PARAMS.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(LGBM_PARAMS.early_stopping_rounds, verbose=False)],
            )
            vp.append(b.predict(X_va, num_iteration=b.best_iteration))
            tp.append(b.predict(X_test, num_iteration=b.best_iteration))
        regime_val[name]  = np.mean(vp, axis=0)
        regime_test[name] = np.mean(tp, axis=0)
    return (
        np.mean(list(regime_val.values()), axis=0),
        np.mean(list(regime_test.values()), axis=0),
    )


def _train_midwind_residual_specialist(
    X_tr, resid_tr, X_va, resid_va, X_test, feat_cols, ws_tr, sample_weight, seeds,
):
    """Fix #2: booster trained on iso-PC residual, weighted 3× in 7-12 m/s."""
    mask_in = (ws_tr >= MID_WIND_LO) & (ws_tr < MID_WIND_HI)
    weights = (np.where(mask_in, MID_WIND_WEIGHT, 0.0) * sample_weight).astype(np.float32)
    vp, tp = [], []
    for s in seeds:
        cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
        dt = lgb.Dataset(X_tr, label=resid_tr, weight=weights,
                         feature_name=feat_cols, free_raw_data=False)
        dv = lgb.Dataset(X_va, label=resid_va,
                         feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(
            cfg.to_params(), dt, num_boost_round=LGBM_PARAMS.num_boost_round,
            valid_sets=[dv], valid_names=["val"],
            callbacks=[lgb.early_stopping(LGBM_PARAMS.early_stopping_rounds, verbose=False)],
        )
        vp.append(b.predict(X_va, num_iteration=b.best_iteration))
        tp.append(b.predict(X_test, num_iteration=b.best_iteration))
    return np.mean(vp, axis=0), np.mean(tp, axis=0)


# ─── Fix #3: per-regime affine calibration ────────────────────────────────

AFFINE_REGIMES = [
    ("sub3",   0.0,  3.0),
    ("mid3_7", 3.0,  7.0),
    ("mid7_12",7.0, 12.0),
    ("rated",  12.0,17.0),
    ("high",   17.0,99.0),
]


def _fit_affine_per_regime(
    oof_pred: np.ndarray,
    oof_target: np.ndarray,
    oof_ws: np.ndarray,
) -> dict:
    """Fit 2-parameter (a, b) affine per wind regime on OOF rows."""
    params = {}
    for name, lo, hi in AFFINE_REGIMES:
        mask = (oof_ws >= lo) & (oof_ws < hi)
        if mask.sum() < 30:
            params[name] = (1.0, 0.0)
            continue
        p = oof_pred[mask]
        y = oof_target[mask]
        # Least squares: [p 1] @ [a; b] ≈ y
        A = np.column_stack([p, np.ones_like(p)])
        try:
            result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = float(result[0]), float(result[1])
            # Safety: don't let the cal flip or over-shrink predictions.
            a = float(np.clip(a, 0.7, 1.3))
            b = float(np.clip(b, -10.0, 10.0))
        except np.linalg.LinAlgError:
            a, b = 1.0, 0.0
        params[name] = (a, b)
        print(f"    regime {name:<10s}: a={a:.4f}  b={b:+.3f} MW  "
              f"n={mask.sum()}  before_nMAE={normalized_mae(y, p):.4f}%  "
              f"after_nMAE={normalized_mae(y, np.clip(a*p+b, 0, CAPACITY_MW)):.4f}%")
    return params


def _apply_affine_per_regime(
    pred: np.ndarray,
    ws: np.ndarray,
    params: dict,
) -> np.ndarray:
    out = pred.copy().astype(np.float64)
    for name, lo, hi in AFFINE_REGIMES:
        mask = (ws >= lo) & (ws < hi)
        a, b = params[name]
        out[mask] = np.clip(a * pred[mask] + b, 0.0, CAPACITY_MW)
    return out


# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["3", "5"], default="3")
    args = ap.parse_args()
    seeds = SEEDS_5 if args.seeds == "5" else SEEDS_3
    set_global_seed(42)

    print("=" * 72)
    print(f"V62: V34.1 + 3 fixes (leak, mid-wind specialist, affine cal)  "
          f"({len(seeds)} seeds)")
    print("=" * 72)

    # ── Build features ───────────────────────────────────────────────
    print("\n[1/5] Loading + building features...")
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

    df_train_full_pre = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full_pre)
    is_impossible_full = pd.Series(False, index=combined.index)
    is_impossible_full.iloc[: len(df_train_full_pre)] = impossible.values
    combined["_is_impossible"] = is_impossible_full.values

    train_end = df_train_full_pre[TIMESTAMP_COL].max()

    # NOTE: We do NOT compute the global availability here. The per-fold
    # loop below recomputes it with freeze_after_ts=fold.train_end for
    # honest cross-validation (Fix #1). The probe at step [2/5] also
    # recomputes with freeze_after_ts=fold5.train_end.
    # We keep avail_underprod_mw as NaN in combined for now.
    df_train_full = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    sample_weight_full = (~df_train_full["_is_impossible"].to_numpy()).astype(np.float32)

    # ── Probe → top-K ────────────────────────────────────────────────
    print("[2/5] Feature selection (probe on Fold-5 with leak fix)...")
    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train_full, fold5)
    fold_train_ = df_train_full.iloc[train_idx]

    # Recompute availability for Fold-5 probe WITH the freeze fix.
    # We need the combined frame at this fold's split.
    probe_combined = combined.copy()
    probe_train_mask = probe_combined[TIMESTAMP_COL] <= fold5.train_end
    probe_combined.loc[~probe_train_mask, "avail_underprod_mw"] = np.nan
    # Recompute cumsum/rolling with freeze.
    probe_combined = add_walk_forward_availability(
        probe_combined, train_end=fold5.train_end,
        wind_col="wind_speed_120m",
        impossible_col="_is_impossible",
        window_days=30,
        freeze_after_ts=fold5.train_end,
    )
    df_train_probe = probe_combined[probe_combined["_split"] == "train"].reset_index(drop=True)
    fold_train_probe = df_train_probe.iloc[train_idx]
    fit_data_ = fold_train_probe[~fold_train_probe["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)
    df_t_ = _add_pc(fold_train_probe, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = _add_pc(df_train_probe.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)

    feat_cols_all = [
        c for c in feature_columns(df_t_)
        if c not in ("_is_impossible", "_split", TARGET_COL)
        and c != "avail_underprod_mw"
    ]
    print(f"  Feature pool: {len(feat_cols_all)}  "
          f"({len(era5v2_columns(df_t_))} era5v2_*, "
          f"{sum(1 for c in feat_cols_all if c.startswith('avail_'))} avail_*)")
    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = _to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = _to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    sw_fold = sample_weight_full[train_idx]
    X_t_ = df_t_[feat_cols_all].to_numpy(dtype=np.float32)
    X_v_ = df_v_[feat_cols_all].to_numpy(dtype=np.float32)
    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(X_t_, label=y_t_cf_, weight=sw_fold,
                      feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_v_, label=y_v_cf_, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    # ── CV-bag ──────────────────────────────────────────────────────
    print(f"\n[3/5] Training CV-bag (folds {[i + 1 for i in FOLD_IDS]}, "
          f"seeds={seeds})...")
    test_cf_per_fold = {}
    test_mw_per_fold = {}
    test_midresid_per_fold = {}
    oof_records: list[dict] = []

    for fold_idx in FOLD_IDS:
        t0 = time.time()
        fold = folds[fold_idx]

        # ── Fix #1: recompute availability with freeze at fold boundary.
        fold_combined = combined.copy()
        fold_combined = add_walk_forward_availability(
            fold_combined, train_end=fold.train_end,
            wind_col="wind_speed_120m",
            impossible_col="_is_impossible",
            window_days=30,
            freeze_after_ts=fold.train_end,
        )
        fold_df_train = fold_combined[fold_combined["_split"] == "train"].reset_index(drop=True)
        fold_df_valid = fold_combined[fold_combined["_split"] == "valid"].reset_index(drop=True)

        train_idx, val_idx = split_indices(fold_df_train, fold)
        fold_train = fold_df_train.iloc[train_idx]
        fold_val   = fold_df_train.iloc[val_idx]
        fit_data   = fold_train[~fold_train["_is_impossible"]]

        pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        wake      = fit_wake_lookup(fit_data, n_sectors=16)

        df_tr = _add_pc(fold_train, pc_sector, pc_global)
        df_tr = add_wake_features(df_tr, wake)
        df_va = _add_pc(fold_val, pc_sector, pc_global)
        df_va = add_wake_features(df_va, wake)
        df_te = _add_pc(fold_df_valid, pc_sector, pc_global)
        df_te = add_wake_features(df_te, wake)
        for c in set(top_k) - set(df_te.columns):
            df_te[c] = 0.0

        active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
        active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
        y_tr_mw   = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_mw   = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        X_tr  = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va  = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr = df_tr["wind_speed_120m"].to_numpy()
        sw    = sample_weight_full[train_idx]

        # CF + MW standard specialists.
        cf_pred_va, cf_pred_te = _train_fold_ensemble(
            X_tr, _to_cf(y_tr_mw, active_tr),
            X_va, _to_cf(y_va_mw, active_va),
            X_test, top_k, ws_tr, sw, seeds,
        )
        mw_pred_va, mw_pred_te = _train_fold_ensemble(
            X_tr, y_tr_mw,
            X_va, y_va_mw,
            X_test, top_k, ws_tr, sw, seeds,
        )

        # ── Fix #2: mid-wind residual specialist ─────────────────────
        iso_pc_tr = _iso_pc_mw(df_tr)
        iso_pc_va = _iso_pc_mw(df_va)
        iso_pc_te = _iso_pc_mw(df_te)
        resid_tr  = (y_tr_mw - iso_pc_tr).astype(np.float32)
        resid_va  = (y_va_mw - iso_pc_va).astype(np.float32)
        mid_pred_va, mid_pred_te = _train_midwind_residual_specialist(
            X_tr, resid_tr, X_va, resid_va, X_test, top_k, ws_tr, sw, seeds,
        )

        # Reconstruct MW predictions.
        val_cf_mw    = np.clip(_from_cf(cf_pred_va, active_va), 0, CAPACITY_MW)
        val_mw_mw    = np.clip(mw_pred_va, 0, CAPACITY_MW)
        val_base     = BLEND_WEIGHT_MW * val_mw_mw + (1 - BLEND_WEIGHT_MW) * val_cf_mw
        val_midresid = np.clip(iso_pc_va + mid_pred_va, 0, CAPACITY_MW)
        val_final    = np.clip(val_base + MID_WIND_ALPHA * (val_midresid - val_base), 0, CAPACITY_MW)

        test_cf_per_fold[fold_idx] = cf_pred_te
        test_mw_per_fold[fold_idx] = mw_pred_te
        test_midresid_per_fold[fold_idx] = iso_pc_te + mid_pred_te

        fold_nmae = float(normalized_mae(y_va_mw, val_final))
        print(f"    Fold {fold_idx + 1}: nMAE={fold_nmae:.4f}%  "
              f"(base={normalized_mae(y_va_mw, val_base):.4f}%)  "
              f"({time.time() - t0:.0f}s)")

        ts_va = df_va[TIMESTAMP_COL].to_numpy()
        ws_va = df_va["wind_speed_120m"].to_numpy()
        for i in range(len(y_va_mw)):
            oof_records.append({
                "fold": int(fold_idx + 1),
                "ts": pd.Timestamp(ts_va[i]),
                "target_mw": float(y_va_mw[i]),
                "active_turbines": float(active_va[i]),
                "ws_120": float(ws_va[i]),
                "pred_blend_mw": float(val_final[i]),
                "pred_base_mw": float(val_base[i]),
            })

    oof_df = pd.DataFrame(oof_records)
    oof_path = _ROOT / "data" / "processed" / "v62_oof.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(oof_path, index=False)
    print(f"\n  OOF saved ({len(oof_df)} rows)")
    for fid in [3, 4, 5]:
        sub = oof_df[oof_df["fold"] == fid]
        n = float(normalized_mae(sub["target_mw"].to_numpy(),
                                 sub["pred_blend_mw"].to_numpy()))
        print(f"    Fold {fid} blend nMAE: {n:.4f}%")

    # ── Fix #3: per-regime affine calibration on Fold-5 OOF ─────────
    print("\n[4/5] Per-regime affine calibration (Fold-5 OOF)...")
    f5_oof = oof_df[oof_df["fold"] == 5]
    affine_params = _fit_affine_per_regime(
        f5_oof["pred_blend_mw"].to_numpy(),
        f5_oof["target_mw"].to_numpy(),
        f5_oof["ws_120"].to_numpy(),
    )

    # ── Final submission ────────────────────────────────────────────
    print("\n[5/5] Building final submission...")
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    ws_valid     = df_valid_sorted["wind_speed_120m"].to_numpy()

    avg_test_cf      = np.mean([test_cf_per_fold[i]      for i in FOLD_IDS], axis=0)
    avg_test_mw      = np.mean([test_mw_per_fold[i]      for i in FOLD_IDS], axis=0)
    avg_test_midresid= np.mean([test_midresid_per_fold[i] for i in FOLD_IDS], axis=0)

    pred_cf_mw = np.clip(_from_cf(avg_test_cf, active_valid), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(avg_test_mw, 0, CAPACITY_MW)
    base_mw    = BLEND_WEIGHT_MW * pred_mw_mw + (1 - BLEND_WEIGHT_MW) * pred_cf_mw
    mid_mw     = np.clip(avg_test_midresid, 0, CAPACITY_MW)
    final_mw   = np.clip(base_mw + MID_WIND_ALPHA * (mid_mw - base_mw), 0, CAPACITY_MW)

    # Apply per-regime affine calibration.
    final_mw = _apply_affine_per_regime(final_mw, ws_valid, affine_params)
    final_mw = np.clip(final_mw, 0.0, CAPACITY_MW).astype(np.float64)

    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    ts_po  = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"\n  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   "
          f"Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
