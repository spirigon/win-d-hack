"""V50: Residual learning on isotonic-PC baseline.

Diagnostic finding (scripts/diagnose_residual_structure.py):

    target std on Fold-5      : 26.20 MW
    iso_PC residual std       : 12.70 MW   (76.5 % of variance explained by PC)
    iso_PC alone nMAE Fold-5  : 11.65 %
    v32 full-pipeline nMAE    :  7.54 %
    iso_PC + 1 feature (era5_wind_speed_84m): 8.78 %

The booster spends most of its capacity re-learning the power curve from
scratch on every split. By targeting the **residual after the fold-fitted
isotonic PC** directly, we let LightGBM focus exclusively on the 13 MW-std
problem (NWP wind bias, wake, curtailment, icing) — the part that's
genuinely orthogonal to the physics it can't already extract.

Pipeline (one target leg, no mw/cf blend):

    1. Build features as in v34 (era5v2 + availability).
    2. Per fold: fit isotonic on v_eff (training rows only).
    3. Compute baseline `iso_pc_mw = isotonic(v_eff) × active/26`.
    4. Train LightGBM with target = `(target_mw - iso_pc_mw)`.
    5. Final prediction = `iso_pc_mw + booster_residual`, clipped to [0, 90.09].

Sample weighting: existing impossible-row mask is preserved.

Outputs:
    data/processed/v50_oof.parquet
    data/processed/v50_test.parquet
    submissions/archive/v50.0_residual.csv

Usage:
    python -m src.training.train_v50_residual --seeds 5
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
    LGBM_PARAMS, K, FOLD_IDS, SEEDS_3, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling,
)

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_PATH = _ROOT / "data" / "processed" / "v50_oof.parquet"
TEST_PATH = _ROOT / "data" / "processed" / "v50_test.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v50.0_residual.csv"


def _iso_pc_mw(df: pd.DataFrame) -> np.ndarray:
    """Iso-PC baseline in MW, per row.

    ``p_curve_global`` is fitted by ``IsotonicPowerCurve().fit(v_eff, target_mw)``,
    so it returns farm-level MW directly using the same operating-mix
    distribution seen during training. We don't re-multiply by the
    operating fraction (would be double-discounting).
    """
    return np.clip(df["p_curve_global"].to_numpy(), 0.0, CAPACITY_MW)


def _train_specialists_residual(
    X_tr, resid_tr, X_va, resid_va, feat_cols, ws_tr, sample_weight, seeds,
):
    """3 specialists × N seeds, target = residual_mw. Returns (val_resid, test_resid)."""
    regime_val = {}
    regime_test_args: dict = {}  # filled at call site

    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        seed_preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=resid_tr, weight=weights,
                             feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=resid_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=LGBM_PARAMS.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(LGBM_PARAMS.early_stopping_rounds, verbose=False)],
            )
            seed_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_val[name] = (np.mean(seed_preds, axis=0), [s for s in seeds])  # store per-regime
    return regime_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["3", "5"], default="5")
    args = ap.parse_args()
    seeds = SEEDS_5 if args.seeds == "5" else SEEDS_3
    set_global_seed(42)

    print("=" * 72)
    print(f"V50: Residual learning on iso-PC ({len(seeds)} seeds)")
    print("=" * 72)

    # --- Load + build features ------------------------------------------
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
    combined = merge_era5_v2(combined)

    df_train_full_pre = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full_pre)
    is_impossible_full = pd.Series(False, index=combined.index)
    is_impossible_full.iloc[: len(df_train_full_pre)] = impossible.values
    combined["_is_impossible"] = is_impossible_full.values

    train_end = df_train_full_pre[TIMESTAMP_COL].max()
    print(f"  Adding walk-forward availability (train_end={train_end})...")
    combined = add_walk_forward_availability(
        combined, train_end=train_end,
        wind_col="wind_speed_120m",
        impossible_col="_is_impossible",
        window_days=30,
    )

    df_train_full = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    sample_weight_full = (~df_train_full["_is_impossible"].to_numpy()).astype(np.float32)

    # --- Probe → top-K with RESIDUAL target ----------------------------
    print("[3/4] Feature selection (probe on Fold-5 residual)...")
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
        and c != "avail_underprod_mw"
    ]
    print(f"  Feature pool: {len(feat_cols_all)}  ({len(era5v2_columns(df_t_))} era5v2_*)")

    # Residual target: y - iso_PC_mw.
    iso_pc_tr = _iso_pc_mw(df_t_)
    iso_pc_va = _iso_pc_mw(df_v_)
    y_t_mw = df_t_[TARGET_COL].to_numpy(dtype=np.float64)
    y_v_mw = df_v_[TARGET_COL].to_numpy(dtype=np.float64)
    resid_t = (y_t_mw - iso_pc_tr).astype(np.float32)
    resid_v = (y_v_mw - iso_pc_va).astype(np.float32)
    print(f"  residual_train: mean={resid_t.mean():.3f}, std={resid_t.std():.3f}")
    print(f"  residual_valid: mean={resid_v.mean():.3f}, std={resid_v.std():.3f}")
    print(f"  iso_PC alone Fold-5 nMAE: {float(normalized_mae(y_v_mw, iso_pc_va)):.4f} %")

    X_t_ = df_t_[feat_cols_all].to_numpy(dtype=np.float32)
    X_v_ = df_v_[feat_cols_all].to_numpy(dtype=np.float32)
    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(X_t_, label=resid_t, feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_v_, label=resid_v, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    n_era5v2_top = sum(1 for n in top_k if n.startswith("era5v2_"))
    n_avail_top = sum(1 for n in top_k if n.startswith("avail_"))
    print(f"  Top-{K}: {n_era5v2_top} era5v2_*, {n_avail_top} avail_*")
    print(f"  Top-10 by gain:")
    for n, g in feat_imp[:10]:
        print(f"    {n:<35s}  {g:>9,.0f}")

    # --- CV-bag training: target = residual ----------------------------
    print(f"\n[4/4] Training CV-bag (folds {[i + 1 for i in FOLD_IDS]}, seeds={seeds})...")
    test_per_fold = {}
    oof_records: list[dict] = []

    for fold_idx in FOLD_IDS:
        t0 = time.time()
        fold = folds[fold_idx]
        train_idx, val_idx = split_indices(df_train_full, fold)
        fold_train = df_train_full.iloc[train_idx]
        fold_val = df_train_full.iloc[val_idx]
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

        # Residual targets per fold.
        iso_pc_tr = _iso_pc_mw(df_tr)
        iso_pc_va = _iso_pc_mw(df_va)
        iso_pc_te = _iso_pc_mw(df_te)
        y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float64)
        y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float64)
        resid_tr = (y_tr_mw - iso_pc_tr).astype(np.float32)
        resid_va = (y_va_mw - iso_pc_va).astype(np.float32)

        X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr = df_tr["wind_speed_120m"].to_numpy()
        sample_weight_fold = sample_weight_full[train_idx]

        # Train 3 specialists × seeds.
        regime_val = {}
        regime_test = {}
        for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
            mask_in = (ws_tr >= lo) & (ws_tr < hi)
            regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
            weights = (regime_w * sample_weight_fold).astype(np.float32)
            val_preds, test_preds = [], []
            for s in seeds:
                cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
                dt = lgb.Dataset(X_tr, label=resid_tr, weight=weights,
                                 feature_name=top_k, free_raw_data=False)
                dv = lgb.Dataset(X_va, label=resid_va, feature_name=top_k, free_raw_data=False)
                b = lgb.train(
                    cfg.to_params(), dt, num_boost_round=LGBM_PARAMS.num_boost_round,
                    valid_sets=[dv], valid_names=["val"],
                    callbacks=[lgb.early_stopping(LGBM_PARAMS.early_stopping_rounds, verbose=False)],
                )
                val_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
                test_preds.append(b.predict(X_test, num_iteration=b.best_iteration))
            regime_val[name] = np.mean(val_preds, axis=0)
            regime_test[name] = np.mean(test_preds, axis=0)

        avg_resid_va = np.mean(list(regime_val.values()), axis=0)
        avg_resid_te = np.mean(list(regime_test.values()), axis=0)
        # Reconstruct MW: iso_PC + residual, clipped.
        pred_mw_va = np.clip(iso_pc_va + avg_resid_va, 0.0, CAPACITY_MW)
        pred_mw_te = np.clip(iso_pc_te + avg_resid_te, 0.0, CAPACITY_MW)
        test_per_fold[fold_idx] = pred_mw_te

        fold_nmae = float(normalized_mae(y_va_mw, pred_mw_va))
        print(f"    Fold {fold_idx + 1}: nMAE={fold_nmae:.4f}%  ({time.time() - t0:.0f}s)")

        ts_va = df_va[TIMESTAMP_COL].to_numpy()
        for i in range(len(y_va_mw)):
            oof_records.append({
                "fold": int(fold_idx + 1),
                "ts": pd.Timestamp(ts_va[i]),
                "target_mw": float(y_va_mw[i]),
                "iso_pc_mw": float(iso_pc_va[i]),
                "pred_resid_mw": float(avg_resid_va[i]),
                "pred_total_mw": float(pred_mw_va[i]),
                "ws_120": float(df_va["wind_speed_120m"].to_numpy()[i]),
                "active_turbines": float(df_va["active_turbines"].to_numpy()[i]),
            })

    # --- OOF + test parquet ---------------------------------------------
    oof_df = pd.DataFrame(oof_records)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH} ({len(oof_df)} rows)")
    print(f"  Mean OOF nMAE across folds: "
          f"{np.mean([float(normalized_mae(g['target_mw'].to_numpy(), g['pred_total_mw'].to_numpy())) for _, g in oof_df.groupby('fold')]):.4f}%")

    test_df = pd.DataFrame({
        TIMESTAMP_COL: df_valid_sorted[TIMESTAMP_COL].to_numpy(),
        "_submission_row": df_valid_sorted["_submission_row"].to_numpy().astype(int),
        "active_turbines": df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32),
    })
    for fold_idx in FOLD_IDS:
        test_df[f"test_total_fold{fold_idx + 1}"] = test_per_fold[fold_idx]
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"  Per-fold test preds saved: {TEST_PATH}")

    # --- Final submission: average across CV-bag folds ------------------
    final_mw = np.mean([test_per_fold[i] for i in FOLD_IDS], axis=0)
    final_mw = np.clip(final_mw, 0.0, CAPACITY_MW).astype(np.float64)

    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    ts_po = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"\n  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW  Std: {final_mw.std():.2f} MW  "
          f"Range: [{final_mw.min():.2f}, {final_mw.max():.2f}]")


if __name__ == "__main__":
    main()
