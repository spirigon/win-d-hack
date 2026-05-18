"""V78: V76 features - power lags + winter weighting + recency fold blend.

Lessons from v76/v77: power lag features consistently hurt the leaderboard
(7.43% -> 7.66% -> 7.86%) despite improving OOF.  Root cause: the test period
is a contiguous future window with no actual power data.  Lag chains built
from model predictions compound errors over 2126 hours.  OOF was misleading
because hold-out rows are adjacent to actual training data.

Changes from V75 (7.43% LB):
    + turbulence features (TI proxies, ensemble rank, WPD, regime features)
    + curtailment-aware sample weights (already in v76, physically sound)
    + January x1.5 weighting  (test period is Q1 2026 - Jan-Mar are hardest)
    + February x1.5 weighting (same rationale)
    + Recency-weighted fold blend: fold3 x0.20, fold4 x0.30, fold5 x0.50
      (fold 5 trained on most data, most temporally close to test)
    - power lag features removed (hurt LB despite OOF gains)
    = K reverted to 80 (proven; K=120 was only justified by lag feature count)

Outputs:
    data/processed/v78_oof.parquet
    data/processed/v78_test.parquet
    submissions/archive/v78.0_season_blend.csv

Usage:
    python -m src.training.train_v78_season_blend --seeds 3
    python -m src.training.train_v78_season_blend --seeds 5
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

from src.data.curtailment_mask import CurtailmentConfig, identify_curtailment_rows, summarize_curtailment
from src.data.loaders import load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.advanced_interactions import add_all_advanced_features, advanced_interaction_columns
from src.features.availability import add_walk_forward_availability
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.era5_v2 import merge_era5_v2
from src.features.extras import add_extra_features
from src.features.multi_nwp_features import build_nwp_consensus, merge_ecmwf_ifs, merge_gfs, multi_nwp_columns
from src.features.nasa_features import merge_nasa_merra2, nasa_columns
from src.features.nwp_ensemble import merge_nwp_ensemble, nwp_ensemble_columns
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.turbulence_features import add_all_turbulence_features, turbulence_columns
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, FOLD_IDS, SEEDS_3, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH  = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH  = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH   = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_PATH    = _ROOT / "data" / "processed" / "v78_oof.parquet"
TEST_PATH   = _ROOT / "data" / "processed" / "v78_test.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v78.0_season_blend.csv"

K             = 80
BLEND_WEIGHT_MW = 0.0  # CF-only

# Sample weight multipliers by month
MONTH_WEIGHTS = {3: 3.0, 1: 1.5, 2: 1.5}  # March x3, Jan/Feb x1.5

# Recency-weighted fold blend: fold5 (most recent, most data) gets 50%
FOLD_BLEND = {2: 0.20, 3: 0.30, 4: 0.50}  # keys are 0-indexed fold IDs


def _train_fold_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sw, seeds):
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = (np.where(mask, 2.0, 0.3) * sw).astype(np.float32)
        vp, tp = [], []
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
            vp.append(b.predict(X_va, num_iteration=b.best_iteration))
            tp.append(b.predict(X_test, num_iteration=b.best_iteration))
        regime_val[name]  = np.mean(vp, axis=0)
        regime_test[name] = np.mean(tp, axis=0)
    return (
        np.mean(list(regime_val.values()), axis=0),
        np.mean(list(regime_test.values()), axis=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["3", "5"], default="5")
    args = ap.parse_args()
    seeds = SEEDS_5 if args.seeds == "5" else SEEDS_3
    set_global_seed(42)

    month_desc = ", ".join(f"Month {m} x{w}" for m, w in sorted(MONTH_WEIGHTS.items()))
    blend_desc = " + ".join(f"fold{i+1} x{w}" for i, w in sorted(FOLD_BLEND.items()))
    print("=" * 72)
    print(f"V78: V76 features - lags + winter weights + recency blend  ({len(seeds)} seeds)")
    print(f"     {month_desc}")
    print(f"     Blend: {blend_desc}")
    print("=" * 72)

    print("\n[1/4] Loading + building features...")
    df_train = _load_train_raw(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # --- Feature pipeline (same as v76, no lags) ---
    combined = build_features(combined, sort_by_time=False)
    combined = _merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = _add_era5_rolling(combined)
    combined = merge_era5_v2(combined)
    combined = merge_nwp_ensemble(combined)
    combined = merge_nasa_merra2(combined)
    combined = merge_gfs(combined)
    combined = merge_ecmwf_ifs(combined)
    combined = build_nwp_consensus(combined)
    combined = add_all_advanced_features(combined)
    combined = add_all_turbulence_features(combined)

    nwp_cols  = nwp_ensemble_columns(combined)
    nasa_cols = nasa_columns(combined)
    mnwp_cols = multi_nwp_columns(combined)
    adv_cols  = advanced_interaction_columns(combined)
    turb_cols = turbulence_columns(combined)

    print(f"  Feature groups: nwp={len(nwp_cols)} nasa={len(nasa_cols)} "
          f"mnwp={len(mnwp_cols)} adv={len(adv_cols)} turb={len(turb_cols)}")

    # --- Impossible + curtailment mask ---
    df_train_full_pre = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full_pre)

    curtailment_cfg = CurtailmentConfig(
        min_wind_ms=5.0, fraction_threshold=0.4,
        min_expected_mw=15.0, min_run_length_hours=3,
    )
    curtailment_mask, curtailment_diag = identify_curtailment_rows(
        df_train_full_pre,
        cfg=curtailment_cfg,
        pre_existing_impossible=impossible,
    )
    curt_summary = summarize_curtailment(curtailment_diag)
    print(f"  Curtailment flagged: {curt_summary['post_run_flagged']} rows "
          f"({curt_summary['pct_flagged']:.1f}%)  "
          f"mean_expected={curt_summary['mean_expected_on_flagged']:.1f}MW  "
          f"mean_actual={curt_summary['mean_target_on_flagged']:.1f}MW")

    combined_impossible = impossible | curtailment_mask

    is_impossible_full = pd.Series(False, index=combined.index)
    is_impossible_full.iloc[: len(df_train_full_pre)] = combined_impossible.values
    combined["_is_impossible"] = is_impossible_full.values

    train_end = df_train_full_pre[TIMESTAMP_COL].max()

    # Sample weights: zero for impossible/curtailment, month multipliers
    sample_weight_full = (~combined_impossible.to_numpy()).astype(np.float32)
    for month, mult in MONTH_WEIGHTS.items():
        mask = df_train_full_pre[TIMESTAMP_COL].dt.month == month
        sample_weight_full[mask.to_numpy()] *= mult
        n = int(mask.sum())
        print(f"  Month {month:2d} rows x{mult}: {n} ({100*n/len(df_train_full_pre):.1f}%)")

    n_impossible = int(combined_impossible.sum())
    total = len(df_train_full_pre)
    print(f"  Zero-weight rows (impossible + curtailment): {n_impossible} ({100*n_impossible/total:.1f}%)")

    # --- Feature selection probe on Fold-5 ---
    print(f"[2/4] Feature selection probe (Fold-5, K={K})...")
    folds = default_folds()
    fold5 = folds[-1]

    probe_combined = add_walk_forward_availability(
        combined, train_end=fold5.train_end,
        wind_col="wind_speed_120m",
        impossible_col="_is_impossible",
        window_days=30,
        freeze_after_ts=fold5.train_end,
    )
    probe_train = probe_combined[probe_combined["_split"] == "train"].reset_index(drop=True)

    train_idx, val_idx = split_indices(probe_train, fold5)
    fold_train_ = probe_train.iloc[train_idx]
    fit_data_ = fold_train_[~fold_train_["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)
    df_t_ = _add_pc(fold_train_, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = _add_pc(probe_train.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)

    all_extra = list(dict.fromkeys(nwp_cols + nasa_cols + mnwp_cols + adv_cols + turb_cols))
    feat_cols_all = [
        c for c in feature_columns(df_t_)
        if c not in ("_is_impossible", "_split", TARGET_COL)
        and c != "avail_underprod_mw"
    ]
    for col in all_extra:
        if col in df_t_.columns and col not in feat_cols_all:
            feat_cols_all.append(col)

    print(f"  Feature pool: {len(feat_cols_all)}  (K={K} will be selected by gain)")

    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = _to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = _to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    sw_fold = sample_weight_full[train_idx]

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(df_t_[feat_cols_all].to_numpy(dtype=np.float32),
                      label=y_t_cf_, weight=sw_fold,
                      feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(df_v_[feat_cols_all].to_numpy(dtype=np.float32),
                      label=y_v_cf_, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    for group, cols in [
        ("turbulence", turb_cols), ("GFS/ECMWF/consensus", mnwp_cols),
        ("NASA", nasa_cols), ("advanced", adv_cols),
    ]:
        in_topk = [c for c in top_k if c in cols]
        print(f"  {group} in top-{K}: {len(in_topk)}  {in_topk[:5]}")

    # --- CV-bag training ---
    print(f"\n[3/4] Training CV-bag (folds {[i+1 for i in FOLD_IDS]}, CF-only, seeds={seeds})...")
    test_cf_per_fold: dict[int, np.ndarray] = {}
    oof_records: list[dict] = []

    for fold_idx in FOLD_IDS:
        t0 = time.time()
        fold = folds[fold_idx]

        fc = add_walk_forward_availability(
            combined, train_end=fold.train_end,
            wind_col="wind_speed_120m",
            impossible_col="_is_impossible",
            window_days=30,
            freeze_after_ts=fold.train_end,
        )
        ft = fc[fc["_split"] == "train"].reset_index(drop=True)
        fv = fc[fc["_split"] == "valid"].reset_index(drop=True)

        train_idx, val_idx = split_indices(ft, fold)
        fold_train = ft.iloc[train_idx]
        fold_val   = ft.iloc[val_idx]
        fit_data   = fold_train[~fold_train["_is_impossible"]]
        pc_sector  = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global  = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        wake       = fit_wake_lookup(fit_data, n_sectors=16)
        df_tr = _add_pc(fold_train, pc_sector, pc_global)
        df_tr = add_wake_features(df_tr, wake)
        df_va = _add_pc(fold_val, pc_sector, pc_global)
        df_va = add_wake_features(df_va, wake)
        df_te = _add_pc(fv, pc_sector, pc_global)
        df_te = add_wake_features(df_te, wake)
        for c in set(top_k) - set(df_te.columns):
            df_te[c] = 0.0

        active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
        active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
        y_tr_mw   = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_mw   = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        y_tr = _to_cf(y_tr_mw, active_tr)
        y_va = _to_cf(y_va_mw, active_va)
        X_tr   = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va   = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr  = df_tr["wind_speed_120m"].to_numpy()
        sw     = sample_weight_full[train_idx]

        val_cf_pred, test_cf_pred = _train_fold_ensemble(
            X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, sw, seeds,
        )
        test_cf_per_fold[fold_idx] = test_cf_pred
        val_mw = np.clip(_from_cf(val_cf_pred, active_va), 0, CAPACITY_MW)

        fold_nmae = float(normalized_mae(y_va_mw, val_mw))
        print(f"  Fold {fold_idx + 1}: CF nMAE={fold_nmae:.4f}%  ({time.time() - t0:.0f}s)")

        ts_va = df_va[TIMESTAMP_COL].to_numpy()
        for i in range(len(y_va_mw)):
            oof_records.append({
                "fold": int(fold_idx + 1),
                "ts": pd.Timestamp(ts_va[i]),
                "target_mw": float(y_va_mw[i]),
                "active_turbines": float(active_va[i]),
                "ws_120": float(df_va["wind_speed_120m"].to_numpy()[i]),
                "pred_cf": float(val_cf_pred[i]),
            })

    oof_df = pd.DataFrame(oof_records)
    oof_df["pred_cf_mw"] = np.clip(
        _from_cf(oof_df["pred_cf"].to_numpy(), oof_df["active_turbines"].to_numpy()),
        0, CAPACITY_MW,
    )
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")
    for fid in [3, 4, 5]:
        sub = oof_df[oof_df.fold == fid]
        n = float(normalized_mae(sub.target_mw.values, sub.pred_cf_mw.values))
        print(f"    Fold {fid} CF nMAE: {n:.4f}%")
    all_n = float(normalized_mae(oof_df.target_mw.values, oof_df.pred_cf_mw.values))
    print(f"    All-fold CF nMAE: {all_n:.4f}%")

    # --- Final test submission (recency-weighted fold blend) ---
    test_combined = add_walk_forward_availability(
        combined, train_end=train_end,
        wind_col="wind_speed_120m",
        impossible_col="_is_impossible",
        window_days=30,
        freeze_after_ts=train_end,
    )
    df_valid_sorted = test_combined[test_combined["_split"] == "valid"].reset_index(drop=True)

    test_df = pd.DataFrame({
        TIMESTAMP_COL: df_valid_sorted[TIMESTAMP_COL].to_numpy(),
        "_submission_row": df_valid_sorted["_submission_row"].to_numpy().astype(int),
        "active_turbines": df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32),
    })
    for fold_idx in FOLD_IDS:
        test_df[f"test_cf_fold{fold_idx + 1}"] = test_cf_per_fold[fold_idx]
    test_df.to_parquet(TEST_PATH, index=False)

    # Recency-weighted blend
    avg_cf = sum(FOLD_BLEND[i] * test_cf_per_fold[i] for i in FOLD_IDS)
    active_val = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    final_mw   = np.clip(_from_cf(avg_cf, active_val), 0.0, CAPACITY_MW).astype(np.float64)

    order  = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po     = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    ts_po  = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"\n  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
