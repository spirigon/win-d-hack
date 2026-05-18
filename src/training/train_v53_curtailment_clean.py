"""V53: V34.1 architecture + curtailment-aware training mask.

V34.1 (LB 7.56) is our champion. The Fold-5 diagnostic showed the worst
hours are persistent multi-hour curtailment / unavailability events that
the legacy ``_is_impossible`` mask doesn't catch. ``CurtailmentMask``
(``src.data.curtailment_mask``) flags ~3.2% of training rows as
underproduction events relative to a clean isotonic baseline.

V53 ORs the curtailment mask into ``_is_impossible`` so those rows get
zero weight during training (and during the per-fold isotonic fit). The
test predictions still cover all 2126 Q1 2026 hours.

Outputs:
    data/processed/v53_oof.parquet
    data/processed/v53_test.parquet
    submissions/archive/v53.0_curtailment_clean.csv
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

from src.data.curtailment_mask import (
    CurtailmentConfig,
    identify_curtailment_rows,
    summarize_curtailment,
)
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
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_PATH = _ROOT / "data" / "processed" / "v53_oof.parquet"
TEST_PATH = _ROOT / "data" / "processed" / "v53_test.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v53.0_curtailment_clean.csv"


def _train_fold_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sample_weight, seeds):
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        val_preds, test_preds = [], []
        for s in seeds:
            cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=LGBM_PARAMS.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(LGBM_PARAMS.early_stopping_rounds, verbose=False)],
            )
            val_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
            test_preds.append(b.predict(X_test, num_iteration=b.best_iteration))
        regime_val[name] = np.mean(val_preds, axis=0)
        regime_test[name] = np.mean(test_preds, axis=0)
    return np.mean(list(regime_val.values()), axis=0), np.mean(list(regime_test.values()), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["3", "5"], default="3")
    ap.add_argument("--min-run", type=int, default=3,
                    help="curtailment min run length (3=default; 1=flag every underproduction)")
    ap.add_argument("--frac", type=float, default=0.4,
                    help="curtailment ratio threshold (target/expected) below which we flag")
    args = ap.parse_args()
    seeds = SEEDS_5 if args.seeds == "5" else SEEDS_3
    set_global_seed(42)

    print("=" * 72)
    print(f"V53: V34.1 + curtailment cleaning  ({len(seeds)} seeds, "
          f"min_run={args.min_run}, frac={args.frac})")
    print("=" * 72)

    print("\n[1/4] Loading + building features...")
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
    legacy_impossible = identify_impossible_rows(df_train_full_pre)

    # Curtailment mask on training set.
    print("[2/4] Identifying curtailment rows...")
    cfg = CurtailmentConfig(min_run_length_hours=args.min_run, fraction_threshold=args.frac)
    curtail_mask, diag = identify_curtailment_rows(
        df_train_full_pre, cfg=cfg, pre_existing_impossible=legacy_impossible,
    )
    summary = summarize_curtailment(diag)
    print(f"  {summary}")

    impossible_combined = (legacy_impossible.to_numpy() | curtail_mask.to_numpy())
    n_legacy = int(legacy_impossible.sum())
    n_curtail = int(curtail_mask.sum())
    n_overlap = int((legacy_impossible & curtail_mask).sum())
    print(f"  legacy: {n_legacy}, curtailment: {n_curtail}, overlap: {n_overlap}")
    print(f"  total flagged: {impossible_combined.sum()} "
          f"({impossible_combined.sum() / len(df_train_full_pre) * 100:.2f}%)")

    # Propagate combined mask back into combined frame.
    is_impossible_full = pd.Series(False, index=combined.index)
    is_impossible_full.iloc[: len(df_train_full_pre)] = impossible_combined
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

    # --- Probe ----------------------------------------------------------
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
        and c != "avail_underprod_mw"
    ]
    print(f"  Feature pool: {len(feat_cols_all)}  ({len(era5v2_columns(df_t_))} era5v2_*)")
    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = _to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = _to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    X_t_ = df_t_[feat_cols_all].to_numpy(dtype=np.float32)
    X_v_ = df_v_[feat_cols_all].to_numpy(dtype=np.float32)
    sw_fold = sample_weight_full[train_idx]
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

    # --- CV-bag (mw + cf) ---------------------------------------------
    print(f"\n[4/4] Training CV-bag (folds {[i + 1 for i in FOLD_IDS]}, seeds={seeds})...")
    test_cf_per_fold = {}
    test_mw_per_fold = {}
    oof_records: list[dict] = []

    for target_mode in ("cf", "mw"):
        print(f"\n  --- {target_mode.upper()} target ---")
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

            active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
            active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
            y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
            y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
            X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
            X_va = df_va[top_k].to_numpy(dtype=np.float32)
            X_test = df_te[top_k].to_numpy(dtype=np.float32)
            ws_tr = df_tr["wind_speed_120m"].to_numpy()
            sw = sample_weight_full[train_idx]

            if target_mode == "cf":
                y_tr = _to_cf(y_tr_mw, active_tr)
                y_va = _to_cf(y_va_mw, active_va)
            else:
                y_tr = y_tr_mw
                y_va = y_va_mw

            val_pred, test_pred = _train_fold_ensemble(
                X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, sw, seeds,
            )

            # Important: validation rows ALSO contain curtailment rows. They
            # affect the OOF nMAE but not the model. Report both numbers.
            if target_mode == "cf":
                val_mw = np.clip(_from_cf(val_pred, active_va), 0, CAPACITY_MW)
                test_cf_per_fold[fold_idx] = test_pred
            else:
                val_mw = np.clip(val_pred, 0, CAPACITY_MW)
                test_mw_per_fold[fold_idx] = test_pred

            fold_nmae_all = float(normalized_mae(y_va_mw, val_mw))
            # Also compute Fold-5 nMAE excluding curtailment rows.
            val_curtail = df_va["_is_impossible"].to_numpy()
            if (~val_curtail).any():
                fold_nmae_clean = float(normalized_mae(
                    y_va_mw[~val_curtail], val_mw[~val_curtail]
                ))
            else:
                fold_nmae_clean = float("nan")
            print(f"    Fold {fold_idx + 1}: nMAE={fold_nmae_all:.4f}%  "
                  f"(clean: {fold_nmae_clean:.4f}%)  ({time.time() - t0:.0f}s)")

            ts_va = df_va[TIMESTAMP_COL].to_numpy()
            for i in range(len(y_va_mw)):
                oof_records.append({
                    "fold": int(fold_idx + 1),
                    "ts": pd.Timestamp(ts_va[i]),
                    "target_mw": float(y_va_mw[i]),
                    "active_turbines": float(active_va[i]),
                    "ws_120": float(df_va["wind_speed_120m"].to_numpy()[i]),
                    "raw_pred": float(val_pred[i]),
                    "target_mode": target_mode,
                })

    # --- OOF + test parquet --------------------------------------------
    oof_df = pd.DataFrame(oof_records)
    oof_wide = (
        oof_df.pivot_table(
            index=["fold", "ts", "target_mw", "active_turbines", "ws_120"],
            columns="target_mode",
            values="raw_pred",
        )
        .reset_index()
    )
    oof_wide.columns.name = None
    oof_wide["pred_cf_mw"] = np.clip(_from_cf(oof_wide["cf"].to_numpy(),
                                              oof_wide["active_turbines"].to_numpy()),
                                     0, CAPACITY_MW)
    oof_wide["pred_mw_mw"] = np.clip(oof_wide["mw"].to_numpy(), 0, CAPACITY_MW)
    oof_wide["pred_blend_mw"] = (
        BLEND_WEIGHT_MW * oof_wide["pred_mw_mw"] + (1 - BLEND_WEIGHT_MW) * oof_wide["pred_cf_mw"]
    ).clip(0, CAPACITY_MW)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_wide.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")
    for fid in [3, 4, 5]:
        sub = oof_wide[oof_wide["fold"] == fid]
        n = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_blend_mw"].to_numpy()))
        print(f"    Fold {fid} blend nMAE: {n:.4f}%")

    test_df = pd.DataFrame({
        TIMESTAMP_COL: df_valid_sorted[TIMESTAMP_COL].to_numpy(),
        "_submission_row": df_valid_sorted["_submission_row"].to_numpy().astype(int),
        "active_turbines": df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32),
    })
    for fold_idx in FOLD_IDS:
        test_df[f"test_cf_fold{fold_idx + 1}"] = test_cf_per_fold[fold_idx]
        test_df[f"test_mw_fold{fold_idx + 1}"] = test_mw_per_fold[fold_idx]
    test_df.to_parquet(TEST_PATH, index=False)

    # --- Submission ----------------------------------------------------
    avg_test_cf = np.mean([test_cf_per_fold[i] for i in FOLD_IDS], axis=0)
    avg_test_mw = np.mean([test_mw_per_fold[i] for i in FOLD_IDS], axis=0)
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    pred_cf_mw = np.clip(_from_cf(avg_test_cf, active_valid), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(avg_test_mw, 0, CAPACITY_MW)
    final_mw = (BLEND_WEIGHT_MW * pred_mw_mw + (1 - BLEND_WEIGHT_MW) * pred_cf_mw)
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
    print(f"  Mean: {final_mw.mean():.2f} MW   "
          f"Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
