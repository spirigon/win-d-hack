"""V32 + persisted OOF predictions.

Identical training to ``src/training/train_v32_era5v2.py`` (v27.1 mw50/cf50
architecture + ERA5v2 features) but writes OOF predictions to disk so
downstream post-processing (iso-recal, ridge stack) can consume them.

For each fold in the CV-bag (folds 3, 4, 5) and each target (CF, MW), we
average specialist predictions across seeds on the validation slice and
store the (timestamp, prediction, target) triple. We also store the test
predictions per fold so a ridge stack can use the per-fold averaged test
predictions, not the final blend.

Outputs:

    data/processed/v32_oof.parquet         OOF rows (timestamp, target, predicted_cf, predicted_mw, fold)
    data/processed/v32_test.parquet        Per-fold test predictions
                                            (timestamp, _submission_row,
                                             test_cf_fold3, test_cf_fold4, test_cf_fold5,
                                             test_mw_fold3, ...,
                                             active_turbines)
    submissions/archive/v32.1_with_oof.csv Final blend submission (should match v32.0)

Usage:

    python -m src.training.train_v32_with_oof --seeds 5
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

# Reuse helpers from v32 script.
from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, K, BLEND_WEIGHT_MW, FOLD_IDS, SEEDS_3, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_PATH = _ROOT / "data" / "processed" / "v32_oof.parquet"
TEST_PATH = _ROOT / "data" / "processed" / "v32_test.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v32.1_with_oof.csv"


def _train_fold_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, seeds):
    """3 specialists × N seeds. Returns (val_preds, test_preds), each averaged."""
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
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
    ap.add_argument("--seeds", choices=["3", "5"], default="5")
    args = ap.parse_args()
    seeds = SEEDS_5 if args.seeds == "5" else SEEDS_3
    set_global_seed(42)

    print("=" * 72)
    print(f"V32 + OOF persistence  ({len(seeds)} seeds)")
    print("=" * 72)

    # --- Build features ---------------------------------------------------
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

    df_train_full = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full)
    df_train_full["_is_impossible"] = impossible.values

    # --- Probe → top-K ----------------------------------------------------
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

    # --- CV-bag with OOF + per-fold test preds saved ---------------------
    print(f"\n[4/4] Training CV-bag (folds {[i + 1 for i in FOLD_IDS]}, "
          f"seeds={seeds})...")
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

            if target_mode == "cf":
                y_tr = _to_cf(y_tr_mw, active_tr)
                y_va = _to_cf(y_va_mw, active_va)
            else:
                y_tr = y_tr_mw
                y_va = y_va_mw

            val_pred, test_pred = _train_fold_ensemble(
                X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, seeds,
            )

            if target_mode == "cf":
                val_mw = np.clip(_from_cf(val_pred, active_va), 0, CAPACITY_MW)
                test_cf_per_fold[fold_idx] = test_pred
            else:
                val_mw = np.clip(val_pred, 0, CAPACITY_MW)
                test_mw_per_fold[fold_idx] = test_pred

            fold_nmae = float(normalized_mae(y_va_mw, val_mw))
            print(f"    Fold {fold_idx + 1}: nMAE={fold_nmae:.4f}%  ({time.time() - t0:.0f}s)")

            # Persist OOF rows.
            ts_va = df_va[TIMESTAMP_COL].to_numpy()
            for i in range(len(y_va_mw)):
                # Find or create the OOF record for this (fold, ts).
                # We'll union both targets at the end.
                oof_records.append({
                    "fold": fold_idx + 1,
                    "ts": pd.Timestamp(ts_va[i]),
                    "target_mw": float(y_va_mw[i]),
                    "active_turbines": float(active_va[i]),
                    "ws_120": float(df_va["wind_speed_120m"].to_numpy()[i]),
                    "n_repair": float(df_va["n_repair"].to_numpy()[i]) if "n_repair" in df_va.columns else np.nan,
                    "raw_pred": float(val_pred[i]),
                    "target_mode": target_mode,
                })

    # --- Build OOF and test parquets -------------------------------------
    oof_df = pd.DataFrame(oof_records)
    # Pivot target_mode -> columns. n_repair can be NaN on Fold-5 (the
    # column is not always present in df_va depending on feature build),
    # so drop it from the index — it's not needed downstream.
    index_cols = ["fold", "ts", "target_mw", "active_turbines", "ws_120"]
    oof_wide = (
        oof_df.pivot_table(
            index=index_cols,
            columns="target_mode",
            values="raw_pred",
        )
        .reset_index()
    )
    oof_wide.columns.name = None
    if "cf" in oof_wide.columns:
        oof_wide["pred_cf_mw"] = np.clip(_from_cf(oof_wide["cf"].to_numpy(),
                                                  oof_wide["active_turbines"].to_numpy()),
                                         0, CAPACITY_MW)
    if "mw" in oof_wide.columns:
        oof_wide["pred_mw_mw"] = np.clip(oof_wide["mw"].to_numpy(), 0, CAPACITY_MW)
    if "pred_cf_mw" in oof_wide.columns and "pred_mw_mw" in oof_wide.columns:
        oof_wide["pred_blend_mw"] = (BLEND_WEIGHT_MW * oof_wide["pred_mw_mw"] +
                                     (1 - BLEND_WEIGHT_MW) * oof_wide["pred_cf_mw"])
        oof_wide["pred_blend_mw"] = oof_wide["pred_blend_mw"].clip(0, CAPACITY_MW)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_wide.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH} ({len(oof_wide)} rows)")

    # Per-fold test predictions.
    test_df = pd.DataFrame({
        TIMESTAMP_COL: df_valid_sorted[TIMESTAMP_COL].to_numpy(),
        "_submission_row": df_valid_sorted["_submission_row"].to_numpy().astype(int),
        "active_turbines": df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32),
    })
    for fold_idx in FOLD_IDS:
        test_df[f"test_cf_fold{fold_idx + 1}"] = test_cf_per_fold[fold_idx]
        test_df[f"test_mw_fold{fold_idx + 1}"] = test_mw_per_fold[fold_idx]
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"  Per-fold test preds saved: {TEST_PATH} ({len(test_df)} rows)")

    # --- Final blend submission (sanity match with v32.0) ----------------
    avg_test_cf = np.mean([test_cf_per_fold[i] for i in FOLD_IDS], axis=0)
    avg_test_mw = np.mean([test_mw_per_fold[i] for i in FOLD_IDS], axis=0)
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    pred_cf_mw = np.clip(_from_cf(avg_test_cf, active_valid), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(avg_test_mw, 0, CAPACITY_MW)
    final_mw = (BLEND_WEIGHT_MW * pred_mw_mw + (1 - BLEND_WEIGHT_MW) * pred_cf_mw).clip(0, CAPACITY_MW)

    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    ts_po = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"\n  Submission saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
