"""V110: XGBoost + LGBM blend on V97b's feature set.

XGBoost uses level-wise tree growth with split-finding based on approximate
quantile histograms — different regularization and tree structure from LightGBM's
leaf-wise growth.  Blending V97b (LGBM regime-specialist) + V110 (XGBoost)
provides genuine model diversity from a different GBM implementation.

Feature pipeline: identical to V97b (byte-identical dedup only, GEM/ICON-G).
Training: XGBoost with hist tree method (fast CPU) per seed per fold.
          Regime-split XGBoost (same 3-regime weights as LGBM).
Blend: 50/50 LGBM+XGBoost at CF level. CF-only (BLEND_WEIGHT_MW=0.0).

Outputs:
    data/processed/v110_oof.parquet
    submissions/archive/v110.0_xgboost.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
from src.features.gem_icon_features import add_gem_icon_features, gem_icon_columns
from src.features.hub_height_features import add_hub_height_features, hub_height_columns
from src.features.multi_nwp_features import (
    build_nwp_consensus, merge_ecmwf_ifs, merge_gfs, multi_nwp_columns,
)
from src.features.nasa_features import merge_nasa_merra2, nasa_columns
from src.features.nwp_ensemble import merge_nwp_ensemble, nwp_ensemble_columns
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.seasonal_features import add_seasonal_features, seasonal_columns
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, K, BLEND_WEIGHT_MW, FOLD_IDS, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH  = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH  = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH   = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_PATH    = _ROOT / "data" / "processed" / "v110_oof.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v110.0_xgboost.csv"

MARCH_WEIGHT    = 3.0
SEEDS           = SEEDS_5
K               = 90
BLEND_WEIGHT_MW = 0.0  # CF-only: optimal per blend_weight_optimizer

# XGBoost hyperparameters (CPU, hist method)
XGB_PARAMS = {
    "objective":       "reg:squarederror",
    "tree_method":     "hist",
    "device":          "cpu",
    "learning_rate":   0.02,
    "max_depth":       8,           # approx equivalent to num_leaves=127 in LGBM
    "min_child_weight": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.6,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "max_bin":         255,
    "verbosity":       0,
}
XGB_NUM_ROUNDS  = 5000
XGB_EARLY_STOP  = 250

_GPU_OVERRIDES: dict = {}


def _train_xgb_fold(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sw, seeds):
    """Regime-split XGBoost ensemble (3 regimes × seeds)."""
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = (np.where(mask, 2.0, 0.3) * sw).astype(np.float32)
        vp, tp = [], []
        for s in seeds:
            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=weights, feature_names=feat_cols)
            dval   = xgb.DMatrix(X_va,   label=y_va,               feature_names=feat_cols)
            dtest  = xgb.DMatrix(X_test,                            feature_names=feat_cols)
            params = {**XGB_PARAMS, "seed": s}
            booster = xgb.train(
                params, dtrain, num_boost_round=XGB_NUM_ROUNDS,
                evals=[(dval, "val")],
                early_stopping_rounds=XGB_EARLY_STOP,
                verbose_eval=False,
            )
            vp.append(booster.predict(dval, iteration_range=(0, booster.best_iteration + 1)))
            tp.append(booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1)))
        regime_val[name]  = np.mean(vp, axis=0)
        regime_test[name] = np.mean(tp, axis=0)
    return (
        np.mean(list(regime_val.values()), axis=0),
        np.mean(list(regime_test.values()), axis=0),
    )


def _train_lgbm_fold(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sw, seeds):
    """Regime-split LGBM ensemble (same as V97b)."""
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
    set_global_seed(42)

    print("=" * 72)
    print(f"V110: XGBoost + LGBM blend  ({len(SEEDS)} seeds, K={K})")
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
    combined = merge_nwp_ensemble(combined)
    combined = merge_nasa_merra2(combined)
    combined = merge_gfs(combined)
    combined = merge_ecmwf_ifs(combined)
    combined = build_nwp_consensus(combined)
    combined = add_all_advanced_features(combined)
    combined = add_hub_height_features(combined)
    combined = add_seasonal_features(combined)
    combined = add_gem_icon_features(combined)

    nwp_cols  = nwp_ensemble_columns(combined)
    nasa_cols = nasa_columns(combined)
    mnwp_cols = multi_nwp_columns(combined)
    adv_cols  = advanced_interaction_columns(combined)
    hub_cols  = hub_height_columns(combined)
    seas_cols = seasonal_columns(combined)
    gem_cols  = gem_icon_columns(combined)

    print(f"  Feature groups: nwp={len(nwp_cols)} nasa={len(nasa_cols)} "
          f"mnwp={len(mnwp_cols)} adv={len(adv_cols)} "
          f"hub={len(hub_cols)} seasonal={len(seas_cols)} gem_icon={len(gem_cols)}")

    df_train_full_pre = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full_pre)
    is_impossible_full = pd.Series(False, index=combined.index)
    is_impossible_full.iloc[: len(df_train_full_pre)] = impossible.values
    combined["_is_impossible"] = is_impossible_full.values
    train_end = df_train_full_pre[TIMESTAMP_COL].max()

    sample_weight_full = (~is_impossible_full.to_numpy()[: len(df_train_full_pre)]).astype(np.float32)
    march_mask = df_train_full_pre[TIMESTAMP_COL].dt.month == 3
    sample_weight_full[march_mask] *= MARCH_WEIGHT
    n_curtail = int(impossible.sum())
    n_march   = int(march_mask.sum())
    n_zero    = int((sample_weight_full == 0).sum())
    print(f"  Curtailment: {n_curtail} rows ({100*n_curtail/len(df_train_full_pre):.1f}%)")
    print(f"  Zero-weight rows: {n_zero} ({100*n_zero/len(df_train_full_pre):.1f}%)")
    print(f"  March rows x{MARCH_WEIGHT}: {n_march} ({100*n_march/len(df_train_full_pre):.1f}%)")

    print(f"[2/4] Feature selection probe (Fold-5, K={K})...")
    folds = default_folds()
    fold5 = folds[-1]

    probe_combined = add_walk_forward_availability(
        combined, train_end=fold5.train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=fold5.train_end,
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

    all_extra_cols = list(dict.fromkeys(
        nwp_cols + nasa_cols + mnwp_cols + adv_cols + hub_cols + seas_cols + gem_cols
    ))
    feat_cols_all = [
        c for c in feature_columns(df_t_)
        if c not in ("_is_impossible", "_split", TARGET_COL) and c != "avail_underprod_mw"
    ]
    for col in all_extra_cols:
        if col in df_t_.columns and col not in feat_cols_all:
            feat_cols_all.append(col)

    # Byte-identical dedup
    seen_keys: dict[bytes, str] = {}
    dedup_cols: list[str] = []
    for col in feat_cols_all:
        try:
            key = df_t_[col].to_numpy(dtype=np.float32).tobytes()
            if key not in seen_keys:
                seen_keys[key] = col
                dedup_cols.append(col)
        except Exception:
            dedup_cols.append(col)
    n_removed = len(feat_cols_all) - len(dedup_cols)
    feat_cols_all = dedup_cols
    print(f"  Feature pool: {len(feat_cols_all)} (removed {n_removed} byte-identical duplicates)")

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

    top3 = [n for n, _ in feat_imp[:3]]
    print(f"  Top-{K} selected  (top-3: {top3})")
    for label, cols in [
        ("hub_height", hub_cols), ("seasonal", seas_cols), ("gem_icon", gem_cols),
    ]:
        hit = [c for c in top_k if c in cols]
        print(f"  {label} in top-{K}: {len(hit)}  {hit}")

    print(f"\n[3/4] Training LGBM + XGBoost CV-bag (folds {[i + 1 for i in FOLD_IDS]}, "
          f"seeds={SEEDS})...")
    test_cf_per_fold: dict[int, np.ndarray] = {}
    oof_records: list[dict] = []

    df_valid_sorted = combined[combined["_split"] == "valid"].sort_values(TIMESTAMP_COL).reset_index(drop=True)
    X_test = df_valid_sorted[top_k].to_numpy(dtype=np.float32)
    ws_test = df_valid_sorted["wind_speed_120m"].to_numpy(dtype=np.float32)
    n_test = len(df_valid_sorted)

    for fold_idx in FOLD_IDS:
        t_fold = time.time()
        fold = folds[fold_idx]

        fold_combined = add_walk_forward_availability(
            combined, train_end=fold.train_end,
            wind_col="wind_speed_120m", impossible_col="_is_impossible",
            window_days=30, freeze_after_ts=fold.train_end,
        )
        fold_train_full = fold_combined[fold_combined["_split"] == "train"].reset_index(drop=True)

        tr_idx, va_idx = split_indices(fold_train_full, fold)
        fold_tr_raw = fold_train_full.iloc[tr_idx]
        fold_va_raw = fold_train_full.iloc[va_idx]

        fit_data = fold_tr_raw[~fold_tr_raw["_is_impossible"]]
        pc_s = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_g = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        w = fit_wake_lookup(fit_data, n_sectors=16)

        fold_tr = _add_pc(fold_tr_raw, pc_s, pc_g)
        fold_tr = add_wake_features(fold_tr, w)
        fold_va = _add_pc(fold_va_raw, pc_s, pc_g)
        fold_va = add_wake_features(fold_va, w)

        # Apply same PC to test
        df_valid_pc = _add_pc(df_valid_sorted, pc_s, pc_g)
        df_valid_pc = add_wake_features(df_valid_pc, w)
        X_test_fold = df_valid_pc[top_k].to_numpy(dtype=np.float32)

        a_tr = fold_tr["active_turbines"].to_numpy(dtype=np.float32)
        a_va = fold_va["active_turbines"].to_numpy(dtype=np.float32)
        y_tr_cf = _to_cf(fold_tr[TARGET_COL].to_numpy(dtype=np.float32), a_tr)
        y_va_cf = _to_cf(fold_va[TARGET_COL].to_numpy(dtype=np.float32), a_va)
        y_va_mw = fold_va[TARGET_COL].to_numpy(dtype=np.float32)

        X_tr = fold_tr[top_k].to_numpy(dtype=np.float32)
        X_va = fold_va[top_k].to_numpy(dtype=np.float32)
        ws_tr = fold_tr["wind_speed_120m"].to_numpy(dtype=np.float32)
        sw = sample_weight_full[tr_idx]

        # LGBM regime-split ensemble
        lgbm_val_cf, lgbm_test_cf = _train_lgbm_fold(
            X_tr, y_tr_cf, X_va, y_va_cf, X_test_fold, top_k, ws_tr, sw, SEEDS
        )
        # XGBoost regime-split ensemble
        xgb_val_cf, xgb_test_cf = _train_xgb_fold(
            X_tr, y_tr_cf, X_va, y_va_cf, X_test_fold, top_k, ws_tr, sw, SEEDS
        )

        # 50/50 CF blend
        val_cf  = 0.5 * lgbm_val_cf  + 0.5 * xgb_val_cf
        test_cf = 0.5 * lgbm_test_cf + 0.5 * xgb_test_cf

        # MW predictions (CF-only: BLEND_WEIGHT_MW=0.0)
        a_va_full = fold_va["active_turbines"].to_numpy(dtype=np.float32)
        val_mw = _from_cf(np.clip(val_cf, 0, 1), a_va_full)
        test_mw = _from_cf(np.clip(test_cf, 0, 1),
                           df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32))

        # LGBM CF-only nMAE
        lgbm_val_mw = _from_cf(np.clip(lgbm_val_cf, 0, 1), a_va_full)
        xgb_val_mw  = _from_cf(np.clip(xgb_val_cf,  0, 1), a_va_full)
        fold_nmae      = float(normalized_mae(y_va_mw, val_mw))
        lgbm_fold_nmae = float(normalized_mae(y_va_mw, lgbm_val_mw))
        xgb_fold_nmae  = float(normalized_mae(y_va_mw, xgb_val_mw))
        elapsed = time.time() - t_fold
        print(f"    Fold {fold_idx + 1}: blend={fold_nmae:.4f}%  "
              f"LGBM={lgbm_fold_nmae:.4f}%  XGB={xgb_fold_nmae:.4f}%  ({elapsed:.0f}s)")

        test_cf_per_fold[fold_idx] = test_cf

        ts_va = fold_va[TIMESTAMP_COL].to_numpy()
        for i, (ts, y_true, y_pred_cf, y_pred_mw) in enumerate(
                zip(ts_va, y_va_mw, val_cf, val_mw, strict=True)):
            oof_records.append({
                "fold": fold_idx + 1,
                "ts": ts,
                "target_mw": float(y_true),
                "pred_cf_mw": float(_from_cf(max(0.0, min(1.0, float(y_pred_cf))),
                                             float(a_va[i]))),
                "pred_mw_mw": float(y_pred_mw),
            })

    # OOF summary
    oof_df = pd.DataFrame(oof_records)
    oof_df["pred_blend_mw"] = (
        BLEND_WEIGHT_MW * oof_df["pred_mw_mw"] + (1 - BLEND_WEIGHT_MW) * oof_df["pred_cf_mw"]
    )
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")
    for fid in sorted(oof_df["fold"].unique()):
        sub = oof_df[oof_df["fold"] == fid]
        nb = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_blend_mw"].to_numpy()))
        nc = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_cf_mw"].to_numpy()))
        print(f"    Fold {fid} blend nMAE: {nb:.4f}%  CF nMAE: {nc:.4f}%")
    all_blend = float(normalized_mae(oof_df["target_mw"].to_numpy(), oof_df["pred_blend_mw"].to_numpy()))
    print(f"    All-fold blend nMAE: {all_blend:.4f}%")

    print("\n[4/4] Building final submission...")
    # Average test CF predictions across folds
    test_cf_avg = np.mean(list(test_cf_per_fold.values()), axis=0)
    final_mw = _from_cf(
        np.clip(test_cf_avg, 0, 1),
        df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32),
    )

    # Restore original submission row order
    order = df_valid_sorted.index.to_numpy()
    n_valid = len(df_valid)
    po = np.empty(n_valid, dtype=np.float32)
    po[order] = final_mw
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    ts_po  = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=n_valid, timestamps=ts_po)
    print(f"  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
