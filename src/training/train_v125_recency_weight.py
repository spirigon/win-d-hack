"""V125: v97b retrained with recency weighting + winning corrections.

Key insight: power output at ws 6-10 m/s dropped from 38.6 MW (2024 Q1)
to 36.6 MW (2025 Q1) — a 5% decline. The model trained on 2023-2025
averages this out, but Q1 2026 likely follows the 2025 trend (lower
efficiency due to aging/operational changes).

Solution: retrain with exponential recency weighting so 2025 data has
much higher influence than 2023 data. This naturally shifts the learned
power curve toward the 2025 (lower) regime.

We also apply the LB-proven correction recipe on top:
  CF + hw_bias*0.7 + co_bias*0.5 + q1_bias*0.7

Additionally: since the test has 41% of wind in 8-12 m/s (vs 29% in
Fold 5), and mid-wind has negative bias, the corrections are even more
justified.
"""

from __future__ import annotations

import gc
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
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, FOLD_IDS, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH  = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_PATH   = _ROOT / "data" / "processed" / "v125_oof.parquet"
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TARGET_COL_NAME = "Выработка. Результирующий расчет"
MARCH_WEIGHT = 3.0
SEEDS = SEEDS_5
K = 90

# Recency weighting: exponential decay from most recent data
# Half-life = 6 months → 2025 Q4 gets weight 1.0, 2023 Q1 gets ~0.25
RECENCY_HALFLIFE_DAYS = 180


def _write(preds: np.ndarray, path: Path, label: str = "") -> None:
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TARGET_COL_NAME: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<50s} mean={preds.mean():.2f}  {label}")


def _compute_recency_weights(timestamps: pd.Series, reference_date: pd.Timestamp,
                             halflife_days: float = RECENCY_HALFLIFE_DAYS) -> np.ndarray:
    """Exponential recency weighting. Most recent = 1.0, decays with age."""
    days_ago = (reference_date - timestamps).dt.total_seconds() / 86400.0
    decay = np.exp(-np.log(2) * days_ago / halflife_days)
    return decay.to_numpy(dtype=np.float32)


def _train_fold_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sw, seeds):
    """3 regime specialists × N seeds."""
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
        regime_val[name] = np.mean(vp, axis=0)
        regime_test[name] = np.mean(tp, axis=0)
    return np.mean(list(regime_val.values()), axis=0), np.mean(list(regime_test.values()), axis=0)


def main() -> None:
    set_global_seed(42)
    t_start = time.time()

    print("=" * 72)
    print(f"V125: Recency-weighted v97b ({len(SEEDS)} seeds, K={K})")
    print(f"  Halflife: {RECENCY_HALFLIFE_DAYS} days")
    print("=" * 72)

    # --- Build features (same as v97b) ---
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

    nwp_cols = nwp_ensemble_columns(combined)
    nasa_cols_list = nasa_columns(combined)
    mnwp_cols = multi_nwp_columns(combined)
    adv_cols = advanced_interaction_columns(combined)
    hub_cols = hub_height_columns(combined)
    seas_cols = seasonal_columns(combined)
    gem_cols = gem_icon_columns(combined)

    df_train_full = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full)
    is_impossible_full = pd.Series(False, index=combined.index)
    is_impossible_full.iloc[:len(df_train_full)] = impossible.values
    combined["_is_impossible"] = is_impossible_full.values
    train_end = df_train_full[TIMESTAMP_COL].max()

    # RECENCY WEIGHTING — the key difference from v97b
    # Reference: end of training period (latest data = highest weight)
    recency_weights = _compute_recency_weights(
        df_train_full[TIMESTAMP_COL], train_end, RECENCY_HALFLIFE_DAYS
    )
    # Combine with impossible-row masking and March boost
    sample_weight_full = (~impossible.to_numpy()).astype(np.float32)
    sample_weight_full *= recency_weights  # Apply recency
    march_mask = df_train_full[TIMESTAMP_COL].dt.month == 3
    sample_weight_full[march_mask] *= MARCH_WEIGHT

    # Report weight distribution
    w_2023 = sample_weight_full[df_train_full[TIMESTAMP_COL].dt.year == 2023].mean()
    w_2024 = sample_weight_full[df_train_full[TIMESTAMP_COL].dt.year == 2024].mean()
    w_2025 = sample_weight_full[df_train_full[TIMESTAMP_COL].dt.year == 2025].mean()
    print(f"  Recency weights: 2023={w_2023:.3f}  2024={w_2024:.3f}  2025={w_2025:.3f}")
    print(f"  Ratio 2025/2023: {w_2025/w_2023:.2f}x")

    # --- Feature selection probe (same as v97b) ---
    print(f"\n[2/4] Feature selection probe (K={K})...")
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
        nwp_cols + nasa_cols_list + mnwp_cols + adv_cols + hub_cols + seas_cols + gem_cols
    ))
    feat_cols_all = [
        c for c in feature_columns(df_t_)
        if c not in ("_is_impossible", "_split", TARGET_COL) and c != "avail_underprod_mw"
    ]
    for col in all_extra_cols:
        if col in df_t_.columns and col not in feat_cols_all:
            feat_cols_all.append(col)

    # Dedup byte-identical
    seen_keys = {}
    dedup_cols = []
    for col in feat_cols_all:
        try:
            key = df_t_[col].to_numpy(dtype=np.float32).tobytes()
            if key not in seen_keys:
                seen_keys[key] = col
                dedup_cols.append(col)
        except Exception:
            dedup_cols.append(col)
    feat_cols_all = dedup_cols
    print(f"  Feature pool: {len(feat_cols_all)}")

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
    feat_imp = sorted(zip(feat_cols_all, imp.tolist()), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    print(f"  Top-{K} selected (top-3: {[n for n, _ in feat_imp[:3]]})")

    del probe_combined, probe_train, df_t_, df_v_, dt_, dv_, probe
    gc.collect()

    # --- CV-bag training ---
    print(f"\n[3/4] Training CV-bag (CF target, recency-weighted)...")
    test_cf_per_fold = {}
    oof_records = []

    for fold_idx in FOLD_IDS:
        t0 = time.time()
        fold = folds[fold_idx]

        fc = add_walk_forward_availability(
            combined, train_end=fold.train_end,
            wind_col="wind_speed_120m", impossible_col="_is_impossible",
            window_days=30, freeze_after_ts=fold.train_end,
        )
        ft = fc[fc["_split"] == "train"].reset_index(drop=True)
        fv = fc[fc["_split"] == "valid"].reset_index(drop=True)

        train_idx, val_idx = split_indices(ft, fold)
        fold_train = ft.iloc[train_idx]
        fold_val = ft.iloc[val_idx]
        fit_data = fold_train[~fold_train["_is_impossible"]]
        pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        wake = fit_wake_lookup(fit_data, n_sectors=16)
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
        y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        y_tr = _to_cf(y_tr_mw, active_tr)
        y_va = _to_cf(y_va_mw, active_va)
        X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr = df_tr["wind_speed_120m"].to_numpy()
        sw = sample_weight_full[train_idx]

        val_pred, test_pred = _train_fold_ensemble(
            X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, sw, SEEDS,
        )
        val_mw = np.clip(_from_cf(val_pred, active_va), 0, CAPACITY_MW)
        test_cf_per_fold[fold_idx] = test_pred

        fold_nmae = float(normalized_mae(y_va_mw, val_mw))
        print(f"  Fold {fold_idx + 1}: CF nMAE={fold_nmae:.4f}%  ({time.time()-t0:.0f}s)")

        ts_va = df_va[TIMESTAMP_COL].to_numpy()
        for i in range(len(y_va_mw)):
            oof_records.append({
                "fold": int(fold_idx + 1),
                "ts": pd.Timestamp(ts_va[i]),
                "target_mw": float(y_va_mw[i]),
                "active_turbines": float(active_va[i]),
                "ws_120": float(df_va["wind_speed_120m"].to_numpy()[i]),
                "pred_cf": float(val_pred[i]),
            })

        del fc, ft, fv, df_tr, df_va, df_te, X_tr, X_va, X_test
        gc.collect()

    # --- Build submission ---
    print(f"\n[4/4] Building submission...")
    test_combined = add_walk_forward_availability(
        combined, train_end=train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=train_end,
    )
    df_valid_sorted = test_combined[test_combined["_split"] == "valid"].reset_index(drop=True)

    avg_cf = np.mean([test_cf_per_fold[i] for i in FOLD_IDS], axis=0)
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    pred_cf_mw = np.clip(_from_cf(avg_cf, active_valid), 0, CAPACITY_MW)

    # OOF evaluation
    oof_df = pd.DataFrame(oof_records)
    oof_df["pred_mw"] = np.clip(
        _from_cf(oof_df["pred_cf"].to_numpy(), oof_df["active_turbines"].to_numpy()),
        0, CAPACITY_MW
    )
    for fid in sorted(oof_df["fold"].unique()):
        sub = oof_df[oof_df["fold"] == fid]
        nm = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_mw"].to_numpy()))
        print(f"  OOF Fold {fid}: {nm:.4f}%")

    # Save OOF
    oof_df["pred_blend_mw"] = oof_df["pred_mw"]
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)

    # Compute bias corrections for the recency-weighted model
    oof_df["ts"] = pd.to_datetime(oof_df["ts"])
    oof_df["error"] = oof_df["target_mw"] - oof_df["pred_mw"]
    hw_b = float(oof_df[(oof_df["ws_120"] >= 12) & (oof_df["ws_120"] < 18)]["error"].mean())
    co_b = float(oof_df[oof_df["ws_120"] >= 18]["error"].mean()) if len(oof_df[oof_df["ws_120"] >= 18]) > 5 else 0
    q1_b = float(oof_df[oof_df["ts"].dt.month.isin([1, 2, 3])]["error"].mean())
    print(f"\n  Recency model biases: hw={hw_b:+.3f} co={co_b:+.3f} q1={q1_b:+.3f}")

    # Write submissions
    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()

    # Reorder to submission order
    pred_ordered = np.empty_like(pred_cf_mw)
    pred_ordered[order] = pred_cf_mw

    # Raw (no correction)
    _write(pred_ordered, OUTPUT_DIR / "v125.0_recency_raw.csv", "raw recency-weighted")

    # With v97b-style corrections
    ws_valid = df_valid_sorted["wind_speed_120m"].to_numpy()
    ws_ordered = np.empty_like(ws_valid)
    ws_ordered[order] = ws_valid

    hw_mask = (ws_ordered >= 12) & (ws_ordered < 18)
    co_mask = ws_ordered >= 18

    # Apply corrections with this model's own biases
    corr = pred_ordered.copy()
    corr[hw_mask] += hw_b * 0.7
    corr[co_mask] += co_b * 0.5
    corr += q1_b * 0.7
    _write(np.clip(corr, 0, CAPACITY_MW), OUTPUT_DIR / "v125.1_recency_corrected.csv",
           "recency + own corrections")

    # Blend with best v97b corrected (v123.B recipe)
    v97b_best = pd.read_csv(OUTPUT_DIR / "v123.B_hw0.7_q0.7.csv")
    v97b_preds = v97b_best[TARGET_COL_NAME].to_numpy(dtype=np.float64)

    for w_new in [0.3, 0.5, 0.7]:
        blend = np.clip(w_new * np.clip(corr, 0, CAPACITY_MW) + (1 - w_new) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v125.2_blend_{int(w_new*100)}_v97b_{int((1-w_new)*100)}.csv",
               f"{w_new:.0%} recency + {1-w_new:.0%} v97b_corr")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
