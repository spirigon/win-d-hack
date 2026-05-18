"""V102: V99 + multi-fold probe (weighted average importance across all folds).

The single fold-5 probe is biased toward fold-5's temporal characteristics.
Features important for fold-3 or fold-4 may be excluded.  This version runs
three probes (folds 3, 4, 5) and selects K by weighted-average GAIN importance:
    fold weight = [1, 1, 2]  (fold-5 2× because it's nearest to the test period)

Also adds longer ERA5 rolling windows: 48h and 72h.

K=90, corr dedup, GEM/ICON-G, 5 seeds, CPU.

Outputs:
    data/processed/v102_oof.parquet
    submissions/archive/v102.0_multifold_probe.csv
"""

from __future__ import annotations

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
OOF_PATH    = _ROOT / "data" / "processed" / "v102_oof.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v102.0_multifold_probe.csv"

MARCH_WEIGHT   = 3.0
BLEND_WEIGHT_MW = 0.0  # CF-only proven optimal by blend_weight_optimizer
SEEDS          = SEEDS_5
K              = 90
CORR_THRESHOLD = 0.999
# Probe weights: fold-3 x1, fold-4 x1, fold-5 x2 (more recent = more similar to test)
PROBE_WEIGHTS  = {2: 1.0, 3: 1.0, 4: 2.0}   # fold_idx → weight

_GPU_OVERRIDES: dict = {}


def _add_longer_era5_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add 48h and 72h ERA5 rolling windows (beyond the 24h in _add_era5_rolling)."""
    df = df.copy()
    ws = df["era5_wind_speed_100m"]
    for w in [48, 72]:
        roll = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = roll.mean()
        df[f"era5_ws100_roll_std_{w}h"]  = roll.std().fillna(0)
    df["era5_ws100_diff24"] = ws.diff(24).fillna(0)  # 24h wind tendency
    df["era5_ws100_diff48"] = ws.diff(48).fillna(0)  # 48h wind tendency (synoptic change)
    return df


def _corr_dedup(feat_cols: list[str], X: np.ndarray, threshold: float) -> list[str]:
    n, p = X.shape
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X_filled = X.copy()
    for j in range(p):
        X_filled[nan_mask[:, j], j] = col_means[j]
    col_stds = X_filled.std(axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    X_normed = (X_filled - X_filled.mean(axis=0)) / col_stds
    corr = (X_normed.T @ X_normed) / n
    kept: list[int] = []
    for i in range(p):
        if not kept:
            kept.append(i)
            continue
        if float(np.max(np.abs(corr[i, kept]))) <= threshold:
            kept.append(i)
    return [feat_cols[i] for i in kept]


def _run_probe(
    combined, fold, sample_weight_full, all_extra_cols, feat_cols_deduplicated,
) -> dict[str, float]:
    """Run fold probe and return {feature_name: gain_importance}."""
    probe_combined = add_walk_forward_availability(
        combined, train_end=fold.train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=fold.train_end,
    )
    probe_train = probe_combined[probe_combined["_split"] == "train"].reset_index(drop=True)
    train_idx, val_idx = split_indices(probe_train, fold)
    fold_train_ = probe_train.iloc[train_idx]
    fit_data_ = fold_train_[~fold_train_["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)
    df_t_ = _add_pc(fold_train_, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = _add_pc(probe_train.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)

    feat_cols = [c for c in feat_cols_deduplicated if c in df_t_.columns]

    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = _to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = _to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    sw_fold = sample_weight_full[train_idx]

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(df_t_[feat_cols].to_numpy(dtype=np.float32),
                      label=y_t_cf_, weight=sw_fold,
                      feature_name=feat_cols, free_raw_data=False)
    dv_ = lgb.Dataset(df_v_[feat_cols].to_numpy(dtype=np.float32),
                      label=y_v_cf_, feature_name=feat_cols, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    return {f: float(g) for f, g in zip(feat_cols, imp.tolist(), strict=True)}


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
    set_global_seed(42)

    print("=" * 72)
    print(f"V102: multi-fold probe + 48h/72h rolling  ({len(SEEDS)} seeds, K={K})")
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
    combined = _add_longer_era5_rolling(combined)   # NEW: 48h + 72h windows
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
    print(f"  Curtailment: {n_curtail} rows ({100*n_curtail/len(df_train_full_pre):.1f}%)")
    print(f"  March rows x{MARCH_WEIGHT}: {n_march} ({100*n_march/len(df_train_full_pre):.1f}%)")

    print(f"[2/4] Multi-fold probe (folds {sorted(PROBE_WEIGHTS)}, K={K})...")
    folds = default_folds()

    # Build deduped pool using fold-5 data (proxy; dedup is data-independent for most features)
    fold5 = folds[-1]
    probe_combined_f5 = add_walk_forward_availability(
        combined, train_end=fold5.train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=fold5.train_end,
    )
    probe_train_f5 = probe_combined_f5[probe_combined_f5["_split"] == "train"].reset_index(drop=True)
    t5_idx, _ = split_indices(probe_train_f5, fold5)
    fold_train_f5 = probe_train_f5.iloc[t5_idx]
    fit_data_f5 = fold_train_f5[~fold_train_f5["_is_impossible"]]
    pc_s_f5 = fit_sector_isotonic(fit_data_f5, n_sectors=8)
    pc_g_f5 = IsotonicPowerCurve().fit(fit_data_f5["v_eff"], fit_data_f5[TARGET_COL])
    w_f5 = fit_wake_lookup(fit_data_f5, n_sectors=16)
    df_t_f5 = _add_pc(fold_train_f5, pc_s_f5, pc_g_f5)
    df_t_f5 = add_wake_features(df_t_f5, w_f5)

    all_extra_cols = list(dict.fromkeys(
        nwp_cols + nasa_cols + mnwp_cols + adv_cols + hub_cols + seas_cols + gem_cols
    ))
    feat_cols_all = [
        c for c in feature_columns(df_t_f5)
        if c not in ("_is_impossible", "_split", TARGET_COL) and c != "avail_underprod_mw"
    ]
    for col in all_extra_cols:
        if col in df_t_f5.columns and col not in feat_cols_all:
            feat_cols_all.append(col)

    # Byte-identical dedup
    seen_keys: dict[bytes, str] = {}
    dedup_cols: list[str] = []
    for col in feat_cols_all:
        try:
            key = df_t_f5[col].to_numpy(dtype=np.float32).tobytes()
            if key not in seen_keys:
                seen_keys[key] = col
                dedup_cols.append(col)
        except Exception:
            dedup_cols.append(col)
    n_byte_removed = len(feat_cols_all) - len(dedup_cols)
    feat_cols_all = dedup_cols
    print(f"  After byte-identical dedup: {len(feat_cols_all)} (removed {n_byte_removed})")

    # Correlation dedup
    X_probe = df_t_f5[feat_cols_all].to_numpy(dtype=np.float64)
    t_corr = time.time()
    feat_cols_all = _corr_dedup(feat_cols_all, X_probe, CORR_THRESHOLD)
    n_corr_removed = len(dedup_cols) - len(feat_cols_all)
    print(f"  After correlation dedup (|r|>{CORR_THRESHOLD}): {len(feat_cols_all)} "
          f"(removed {n_corr_removed} near-duplicates, {time.time()-t_corr:.1f}s)")

    # Multi-fold probe: run probe on each fold and aggregate importance
    all_imp: dict[str, float] = {f: 0.0 for f in feat_cols_all}
    total_weight = sum(PROBE_WEIGHTS.values())
    for fold_idx, weight in sorted(PROBE_WEIGHTS.items()):
        t0 = time.time()
        fold = folds[fold_idx]
        imp_dict = _run_probe(combined, fold, sample_weight_full, all_extra_cols, feat_cols_all)
        for f in feat_cols_all:
            all_imp[f] += weight * imp_dict.get(f, 0.0)
        top3 = [k for k, _ in sorted(imp_dict.items(), key=lambda x: -x[1])[:3]]
        print(f"  Fold {fold_idx+1} probe: top-3={top3}  ({time.time()-t0:.0f}s, weight={weight})")

    # Normalize by total weight and select top-K
    for f in all_imp:
        all_imp[f] /= total_weight
    feat_imp = sorted(all_imp.items(), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    top3_agg = [n for n, _ in feat_imp[:3]]
    print(f"  Weighted top-{K} selected  (top-3: {top3_agg})")
    for label, cols in [("hub_height", hub_cols), ("seasonal", seas_cols), ("gem_icon", gem_cols)]:
        hit = [c for c in top_k if c in cols]
        print(f"  {label} in top-{K}: {len(hit)}  {hit}")

    print(f"\n[3/4] Training CV-bag (folds {[i + 1 for i in FOLD_IDS]}, "
          f"CF+MW blend, seeds={SEEDS})...")
    test_cf_per_fold: dict[int, np.ndarray] = {}
    test_mw_per_fold: dict[int, np.ndarray] = {}
    oof_records: list[dict] = []

    for target_mode in ("cf", "mw"):
        print(f"\n  --- {target_mode.upper()} ---")
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
            X_tr   = df_tr[top_k].to_numpy(dtype=np.float32)
            X_va   = df_va[top_k].to_numpy(dtype=np.float32)
            X_test = df_te[top_k].to_numpy(dtype=np.float32)
            ws_tr  = df_tr["wind_speed_120m"].to_numpy()
            sw     = sample_weight_full[train_idx]

            y_tr = _to_cf(y_tr_mw, active_tr) if target_mode == "cf" else y_tr_mw
            y_va = _to_cf(y_va_mw, active_va) if target_mode == "cf" else y_va_mw

            val_pred, test_pred = _train_fold_ensemble(
                X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, sw, SEEDS,
            )
            if target_mode == "cf":
                val_mw = np.clip(_from_cf(val_pred, active_va), 0, CAPACITY_MW)
                test_cf_per_fold[fold_idx] = test_pred
            else:
                val_mw = np.clip(val_pred, 0, CAPACITY_MW)
                test_mw_per_fold[fold_idx] = test_pred

            fold_nmae = float(normalized_mae(y_va_mw, val_mw))
            print(f"    Fold {fold_idx + 1}: {target_mode.upper()} nMAE={fold_nmae:.4f}%  "
                  f"({time.time() - t0:.0f}s)")

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

    oof_df = pd.DataFrame(oof_records)
    oof_wide = (
        oof_df.pivot_table(
            index=["fold", "ts", "target_mw", "active_turbines", "ws_120"],
            columns="target_mode", values="raw_pred",
        ).reset_index()
    )
    oof_wide.columns.name = None
    oof_wide["pred_cf_mw"]    = np.clip(
        _from_cf(oof_wide["cf"].to_numpy(), oof_wide["active_turbines"].to_numpy()),
        0, CAPACITY_MW,
    )
    oof_wide["pred_mw_mw"]    = np.clip(oof_wide["mw"].to_numpy(), 0, CAPACITY_MW)
    oof_wide["pred_blend_mw"] = (
        BLEND_WEIGHT_MW * oof_wide["pred_mw_mw"] + (1 - BLEND_WEIGHT_MW) * oof_wide["pred_cf_mw"]
    ).clip(0, CAPACITY_MW)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_wide.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")
    for fid in [3, 4, 5]:
        sub = oof_wide[oof_wide["fold"] == fid]
        nb = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_blend_mw"].to_numpy()))
        nc = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_cf_mw"].to_numpy()))
        print(f"    Fold {fid} blend nMAE: {nb:.4f}%  CF nMAE: {nc:.4f}%")
    all_blend = float(normalized_mae(
        oof_wide["target_mw"].to_numpy(), oof_wide["pred_blend_mw"].to_numpy()
    ))
    print(f"    All-fold blend nMAE: {all_blend:.4f}%")

    print("\n[4/4] Building final submission...")
    test_combined = add_walk_forward_availability(
        combined, train_end=train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=train_end,
    )
    df_valid_sorted = test_combined[test_combined["_split"] == "valid"].reset_index(drop=True)

    avg_cf = np.mean([test_cf_per_fold[i] for i in FOLD_IDS], axis=0)
    avg_mw = np.mean([test_mw_per_fold[i] for i in FOLD_IDS], axis=0)
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    pred_cf_mw = np.clip(_from_cf(avg_cf, active_valid), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(avg_mw, 0, CAPACITY_MW)
    final_mw   = np.clip(
        BLEND_WEIGHT_MW * pred_mw_mw + (1 - BLEND_WEIGHT_MW) * pred_cf_mw,
        0.0, CAPACITY_MW,
    ).astype(np.float64)

    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po    = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    ts_po  = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
