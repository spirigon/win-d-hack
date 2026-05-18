"""V115: V97b + AIFS and GraphCast AI weather model features.

AIFS (ECMWF AI Integrated Forecasting System) and GraphCast provide
genuinely independent NWP signals (r=0.93 vs ERA5 at 100m). Coverage:
  - Valid/test set (2026 Q1): 100% AIFS, 100% GraphCast
  - Training: AIFS from 2025-02-17 (22% of train rows, all NaN in probe)
  - Training: GraphCast from 2024-02-05 (48% of train rows, ~28% in probe)

Strategy:
  - Standard K=90 fold-5 probe — GraphCast competes normally (10mo train coverage)
  - AIFS force-included after probe (all-NaN in probe training, can't win fairly)
  - Final features: top_90_base + aifs_forced (6 features) + any GC selected

Outputs:
    data/processed/v115_oof.parquet
    submissions/archive/v115.0_cfonly.csv
    submissions/archive/v115.1_blend50.csv
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
from src.features.datasheet_power_curve import add_datasheet_power_features, farm_theoretical_power_mw
from src.features.era5_v2 import merge_era5_v2
from src.features.extras import add_extra_features
from src.features.gem_icon_features import add_gem_icon_features, gem_icon_columns
from src.features.hub_height_features import add_hub_height_features, hub_height_columns
from src.features.multi_nwp_features import (
    build_nwp_consensus, merge_ecmwf_ifs, merge_gfs, multi_nwp_columns,
)
from src.features.nasa_features import merge_nasa_merra2, nasa_columns
from src.features.nwp_ensemble import merge_nwp_ensemble, nwp_ensemble_columns
from src.features.physics import compute_air_density, compute_v_eff
from src.features.pipeline import build_features, feature_columns
from src.features.seasonal_features import add_seasonal_features, seasonal_columns
from src.features.wake import add_wake_features, fit_wake_lookup
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, K, BLEND_WEIGHT_MW, FOLD_IDS, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH     = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH     = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH      = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
AI_MODELS_PATH = _ROOT / "data" / "external" / "ai_weather_models.parquet"
OOF_PATH          = _ROOT / "data" / "processed" / "v115_oof.parquet"
OUTPUT_PATH       = _ROOT / "submissions" / "archive" / "v115.0_cfonly.csv"
OUTPUT_BLEND_PATH = _ROOT / "submissions" / "archive" / "v115.1_blend50.csv"

MARCH_WEIGHT = 3.0
BLEND_WEIGHT_MW = 0.0  # CF-only OOF submission; blend50 written separately
SEEDS = SEEDS_5
K = 90

_GPU_OVERRIDES: dict = {}

# AIFS features to force-include after K probe (all NaN in probe training period)
_AIFS_FORCE_COLS = [
    "aifs_wind_speed_100m",
    "aifs_v_eff_100m",
    "aifs_ds_farm_mw",
    "aifs_vs_era5_ws100m",
    "aifs_dir100m_sin",
    "aifs_dir100m_cos",
]

# GraphCast derived feature names (included in probe normally)
_GC_COLS = [
    "graphcast_wind_speed_10m",
    "graphcast_ws100m_extrap",
    "graphcast_vs_era5_ws10m",
    "graphcast_dir10m_sin",
    "graphcast_dir10m_cos",
]


def _merge_ai_models(combined: pd.DataFrame) -> pd.DataFrame:
    """Left-join ai_weather_models on timestamp. NaN where model has no forecast."""
    ai = pd.read_parquet(AI_MODELS_PATH)
    # Convert object columns (parquet schema issue) to float
    for col in ai.columns:
        if ai[col].dtype == object:
            ai[col] = pd.to_numeric(ai[col], errors="coerce")
    ai = ai.rename(columns={"time": TIMESTAMP_COL})
    combined = combined.merge(ai, on=TIMESTAMP_COL, how="left")
    return combined


def _add_ai_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute derived AIFS and GraphCast features."""
    df = combined.copy()

    # --- AIFS features ---
    if "aifs_wind_speed_100m" in df.columns:
        ws_aifs = df["aifs_wind_speed_100m"].astype(float)

        # Density-corrected v_eff using local pressure/temp
        rho = compute_air_density(
            df["pressure_msl"].astype(float),
            df["temperature_80m"].astype(float),
        )
        df["aifs_v_eff_100m"] = compute_v_eff(ws_aifs, rho).astype(np.float32)

        # Theoretical farm power from AIFS wind speed
        active = df["active_turbines"].to_numpy(dtype=float)
        ws_arr = ws_aifs.to_numpy(dtype=float)
        rho_arr = rho.to_numpy(dtype=float)
        # NaN-safe: compute where ws not null, leave NaN elsewhere
        valid_mask = np.isfinite(ws_arr) & np.isfinite(rho_arr)
        ds_mw = np.full(len(df), np.nan, dtype=np.float32)
        ds_mw[valid_mask] = farm_theoretical_power_mw(
            ws_arr[valid_mask], rho_arr[valid_mask], active[valid_mask]
        ).astype(np.float32)
        df["aifs_ds_farm_mw"] = ds_mw

        # Disagreement vs ERA5 100m wind speed
        if "era5_wind_speed_100m" in df.columns:
            df["aifs_vs_era5_ws100m"] = (
                ws_aifs - df["era5_wind_speed_100m"].astype(float)
            ).astype(np.float32)

        # Direction components
        if "aifs_wind_direction_100m" in df.columns:
            dir_rad = np.deg2rad(df["aifs_wind_direction_100m"].astype(float).to_numpy())
            df["aifs_dir100m_sin"] = np.sin(dir_rad).astype(np.float32)
            df["aifs_dir100m_cos"] = np.cos(dir_rad).astype(np.float32)

    # --- GraphCast features ---
    if "graphcast_wind_speed_10m" in df.columns:
        gc10 = df["graphcast_wind_speed_10m"].astype(float)

        # Extrapolate to 100m via 1/7 power law
        df["graphcast_ws100m_extrap"] = (gc10 * (100 / 10) ** (1 / 7)).astype(np.float32)

        # Disagreement vs ERA5 10m
        if "wind_speed_10m" in df.columns:
            df["graphcast_vs_era5_ws10m"] = (
                gc10 - df["wind_speed_10m"].astype(float)
            ).astype(np.float32)

        # Direction components
        if "graphcast_wind_direction_10m" in df.columns:
            dir_rad = np.deg2rad(
                df["graphcast_wind_direction_10m"].astype(float).to_numpy()
            )
            df["graphcast_dir10m_sin"] = np.sin(dir_rad).astype(np.float32)
            df["graphcast_dir10m_cos"] = np.cos(dir_rad).astype(np.float32)

    return df


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
    print(f"V115: V97b + AIFS + GraphCast  ({len(SEEDS)} seeds, K={K})")
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

    # AI weather model features
    combined = _merge_ai_models(combined)
    combined = _add_ai_features(combined)

    nwp_cols  = nwp_ensemble_columns(combined)
    nasa_cols = nasa_columns(combined)
    mnwp_cols = multi_nwp_columns(combined)
    adv_cols  = advanced_interaction_columns(combined)
    hub_cols  = hub_height_columns(combined)
    seas_cols = seasonal_columns(combined)
    gem_cols  = gem_icon_columns(combined)
    gc_cols   = [c for c in _GC_COLS if c in combined.columns]
    aifs_force = [c for c in _AIFS_FORCE_COLS if c in combined.columns]

    # Coverage report
    for src, col in [("AIFS", "aifs_wind_speed_100m"), ("GraphCast", "graphcast_wind_speed_10m")]:
        if col in combined.columns:
            n_valid = combined[col].notna().sum()
            pct = 100 * n_valid / len(combined)
            tr_mask = combined["_split"] == "train"
            tr_valid = combined.loc[tr_mask, col].notna().sum()
            va_valid = combined.loc[~tr_mask, col].notna().sum()
            print(f"  {src}: {n_valid}/{len(combined)} ({pct:.1f}%) — "
                  f"train={tr_valid}, valid={va_valid}")

    print(f"  Feature groups: nwp={len(nwp_cols)} nasa={len(nasa_cols)} "
          f"mnwp={len(mnwp_cols)} adv={len(adv_cols)} "
          f"hub={len(hub_cols)} seasonal={len(seas_cols)} "
          f"gem_icon={len(gem_cols)} gc_in_probe={len(gc_cols)} "
          f"aifs_forced={len(aifs_force)}")

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

    print(f"\n[2/4] Feature selection probe (Fold-5, K={K}, GraphCast in pool)...")
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

    # Build feature pool: base + GraphCast (for probe); AIFS excluded (all NaN in probe train)
    all_extra_cols = list(dict.fromkeys(
        nwp_cols + nasa_cols + mnwp_cols + adv_cols + hub_cols + seas_cols + gem_cols + gc_cols
    ))
    feat_cols_all = [
        c for c in feature_columns(df_t_)
        if c not in ("_is_impossible", "_split", TARGET_COL) and c != "avail_underprod_mw"
    ]
    for col in all_extra_cols:
        if col in df_t_.columns and col not in feat_cols_all:
            feat_cols_all.append(col)

    # Byte-dedup (ERA5 = ECMWF identical)
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
    print(f"  Feature pool (probe): {len(feat_cols_all)} (removed {n_removed} byte-identical)")

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
    top_k_base = [n for n, _ in feat_imp[:K]]

    top3 = [n for n, _ in feat_imp[:3]]
    print(f"  Top-{K} base selected  (top-3: {top3})")
    for label, cols in [
        ("hub_height", hub_cols), ("seasonal", seas_cols),
        ("gem_icon", gem_cols), ("graphcast", gc_cols),
    ]:
        hit = [c for c in top_k_base if c in cols]
        print(f"  {label} in top-{K}: {len(hit)}  {hit}")

    # Merge: base K + AIFS force-include
    aifs_already_in = [c for c in aifs_force if c in top_k_base]
    aifs_added = [c for c in aifs_force if c not in top_k_base and c in df_t_.columns]
    top_k = top_k_base + aifs_added
    print(f"  AIFS force-added: {len(aifs_added)} {aifs_added}")
    print(f"  AIFS already in top-{K}: {len(aifs_already_in)} {aifs_already_in}")
    print(f"  Final feature count: {len(top_k)}")

    # Free probe copy — not needed during fold loop
    del probe_combined, probe_train, df_t_, df_v_, dt_, dv_, probe
    gc.collect()

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

            # Report AIFS/GC coverage in this fold's val and test
            for src_col in ["aifs_wind_speed_100m", "graphcast_wind_speed_10m"]:
                if src_col in df_va.columns:
                    n_va = df_va[src_col].notna().sum()
                    n_te = df_te[src_col].notna().sum() if src_col in df_te.columns else 0
                    label = src_col.split("_")[0].upper()
                    print(f"      {label} coverage: val={n_va}/{len(df_va)}, test={n_te}/{len(df_te)}")

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
            sys.stdout.flush()

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

            # Free fold-specific large allocations
            del fc, ft, fv, df_tr, df_va, df_te, X_tr, X_va, X_test
            gc.collect()

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

    # AIFS/GC coverage in the submission target
    for src_col in ["aifs_wind_speed_100m", "graphcast_wind_speed_10m"]:
        if src_col in df_valid_sorted.columns:
            n = df_valid_sorted[src_col].notna().sum()
            label = src_col.split("_")[0].upper()
            print(f"  {label} coverage in submission: {n}/{len(df_valid_sorted)} "
                  f"({100*n/len(df_valid_sorted):.1f}%)")

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

    blend50_mw = np.clip(0.5 * pred_mw_mw + 0.5 * pred_cf_mw, 0.0, CAPACITY_MW).astype(np.float64)
    blend50_po = np.empty_like(blend50_mw)
    blend50_po[order] = blend50_mw
    write_submission(blend50_po, OUTPUT_BLEND_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"  Blend-50/50 saved: {OUTPUT_BLEND_PATH}")

    oof_cf_f5  = oof_wide[oof_wide["fold"] == 5]
    cf_f5_nmae = float(normalized_mae(oof_cf_f5["target_mw"].to_numpy(), oof_cf_f5["pred_cf_mw"].to_numpy()))
    b5_nmae    = float(normalized_mae(oof_cf_f5["target_mw"].to_numpy(),
        (0.5 * oof_cf_f5["pred_mw_mw"] + 0.5 * oof_cf_f5["pred_cf_mw"]).to_numpy()))
    print(f"\n  OOF F5: CF-only={cf_f5_nmae:.4f}%  Blend-50={b5_nmae:.4f}%")


if __name__ == "__main__":
    main()
