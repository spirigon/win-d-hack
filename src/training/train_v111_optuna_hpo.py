"""V111: Optuna hyperparameter search for LGBM on the V97b feature set.

The current LGBM hyperparameters (v32 era) were tuned for an earlier pipeline.
This script re-optimises tree-structure params on the V97b feature pipeline:
  - byte-identical dedup, K=90, GEM/ICON-G features, CF-only target.
  - Optuna on Fold-5 only (20 trials, 1 seed, 3 regimes) for speed.
  - Best params → full 3-fold training (5 seeds, 3 regimes).

Outputs:
    data/processed/v111_oof.parquet
    submissions/archive/v111.0_optuna_hpo.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
OOF_PATH    = _ROOT / "data" / "processed" / "v111_oof.parquet"
OUTPUT_PATH       = _ROOT / "submissions" / "archive" / "v111.0_optuna_hpo.csv"
OUTPUT_BLEND_PATH = _ROOT / "submissions" / "archive" / "v111.1_blend50.csv"

MARCH_WEIGHT    = 3.0
SEEDS           = SEEDS_5
K               = 90
BLEND_WEIGHT_MW = 0.0  # CF-only: optimal per blend_weight_optimizer
N_OPTUNA_TRIALS = 20   # fewer trials for speed; fold-5 only
HPO_SEED        = 1    # single seed per trial for HPO speed

_GPU_OVERRIDES: dict = {}


def _train_regime_split(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sw, seeds, params):
    """Regime-split LGBM ensemble with given params."""
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = (np.where(mask, 2.0, 0.3) * sw).astype(np.float32)
        vp, tp = [], []
        for s in seeds:
            cfg_params = {**params, "seed": s, "verbose": -1}
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights,
                             feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg_params, dt, num_boost_round=5000,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(250, verbose=False)],
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
    print(f"V111: Optuna HPO on V97b pipeline  ({len(SEEDS)} seeds, K={K}, "
          f"{N_OPTUNA_TRIALS} trials)")
    print("=" * 72)

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

    sample_weight_full = (~is_impossible_full.to_numpy()[: len(df_train_full_pre)]).astype(np.float32)
    march_mask = df_train_full_pre[TIMESTAMP_COL].dt.month == 3
    sample_weight_full[march_mask] *= MARCH_WEIGHT
    n_curtail = int(impossible.sum())
    n_march   = int(march_mask.sum())
    print(f"  Curtailment: {n_curtail} rows ({100*n_curtail/len(df_train_full_pre):.1f}%)")
    print(f"  March rows x{MARCH_WEIGHT}: {n_march} ({100*n_march/len(df_train_full_pre):.1f}%)")

    print(f"\n[2/5] Feature selection probe (Fold-5, K={K})...")
    folds = default_folds()
    fold5 = folds[-1]

    probe_combined = add_walk_forward_availability(
        combined, train_end=fold5.train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=fold5.train_end,
    )
    probe_train = probe_combined[probe_combined["_split"] == "train"].reset_index(drop=True)

    train_idx_probe, val_idx_probe = split_indices(probe_train, fold5)
    fold_train_probe = probe_train.iloc[train_idx_probe]
    fit_data_probe = fold_train_probe[~fold_train_probe["_is_impossible"]]
    pc_s_probe = fit_sector_isotonic(fit_data_probe, n_sectors=8)
    pc_g_probe = IsotonicPowerCurve().fit(fit_data_probe["v_eff"], fit_data_probe[TARGET_COL])
    w_probe = fit_wake_lookup(fit_data_probe, n_sectors=16)
    df_t_probe = _add_pc(fold_train_probe, pc_s_probe, pc_g_probe)
    df_t_probe = add_wake_features(df_t_probe, w_probe)
    df_v_probe = _add_pc(probe_train.iloc[val_idx_probe], pc_s_probe, pc_g_probe)
    df_v_probe = add_wake_features(df_v_probe, w_probe)

    all_extra_cols = list(dict.fromkeys(
        nwp_cols + nasa_cols + mnwp_cols + adv_cols + hub_cols + seas_cols + gem_cols
    ))
    feat_cols_all = [
        c for c in feature_columns(df_t_probe)
        if c not in ("_is_impossible", "_split", TARGET_COL) and c != "avail_underprod_mw"
    ]
    for col in all_extra_cols:
        if col in df_t_probe.columns and col not in feat_cols_all:
            feat_cols_all.append(col)

    # Byte-identical dedup
    seen_keys: dict[bytes, str] = {}
    dedup_cols: list[str] = []
    for col in feat_cols_all:
        try:
            key = df_t_probe[col].to_numpy(dtype=np.float32).tobytes()
            if key not in seen_keys:
                seen_keys[key] = col
                dedup_cols.append(col)
        except Exception:
            dedup_cols.append(col)
    n_removed = len(feat_cols_all) - len(dedup_cols)
    feat_cols_all = dedup_cols
    print(f"  Feature pool: {len(feat_cols_all)} (removed {n_removed} byte-identical duplicates)")

    # LGBM probe for feature selection
    a_tr_probe = df_t_probe["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_probe = _to_cf(df_t_probe[TARGET_COL].to_numpy(dtype=np.float32), a_tr_probe)
    a_va_probe = df_v_probe["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_probe = _to_cf(df_v_probe[TARGET_COL].to_numpy(dtype=np.float32), a_va_probe)
    sw_probe = sample_weight_full[train_idx_probe]

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_p = lgb.Dataset(df_t_probe[feat_cols_all].to_numpy(dtype=np.float32),
                       label=y_t_cf_probe, weight=sw_probe,
                       feature_name=feat_cols_all, free_raw_data=False)
    dv_p = lgb.Dataset(df_v_probe[feat_cols_all].to_numpy(dtype=np.float32),
                       label=y_v_cf_probe, feature_name=feat_cols_all, free_raw_data=False)
    probe_model = lgb.train(
        cfg0.to_params(), dt_p, num_boost_round=5000,
        valid_sets=[dv_p], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    top3 = [n for n, _ in feat_imp[:3]]
    print(f"  Top-{K} selected  (top-3: {top3})")
    for label, cols in [("hub_height", hub_cols), ("seasonal", seas_cols), ("gem_icon", gem_cols)]:
        hit = [c for c in top_k if c in cols]
        print(f"  {label} in top-{K}: {len(hit)}  {hit}")

    # Prepare fold-5 data for Optuna HPO
    X_tr_hpo = df_t_probe[top_k].to_numpy(dtype=np.float32)
    y_tr_hpo = y_t_cf_probe
    X_va_hpo = df_v_probe[top_k].to_numpy(dtype=np.float32)
    y_va_hpo = y_v_cf_probe
    y_va_mw_hpo = df_v_probe[TARGET_COL].to_numpy(dtype=np.float32)
    ws_tr_hpo = df_t_probe["wind_speed_120m"].to_numpy(dtype=np.float32)
    a_va_hpo = a_va_probe
    sw_hpo = sw_probe

    # Dummy X_test (not used in HPO, just need shape for _train_regime_split)
    X_test_dummy = X_va_hpo[:1]

    print(f"\n[3/5] Optuna HPO  ({N_OPTUNA_TRIALS} trials, Fold-5 only, 1 seed)...")

    BASE_PARAMS = {
        "objective":  "regression",
        "metric":     "rmse",
        "learning_rate": LGBM_PARAMS.learning_rate,  # keep LR fixed
        "bagging_freq": 3,
        "verbose":    -1,
    }

    def optuna_objective(trial: optuna.Trial) -> float:
        params = {
            **BASE_PARAMS,
            "num_leaves":         trial.suggest_int("num_leaves", 40, 180),
            "min_data_in_leaf":   trial.suggest_int("min_data_in_leaf", 5, 50),
            "feature_fraction":   trial.suggest_float("feature_fraction", 0.25, 0.75),
            "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.4, 0.95),
            "lambda_l1":          trial.suggest_float("lambda_l1", 1e-3, 1.0, log=True),
            "lambda_l2":          trial.suggest_float("lambda_l2", 1e-3, 1.0, log=True),
        }
        val_cf, _ = _train_regime_split(
            X_tr_hpo, y_tr_hpo, X_va_hpo, y_va_hpo, X_test_dummy,
            top_k, ws_tr_hpo, sw_hpo, [HPO_SEED], params
        )
        val_mw = _from_cf(np.clip(val_cf, 0, 1), a_va_hpo)
        return float(normalized_mae(y_va_mw_hpo, val_mw))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(optuna_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_trial
    X_va_probe_full = df_v_probe[feat_cols_all].to_numpy(dtype=np.float32)
    print(f"\n  Best F5 CF nMAE: {best.value:.4f}%  (baseline: "
          f"{normalized_mae(y_va_mw_hpo, _from_cf(np.clip(probe_model.predict(X_va_probe_full), 0, 1), a_va_hpo)):.4f}%)")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Build final LGBM params from Optuna best
    BEST_PARAMS = {
        **BASE_PARAMS,
        **best.params,
        "num_boost_round": 5000,
    }

    print(f"\n[4/5] Training with best params (folds {[i + 1 for i in FOLD_IDS]}, "
          f"CF-only, seeds={SEEDS})...")
    test_cf_per_fold: dict[int, np.ndarray] = {}
    test_mw_per_fold: dict[int, np.ndarray] = {}
    oof_records: list[dict] = []

    df_valid_sorted = combined[combined["_split"] == "valid"].sort_values(TIMESTAMP_COL).reset_index(drop=True)

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

            tr_idx, va_idx = split_indices(ft, fold)
            fold_train = ft.iloc[tr_idx]
            fold_val   = ft.iloc[va_idx]
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
            sw     = sample_weight_full[tr_idx]

            y_tr = _to_cf(y_tr_mw, active_tr) if target_mode == "cf" else y_tr_mw
            y_va = _to_cf(y_va_mw, active_va) if target_mode == "cf" else y_va_mw

            val_pred, test_pred = _train_regime_split(
                X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, sw, SEEDS, BEST_PARAMS,
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
            ws_va = df_va["wind_speed_120m"].to_numpy()
            for i in range(len(y_va_mw)):
                oof_records.append({
                    "fold": int(fold_idx + 1),
                    "ts": pd.Timestamp(ts_va[i]),
                    "target_mw": float(y_va_mw[i]),
                    "active_turbines": float(active_va[i]),
                    "ws_120": float(ws_va[i]),
                    "raw_pred": float(val_pred[i]),
                    "target_mode": target_mode,
                })

    # OOF — pivot target_mode to columns, compute derived predictions
    oof_df = pd.DataFrame(oof_records)
    oof_wide = (
        oof_df.pivot_table(
            index=["fold", "ts", "target_mw", "active_turbines", "ws_120"],
            columns="target_mode", values="raw_pred",
        ).reset_index()
    )
    oof_wide.columns.name = None
    oof_wide["pred_cf_mw"] = np.clip(
        _from_cf(oof_wide["cf"].to_numpy(), oof_wide["active_turbines"].to_numpy()),
        0, CAPACITY_MW,
    )
    oof_wide["pred_mw_mw"] = np.clip(oof_wide["mw"].to_numpy(), 0, CAPACITY_MW)
    oof_wide["pred_blend_mw"] = (
        BLEND_WEIGHT_MW * oof_wide["pred_mw_mw"] + (1 - BLEND_WEIGHT_MW) * oof_wide["pred_cf_mw"]
    ).clip(0, CAPACITY_MW)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_wide.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")
    for fid in sorted(oof_wide["fold"].unique()):
        sub = oof_wide[oof_wide["fold"] == fid]
        nb = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_blend_mw"].to_numpy()))
        nc = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_cf_mw"].to_numpy()))
        print(f"    Fold {fid} blend nMAE: {nb:.4f}%  CF nMAE: {nc:.4f}%")
    all_blend = float(normalized_mae(oof_wide["target_mw"].to_numpy(), oof_wide["pred_blend_mw"].to_numpy()))
    print(f"    All-fold blend nMAE: {all_blend:.4f}%")

    print("\n[5/5] Building final submission...")
    # Average CF preds across folds, then convert to MW
    test_cf_avg = np.mean(list(test_cf_per_fold.values()), axis=0)
    final_mw = _from_cf(
        np.clip(test_cf_avg, 0, 1),
        df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32),
    )

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

    # 50/50 CF+MW blend — LB favors blend over CF-only (~+0.04pp, confirmed v97b vs v97b.0)
    test_mw_avg = np.mean(list(test_mw_per_fold.values()), axis=0)
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    cf_mw = np.clip(_from_cf(np.clip(test_cf_avg, 0, 1), active_valid), 0.0, CAPACITY_MW)
    mw_mw = np.clip(test_mw_avg, 0.0, CAPACITY_MW)
    blend50_mw = (0.5 * cf_mw + 0.5 * mw_mw).astype(np.float64)
    blend50_po = np.empty(n_valid, dtype=np.float64)
    blend50_po[order] = blend50_mw
    write_submission(blend50_po, OUTPUT_BLEND_PATH, expected_rows=n_valid, timestamps=ts_po)
    print(f"  Blend-50/50 saved: {OUTPUT_BLEND_PATH}")
    print(f"  Blend-50 mean: {blend50_mw.mean():.2f} MW")


if __name__ == "__main__":
    main()
