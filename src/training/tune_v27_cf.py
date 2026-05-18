"""V27: K-sweep + Optuna HP tuning for the CF-target regime-specialist pipeline.

Two phases in one script:

Phase 1 — K-sweep (fast, ~15 min)
    Test K ∈ {55, 60, 65, 70, 75, 80} on Fold-5 with the current v15 HPs
    and CF target. Picks the K with the lowest Fold-5 nMAE.

Phase 2 — Optuna HP tuning (~2 h, 100 trials)
    Tune num_leaves, min_data_in_leaf, learning_rate, feature_fraction,
    bagging_fraction, bagging_freq, lambda_l1, lambda_l2 using the winning K.
    Objective: 0.7 × Fold-5 + 0.3 × Fold-4 (multi-fold robustness).
    Saves best params to configs/model/lgbm_v27_cf_tuned.yaml and the
    Optuna study to models/optuna_v27_cf_study.pkl.

Usage:
    python -m src.training.tune_v27_cf                  # full run
    python -m src.training.tune_v27_cf --skip-ksweep    # skip K-sweep, use K=70
    python -m src.training.tune_v27_cf --n-trials 50    # fewer Optuna trials
    python -m src.training.tune_v27_cf --skip-optuna    # K-sweep only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.pipeline import build_features, feature_columns
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

# -------------------------------------------------------------------- paths
TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
STUDY_PATH = _ROOT / "models" / "optuna_v27_cf_study.pkl"
CONFIG_OUT = _ROOT / "configs" / "model" / "lgbm_v27_cf_tuned.yaml"
REPORT_PATH = _ROOT / "data" / "processed" / "v27_ksweep_report.json"

# -------------------------------------------------------------------- constants
TURBINE_RATED_MW = 3.465
SEEDS_SPEC = [42, 123, 456, 789, 2026]  # 5 seeds per specialist (same as v15)
K_CANDIDATES = [55, 60, 65, 70, 75, 80]

# Current best HPs (v14 Optuna, raw-MW target). Used as probe config and
# as the Optuna starting point.
CURRENT_CONFIG = LGBMConfig(
    num_leaves=86,
    min_data_in_leaf=14,
    learning_rate=0.00844,
    feature_fraction=0.430,
    bagging_fraction=0.564,
    bagging_freq=3,
    lambda_l1=0.253,
    lambda_l2=0.00971,
    num_boost_round=5000,
    early_stopping_rounds=250,
    log_period=0,
)


# ============================================================ feature helpers


def _load_train_raw(path: str | Path) -> pd.DataFrame:
    """Read raw CSV preserving all weather columns (bypasses GenerationSchema filter)."""
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")
    return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)


def _add_pc(df, pc_sector, pc_global):
    df = df.copy()
    v_eff = df["v_eff"].to_numpy()
    dir_deg = (df["wind_direction_120m"] * 1000.0).to_numpy()
    df["p_curve_sector"] = pc_sector.predict(v_eff, dir_deg)
    df["p_curve_global"] = pc_global.predict(v_eff)
    df["p_curve_rews"] = pc_global.predict(df["rews"].to_numpy())
    df["p_curve_x_active"] = df["p_curve_sector"] * df["active_turbines_ratio"]
    df["p_curve_global_x_active"] = df["p_curve_global"] * df["active_turbines_ratio"]
    df["p_curve_ratio"] = df["p_curve_sector"] / CAPACITY_MW
    df["p_curve_sector_minus_global"] = df["p_curve_sector"] - df["p_curve_global"]
    return df


def _merge_era5(df, era5):
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")
    df["ws10_bias"] = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"] = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"] = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    era5_new = [
        c for c in df.columns
        if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"
    ]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def _add_era5_rolling(df):
    df = df.copy()
    ws = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]
    for w in [3, 6, 12, 24]:
        roll = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = roll.mean()
        df[f"era5_ws100_roll_std_{w}h"] = roll.std().fillna(0)
    df["era5_ws100_diff1"] = ws.diff(1).fillna(0)
    df["era5_ws100_diff3"] = ws.diff(3).fillna(0)
    df["era5_ws100_diff6"] = ws.diff(6).fillna(0)
    df["era5_pressure_diff3"] = pres.diff(3).fillna(0)
    df["era5_pressure_diff6"] = pres.diff(6).fillna(0)
    df["era5_pressure_diff12"] = pres.diff(12).fillna(0)
    roll6 = ws.rolling(6, min_periods=1)
    df["era5_turb_intensity_6h"] = roll6.std().fillna(0) / (roll6.mean() + 1e-3)
    df["era5_dir_sin_diff1"] = df["era5_dir100_sin"].diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = df["era5_dir100_cos"].diff(1).fillna(0)
    return df


def _to_cf(y_mw, active_turbines):
    denom = np.maximum(active_turbines.astype(np.float32) * TURBINE_RATED_MW, 1e-3)
    return (y_mw / denom).astype(np.float32)


def _from_cf(cf, active_turbines):
    cf = np.clip(cf, 0.0, 1.0)
    return (cf * active_turbines.astype(np.float32) * TURBINE_RATED_MW).astype(np.float32)


# ============================================================ data preparation


def _prepare_splits() -> dict[str, dict]:
    """Build Fold-4 and Fold-5 splits with all features.

    Returns a dict keyed by fold name, each containing:
        df_tr, df_va, feat_cols_all, feat_imp_sorted
    """
    print("Loading and building features...")
    df_train_raw = _load_train_raw(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train_raw["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train_raw, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = _merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = _add_era5_rolling(combined)

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    folds = default_folds()
    splits = {}
    for fold_name in ("fold4_2024Q4", "fold5_2025Q1"):
        fold = next(f for f in folds if f.name == fold_name)
        train_idx, val_idx = split_indices(df_train, fold)
        fold_train = df_train.iloc[train_idx]
        fit_data = fold_train[~fold_train["_is_impossible"]]

        pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        wake = fit_wake_lookup(fit_data, n_sectors=16)

        df_tr = _add_pc(fold_train, pc_sector, pc_global)
        df_tr = add_wake_features(df_tr, wake)
        df_va = _add_pc(df_train.iloc[val_idx], pc_sector, pc_global)
        df_va = add_wake_features(df_va, wake)

        feat_cols_all = [
            c for c in feature_columns(df_tr)
            if c not in ("_is_impossible", "_split", TARGET_COL)
        ]
        sample_weight = (~fold_train["_is_impossible"].to_numpy()).astype(np.float32)

        # Probe run to get feature importance ordering (CF target).
        active_tr = df_tr["active_turbines"].to_numpy()
        active_va = df_va["active_turbines"].to_numpy()
        y_tr_cf = _to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
        y_va_cf = _to_cf(df_va[TARGET_COL].to_numpy(dtype=np.float32), active_va)

        X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
        X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)

        probe_cfg = LGBMConfig(**{**CURRENT_CONFIG.__dict__, "seed": 42})
        dtrain = lgb.Dataset(
            X_tr_all, label=y_tr_cf, weight=sample_weight,
            feature_name=feat_cols_all, free_raw_data=False,
        )
        dval = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
        probe = lgb.train(
            probe_cfg.to_params(), dtrain, num_boost_round=5000,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(250, verbose=False)],
        )
        imp = probe.feature_importance(importance_type="gain")
        feat_imp_sorted = sorted(zip(feat_cols_all, imp, strict=True), key=lambda x: -x[1])

        splits[fold_name] = {
            "df_tr": df_tr,
            "df_va": df_va,
            "feat_imp_sorted": feat_imp_sorted,
            "sample_weight": sample_weight,
            "active_tr": active_tr,
            "active_va": active_va,
            "y_tr_mw": df_tr[TARGET_COL].to_numpy(dtype=np.float32),
            "y_va_mw": df_va[TARGET_COL].to_numpy(dtype=np.float32),
            "y_tr_cf": y_tr_cf,
            "y_va_cf": y_va_cf,
            "ws_tr": df_tr["wind_speed_120m"].to_numpy(),
        }
        print(f"  {fold_name}: {len(df_tr)} train, {len(df_va)} val rows")

    return splits


# ============================================================ specialist ensemble


def _run_specialists(
    X_tr: np.ndarray,
    y_tr_cf: np.ndarray,
    X_va: np.ndarray,
    y_va_cf: np.ndarray,
    y_va_mw: np.ndarray,
    active_va: np.ndarray,
    ws_tr: np.ndarray,
    feat_cols: list[str],
    sample_weight: np.ndarray,
    params: dict,
    seeds: list[int] = SEEDS_SPEC,
) -> float:
    """Run 3 regime specialists, return Fold nMAE (MW)."""
    regime_preds = []
    for _name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        seed_preds = []
        for s in seeds:
            cfg_params = {**params, "seed": s}
            dtrain = lgb.Dataset(
                X_tr, label=y_tr_cf, weight=weights,
                feature_name=feat_cols, free_raw_data=False,
            )
            dval = lgb.Dataset(X_va, label=y_va_cf, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg_params, dtrain,
                num_boost_round=params.get("num_boost_round", 5000),
                valid_sets=[dval], valid_names=["val"],
                callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 250), verbose=False)],
            )
            seed_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds.append(np.mean(seed_preds, axis=0))

    avg_cf = np.mean(regime_preds, axis=0)
    avg_mw = np.clip(_from_cf(avg_cf, active_va), 0.0, CAPACITY_MW)
    return float(normalized_mae(y_va_mw, avg_mw))


# ============================================================ Phase 1: K-sweep


def run_k_sweep(splits: dict) -> int:
    """Test K ∈ K_CANDIDATES on Fold-5 with current HPs. Return best K."""
    print("\n" + "=" * 60)
    print("Phase 1: K-sweep")
    print("=" * 60)

    fold5 = splits["fold5_2025Q1"]
    feat_imp = fold5["feat_imp_sorted"]
    results = {}

    base_params = {
        **CURRENT_CONFIG.to_params(),
        "num_boost_round": 5000,
        "early_stopping_rounds": 250,
    }

    for k in K_CANDIDATES:
        top_k = [n for n, _ in feat_imp[:k]]
        X_tr = fold5["df_tr"][top_k].to_numpy(dtype=np.float32)
        X_va = fold5["df_va"][top_k].to_numpy(dtype=np.float32)

        nmae = _run_specialists(
            X_tr, fold5["y_tr_cf"], X_va, fold5["y_va_cf"],
            fold5["y_va_mw"], fold5["active_va"], fold5["ws_tr"],
            top_k, fold5["sample_weight"], base_params,
        )
        results[k] = nmae
        print(f"  K={k:3d}: {nmae:.4f}%")

    best_k = min(results, key=results.__getitem__)
    print(f"\n  Best K = {best_k} ({results[best_k]:.4f}%)")

    # Save report.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"k_results": {str(k): v for k, v in results.items()}, "best_k": best_k}, indent=2)
    )
    return best_k


# ============================================================ Phase 2: Optuna


def _optuna_objective(trial: optuna.Trial, splits: dict, top_k_f5: list[str], top_k_f4: list[str]) -> float:
    """Optuna objective: 0.7 × Fold-5 + 0.3 × Fold-4 nMAE."""
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": -1,
        "num_boost_round": 3000,  # capped for speed; best iter × 1.2 used in final train
        "early_stopping_rounds": 200,
        "num_leaves": trial.suggest_int("num_leaves", 50, 200),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 8, 100, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.35, 0.85),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.40, 0.90),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 8),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 2.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
        "seed": 42,
    }

    fold5 = splits["fold5_2025Q1"]
    fold4 = splits["fold4_2024Q4"]

    X_tr5 = fold5["df_tr"][top_k_f5].to_numpy(dtype=np.float32)
    X_va5 = fold5["df_va"][top_k_f5].to_numpy(dtype=np.float32)
    nmae5 = _run_specialists(
        X_tr5, fold5["y_tr_cf"], X_va5, fold5["y_va_cf"],
        fold5["y_va_mw"], fold5["active_va"], fold5["ws_tr"],
        top_k_f5, fold5["sample_weight"], params,
        seeds=[42, 123, 456],  # 3 seeds for speed during tuning
    )

    X_tr4 = fold4["df_tr"][top_k_f4].to_numpy(dtype=np.float32)
    X_va4 = fold4["df_va"][top_k_f4].to_numpy(dtype=np.float32)
    nmae4 = _run_specialists(
        X_tr4, fold4["y_tr_cf"], X_va4, fold4["y_va_cf"],
        fold4["y_va_mw"], fold4["active_va"], fold4["ws_tr"],
        top_k_f4, fold4["sample_weight"], params,
        seeds=[42, 123, 456],
    )

    trial.set_user_attr("fold5_nmae", nmae5)
    trial.set_user_attr("fold4_nmae", nmae4)
    return 0.7 * nmae5 + 0.3 * nmae4


def run_optuna(splits: dict, best_k: int, n_trials: int) -> dict:
    """Run Optuna HP search. Return best params dict."""
    print("\n" + "=" * 60)
    print(f"Phase 2: Optuna ({n_trials} trials, K={best_k})")
    print("=" * 60)

    # Top-K feature lists per fold (each fold has its own importance ordering).
    top_k_f5 = [n for n, _ in splits["fold5_2025Q1"]["feat_imp_sorted"][:best_k]]
    top_k_f4 = [n for n, _ in splits["fold4_2024Q4"]["feat_imp_sorted"][:best_k]]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=20)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Seed with current best params so TPE starts from a good region.
    study.enqueue_trial({
        "num_leaves": 86,
        "min_data_in_leaf": 14,
        "learning_rate": 0.00844,
        "feature_fraction": 0.430,
        "bagging_fraction": 0.564,
        "bagging_freq": 3,
        "lambda_l1": 0.253,
        "lambda_l2": 0.00971,
    })

    def _cb(study: optuna.Study, trial: optuna.Trial) -> None:
        f5 = trial.user_attrs.get("fold5_nmae", float("nan"))
        f4 = trial.user_attrs.get("fold4_nmae", float("nan"))
        best_f5 = study.best_trial.user_attrs.get("fold5_nmae", float("nan"))
        print(
            f"  trial {trial.number:3d}  obj={trial.value:.4f}  "
            f"f5={f5:.4f}  f4={f4:.4f}  best_f5={best_f5:.4f}"
        )

    study.optimize(
        lambda t: _optuna_objective(t, splits, top_k_f5, top_k_f4),
        n_trials=n_trials,
        callbacks=[_cb],
    )

    best = study.best_trial
    print(f"\n  Best objective: {study.best_value:.4f}")
    print(f"  Best Fold-5:   {best.user_attrs['fold5_nmae']:.4f}%")
    print(f"  Best Fold-4:   {best.user_attrs['fold4_nmae']:.4f}%")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Persist study.
    STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDY_PATH, "wb") as fh:
        pickle.dump(study, fh)
    print(f"  Study saved: {STUDY_PATH}")

    # Write YAML config.
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated by tune_v27_cf.py",
        f"# Fold-5 nMAE: {best.user_attrs['fold5_nmae']:.4f}",
        f"# Fold-4 nMAE: {best.user_attrs['fold4_nmae']:.4f}",
        f"# K: {best_k}",
        "",
    ]
    for k, v in best.params.items():
        lines.append(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    CONFIG_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Config saved: {CONFIG_OUT}")

    return best.params


# ============================================================ main


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.training.tune_v27_cf",
        description="K-sweep + Optuna HP tuning for CF-target regime-specialist pipeline.",
    )
    p.add_argument("--n-trials", type=int, default=100, help="Optuna trial count (default 100)")
    p.add_argument("--skip-ksweep", action="store_true", help="Skip K-sweep, use K=70")
    p.add_argument("--skip-optuna", action="store_true", help="Skip Optuna, run K-sweep only")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    set_global_seed(42)
    t0 = time.time()

    splits = _prepare_splits()

    if args.skip_ksweep:
        best_k = 70
        print(f"\nSkipping K-sweep, using K={best_k}")
    else:
        best_k = run_k_sweep(splits)

    if not args.skip_optuna:
        run_optuna(splits, best_k=best_k, n_trials=args.n_trials)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
