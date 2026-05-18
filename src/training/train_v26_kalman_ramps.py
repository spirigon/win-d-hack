"""V26: Fold-5 ablation of the three methods from plan.md worth trying.

Scope
-----
Only evaluates on Fold-5 (the Q1-2026 surrogate). Four configurations
are run end-to-end against the v15-regime baseline (CF target + 3
specialists averaged, Fold-5 ≈ 7.62 % from PROJECT.md):

* **baseline**   — current best recipe, no additions.
* **kalman**     — baseline + Kalman-smoothed NWP↔ERA5 bias added as
                   two features ``ws120_kalman`` and
                   ``ws120_bias_kalman_smooth``.
* **ramps**      — baseline + ``dv_3h``, ``dv_6h``, ``is_sub_cutin``,
                   ``is_sub_cutin_soft`` and their interactions.
* **full**       — baseline + kalman + ramps.

The per-month affine calibration is applied *on top* of each of those
four configurations using OOF predictions from folds 1–4 (the
calibration is fit on OOF → applied on Fold-5). This gives us the
headline decision table:

+---------+----------+---------+
|         | raw      | +affine |
+=========+==========+=========+
|baseline | ...      | ...     |
+---------+----------+---------+
|kalman   | ...      | ...     |
+---------+----------+---------+
|ramps    | ...      | ...     |
+---------+----------+---------+
|full     | ...      | ...     |
+---------+----------+---------+

Stop conditions (mirrors TODO.md):

* If any component regresses Fold-5 by > 0.02 pp we revert it.
* If the per-month affine net lift is < 0.03 pp we do not ship it.
* We do NOT train on the full 2022..2025 range here — this is an
  ablation harness, not a submission builder. Ship-quality training is
  a follow-up once this script picks a winning config.

Usage
-----
    python -m src.training.train_v26_kalman_ramps
"""

from __future__ import annotations

import json
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
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.ramp import add_ramp_features
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.postprocess.calibration import PerMonthAffine
from src.postprocess.kalman import add_kalman_bias_features
from src.utils.seeding import set_global_seed

# -------------------------------------------------------------------- paths
TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_TUNED_PATH = _ROOT / "data" / "processed" / "oof_lgbm_tuned.parquet"
REPORT_DIR = _ROOT / "data" / "processed"
REPORT_PATH = REPORT_DIR / "v26_kalman_ramps_ablation.json"

# -------------------------------------------------------------------- model
# Tuned v14 hyperparameters + v15 ensemble layout. Same as
# train_v15_regime.py so Fold-5 numbers stay comparable.
SEEDS_BASE = [42, 123, 456, 789, 2026, 3141, 1618, 2718, 7777, 12345]
SEEDS_SPEC = [42, 123, 456, 789, 2026]
K = 70
TURBINE_RATED_MW = 3.465  # SG 3.4-132

CONFIG = LGBMConfig(
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
        c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"
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


# ============================================================ CF target utils


def _to_cf(y_mw, active_turbines):
    denom = np.maximum(active_turbines.astype(np.float32) * TURBINE_RATED_MW, 1e-3)
    return (y_mw / denom).astype(np.float32)


def _from_cf(cf, active_turbines):
    cf = np.clip(cf, 0.0, 1.0)
    return (cf * active_turbines.astype(np.float32) * TURBINE_RATED_MW).astype(np.float32)


# ============================================================ ensemble


def _train_seed_ensemble_valid(X_tr, y_tr, X_va, feat_cols, seeds, weights=None):
    preds = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        if weights is not None:
            dtrain = lgb.Dataset(
                X_tr,
                label=y_tr,
                weight=weights,
                feature_name=feat_cols,
                free_raw_data=False,
            )
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        # Use X_va as evaluation set for early stopping — same as
        # train_v15_regime.py, on purpose (we want an honest stop round).
        dval = lgb.Dataset(X_va, label=np.zeros(len(X_va)), feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(
            cfg.to_params(),
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(250, verbose=False)],
        )
        preds.append(b.predict(X_va, num_iteration=b.best_iteration))
    return np.mean(preds, axis=0)


def _train_seed_ensemble_valid_with_y(X_tr, y_tr, X_va, y_va, feat_cols, seeds, weights=None):
    """Train ensemble with a real early-stopping target on the validation fold.

    Used when ``y_va`` is available (the Fold-5 labels) — matches the
    honest evaluation pattern from train_v15_regime.py.
    """
    preds = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        if weights is not None:
            dtrain = lgb.Dataset(
                X_tr,
                label=y_tr,
                weight=weights,
                feature_name=feat_cols,
                free_raw_data=False,
            )
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(
            cfg.to_params(),
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(250, verbose=False)],
        )
        preds.append(b.predict(X_va, num_iteration=b.best_iteration))
    return np.mean(preds, axis=0)


# ============================================================ main ablation


def _load_train_raw(path: str | Path) -> pd.DataFrame:
    """Load training CSV preserving ALL columns (bypasses GenerationSchema filter).

    ``load_train`` from ``src.data.loaders`` runs ``GenerationSchema`` with
    ``strict="filter"``, which strips every column not declared in the schema
    (i.e. all weather columns). Training scripts need the weather columns for
    feature engineering, so we read the raw CSV directly here — the same
    approach used by the working v15/v27 scripts before the schema-hardening
    spec was added.

    We still parse the timestamp and sort ascending to match ``load_train``
    semantics.
    """
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def _prepare_combined(use_kalman: bool, use_ramps: bool):
    """Return the pre-feature-selection frames for a given ablation config."""
    print(f"  preparing combined frame (kalman={use_kalman}, ramps={use_ramps})")
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

    if use_kalman:
        train_mask = (combined["_split"] == "train").to_numpy()
        combined, _ = add_kalman_bias_features(combined, train_mask=train_mask)

    if use_ramps:
        combined = add_ramp_features(combined)

    return combined


def _run_ablation(combined: pd.DataFrame, label: str):
    """Run the v15-regime recipe end-to-end on Fold-5, return predictions."""
    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    # Fold-5 train/val split.
    fold5 = default_folds()[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
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
        c
        for c in feature_columns(df_tr)
        if c not in ("_is_impossible", "_split", TARGET_COL)
    ]

    # CF target.
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_train.iloc[val_idx][TARGET_COL].to_numpy(dtype=np.float32)
    active_tr = df_tr["active_turbines"].to_numpy()
    active_va = df_train.iloc[val_idx]["active_turbines"].to_numpy()
    y_tr_cf = _to_cf(y_tr_mw, active_tr)
    y_va_cf = _to_cf(y_va_mw, active_va)

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    # Sample weight: zero on impossible rows, 1.0 elsewhere.
    sample_weight = (~fold_train["_is_impossible"].to_numpy()).astype(np.float32)

    # Probe for top-K.
    probe_cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain_probe = lgb.Dataset(
        X_tr_all,
        label=y_tr_cf,
        weight=sample_weight,
        feature_name=feat_cols_all,
        free_raw_data=False,
    )
    dval_probe = lgb.Dataset(
        X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False
    )
    probe = lgb.train(
        probe_cfg.to_params(),
        dtrain_probe,
        num_boost_round=5000,
        valid_sets=[dval_probe],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp, strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    print(f"  [{label}] top-K features picked, sample of new ones in top-K:")
    for name in ("ws120_kalman", "ws120_bias_kalman_smooth", "dv_3h", "dv_6h", "is_sub_cutin_soft"):
        if name in top_k:
            rank = top_k.index(name) + 1
            print(f"    - {name}: rank {rank}")

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    # Base ensemble — CF target.
    preds_base_cf = _train_seed_ensemble_valid_with_y(
        X_tr, y_tr_cf, X_va, y_va_cf, top_k, SEEDS_BASE, weights=sample_weight
    )
    # Specialists.
    regime_preds_cf = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight).astype(np.float32)
        regime_preds_cf[name] = _train_seed_ensemble_valid_with_y(
            X_tr, y_tr_cf, X_va, y_va_cf, top_k, SEEDS_SPEC, weights=weights
        )
    preds_avg3_cf = np.mean(list(regime_preds_cf.values()), axis=0)

    # v15 50/50 blend = base + avg3. Same rule as train_v15_regime.py.
    preds_blend_cf = 0.5 * preds_base_cf + 0.5 * preds_avg3_cf
    preds_mw = _from_cf(preds_blend_cf, active_va)
    preds_mw = np.clip(preds_mw, 0.0, CAPACITY_MW)

    # Timestamps of val rows (needed for per-month calibration).
    ts_val = df_train.iloc[val_idx][TIMESTAMP_COL].to_numpy()
    months_val = pd.to_datetime(ts_val).month.to_numpy()

    return {
        "label": label,
        "preds_mw": preds_mw,
        "y_true_mw": y_va_mw,
        "months_val": months_val,
        "n_features": len(top_k),
    }


def _load_oof_for_calibration() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load folds 1–4 OOF predictions from the tuned baseline.

    We deliberately reuse the *existing* OOF cache rather than rerunning
    folds 1–4 inside this script: folds 1–4 OOF is generated by the
    standard training pipeline and re-running it here would dominate the
    ablation runtime without changing the point of the exercise (which is
    to check whether per-month affine on OOF helps Fold-5).
    """
    if not OOF_TUNED_PATH.is_file():
        raise FileNotFoundError(
            f"{OOF_TUNED_PATH} not present. Per-month calibration needs OOF "
            "predictions from folds 1–4. Run the tuned baseline training "
            "first to produce this file."
        )
    oof = pd.read_parquet(OOF_TUNED_PATH)
    oof = oof[oof["fold"] != "fold5_2025Q1"].copy()
    y_pred = oof["y_pred"].to_numpy(dtype=float)
    y_true = oof["y_true"].to_numpy(dtype=float)
    months = pd.to_datetime(oof[TIMESTAMP_COL]).dt.month.to_numpy()
    return y_pred, y_true, months


def main() -> None:
    set_global_seed(42)
    start = time.time()
    print("Loading OOF (folds 1-4) for per-month calibration fit...")
    try:
        oof_pred, oof_true, oof_months = _load_oof_for_calibration()
        affine = PerMonthAffine().fit(oof_pred, oof_true, oof_months)
        print(f"  affine params per month: {sorted(affine.params.items())}")
        have_affine = bool(affine.params)
    except FileNotFoundError as exc:
        print(f"  {exc}")
        print("  → skipping per-month calibration column in the report table.")
        affine = None
        have_affine = False

    # Prepare combined frames once per feature set (4 in total, but we can
    # cache the base and kalman/ramps additions by reusing `baseline` whenever
    # possible). Keeping it simple: build each fresh. The dominant cost is
    # the LGBM training, not feature construction.
    configs = [
        ("baseline", False, False),
        ("kalman", True, False),
        ("ramps", False, True),
        ("full", True, True),
    ]
    results = []
    for label, use_k, use_r in configs:
        print(f"\n=== {label} ===")
        combined = _prepare_combined(use_kalman=use_k, use_ramps=use_r)
        res = _run_ablation(combined, label=label)
        nmae_raw = normalized_mae(res["y_true_mw"], res["preds_mw"])
        print(f"  [{label}] Fold-5 nMAE (raw)         : {nmae_raw:.4f}%")

        if have_affine and affine is not None:
            preds_cal = affine.predict(res["preds_mw"], res["months_val"])
            preds_cal = np.clip(preds_cal, 0.0, CAPACITY_MW)
            nmae_cal = normalized_mae(res["y_true_mw"], preds_cal)
            print(f"  [{label}] Fold-5 nMAE (per-month) : {nmae_cal:.4f}%")
        else:
            nmae_cal = None

        results.append(
            {
                "label": label,
                "n_features": res["n_features"],
                "nmae_raw": nmae_raw,
                "nmae_affine": nmae_cal,
            }
        )

    # ---------------- print decision table
    print("\n==============================================================")
    print("  config                 nmae_raw   nmae_affine")
    for row in results:
        cal = "  n/a" if row["nmae_affine"] is None else f"{row['nmae_affine']:.4f}%"
        print(f"  {row['label']:20s}  {row['nmae_raw']:.4f}%   {cal}")
    print("==============================================================")

    # Baseline reference.
    base = next(r for r in results if r["label"] == "baseline")
    print("\nΔ vs baseline (positive = better):")
    for row in results:
        delta_raw = base["nmae_raw"] - row["nmae_raw"]
        delta_cal = (
            None
            if row["nmae_affine"] is None or base["nmae_affine"] is None
            else (base["nmae_affine"] - row["nmae_affine"])
        )
        cal = "  n/a" if delta_cal is None else f"{delta_cal:+.4f}pp"
        print(f"  {row['label']:20s}  {delta_raw:+.4f}pp   {cal}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved: {REPORT_PATH}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
