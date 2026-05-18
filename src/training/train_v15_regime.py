"""V15: v14 + regime-specialist ensemble.

Combines:
- ERA5 rolling features (v14)
- Datasheet power curve (v6+)
- Wake correction (v8+)
- Wind vector extras (v7+)
- Feature selection K=70
- Tuned params (num_leaves=86, lr=0.00844)
- 10-seed base ensemble + 3 regime specialists (low/mid/high wind)
- Multi-level blend

Usage:
    python -m src.training.train_v15_regime
    python -m src.training.train_v15_regime --residual-clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.outliers import add_nwp_era5_disagreement_flag, compute_training_weights, identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v15.0_regime.csv"

SEEDS_BASE = [42, 123, 456, 789, 2026, 3141, 1618, 2718, 7777, 12345]  # 10 seeds
SEEDS_SPEC = [42, 123, 456, 789, 2026]  # 5 seeds per specialist
K = 70

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)


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


def merge_era5(df, era5):
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
    era5_new = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def add_era5_rolling(df):
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
    dir_sin = df["era5_dir100_sin"]
    dir_cos = df["era5_dir100_cos"]
    df["era5_dir_sin_diff1"] = dir_sin.diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = dir_cos.diff(1).fillna(0)
    return df


def train_seed_ensemble_valid(X_tr, y_tr, X_va, y_va, feat_cols, seeds, weights=None):
    """Train seed ensemble, return array of predictions on X_va + best_iters."""
    preds = []
    best_iters = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        if weights is not None:
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        p = np.clip(b.predict(X_va, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        preds.append(p)
        best_iters.append(b.best_iteration)
    return np.mean(preds, axis=0), best_iters


def train_seed_ensemble_full(X, y, feat_cols, seeds, n_rounds, X_test, weights=None):
    """Train on full data (no early stopping), predict on X_test."""
    preds = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        if weights is not None:
            dtrain = lgb.Dataset(X, label=y, weight=weights, feature_name=feat_cols, free_raw_data=False)
        else:
            dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=n_rounds)
        p = np.clip(b.predict(X_test), 0, CAPACITY_MW)
        preds.append(p)
    return np.mean(preds, axis=0)


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser used by this entrypoint.

    Exposed as a top-level helper so CLI tests can probe the parsed
    namespace without invoking ``main`` (see Task 7.2 verification step 2).
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.training.train_v15_regime",
        description="V15 regime-specialist ensemble training.",
    )
    parser.add_argument(
        "--residual-clean",
        action="store_true",
        dest="residual_clean",
        help=(
            "Flag label-noise candidates via OOF residuals and zero their "
            "sample weight (opt-in; requirement 6.3)."
        ),
    )
    return parser


def _resolve_oof_path() -> Path:
    """Resolve the OOF parquet used by the residual cleaner.

    Prefers ``data/processed/oof_*tuned*.parquet`` (the tuned baseline),
    falling back to ``data/processed/oof_lgbm.parquet``. Raises
    ``FileNotFoundError`` when neither exists so the operator knows to
    run the baseline training first to produce OOFs.
    """
    processed_dir = _ROOT / "data" / "processed"
    tuned_candidates = sorted(processed_dir.glob("oof_*tuned*.parquet"))
    if tuned_candidates:
        return tuned_candidates[0]
    plain = processed_dir / "oof_lgbm.parquet"
    if plain.is_file():
        return plain
    raise FileNotFoundError(
        "residual-clean requires an OOF parquet under data/processed/ "
        "(looked for oof_*tuned*.parquet then oof_lgbm.parquet). "
        "Run the baseline training first to produce OOFs before passing "
        "--residual-clean."
    )


def _residual_clean_mask(df_train: pd.DataFrame, fold_name: str = "all") -> np.ndarray:
    """Return a ``float32`` multiplier aligned to ``df_train`` rows.

    ``1.0`` on rows the residual cleaner keeps, ``0.0`` on flagged rows.
    The ``flag_residuals`` import is scoped to this helper so the default
    (non-opt-in) code path never pays its import cost.
    """
    from src.data.residual_cleaner import flag_residuals  # lazy: opt-in only

    oof_path = _resolve_oof_path()
    flagged = flag_residuals(
        df_train,
        oof_path=oof_path,
        k=3.0,
        ws_bin_width=1.0,
        fold_name=fold_name,
    )
    return (~flagged.to_numpy()).astype(np.float32)


def main(*, residual_clean: bool = False):
    set_global_seed(42)
    print("Preparing data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    # NWP/ERA5 disagreement flag — earliest point where both wind_speed_120m
    # and era5_wind_speed_100m are present on the same frame (Requirement 2.7).
    combined = add_nwp_era5_disagreement_flag(combined)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    # Explicit training-weight policy (Requirements 1.5, 9.3, 9.6):
    # enforce the library default at the call site so a future change to
    # compute_training_weights' default cannot silently alter training
    # behaviour. downweight_2022=1.0 is inert — impossible rows get weight
    # 0.0, all other rows get 1.0. See src/data/outliers.py and
    # PROJECT.md §10.
    sample_weight_full = compute_training_weights(df_train, downweight_2022=1.0)

    # Residual-based second-pass cleaner (Requirements 6.3, 6.7; Task 7.2).
    # When --residual-clean is absent we do NOT invoke flag_residuals and
    # do NOT mutate the weights. When present, flagged rows' weights are
    # multiplied by 0.0 (NOT dropped — lag-history semantics are preserved
    # identically to the is_maintenance pattern). The mask is aligned to
    # df_train rows; downstream sample_weight_fold = sample_weight_full[
    # train_idx] then picks up the cleaned weights automatically.
    if residual_clean:
        keep_mask = _residual_clean_mask(df_train, fold_name="fold5")
        sample_weight_full = (sample_weight_full * keep_mask).astype(np.float32)
        n_zeroed = int((keep_mask == 0.0).sum())
        print(f"  residual-clean: zeroed sample_weight on {n_zeroed} flagged rows")

    folds = default_folds()
    fold5 = folds[-1]
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
    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    # Restrict the global training-weight array to the fold's train rows.
    sample_weight_fold = sample_weight_full[train_idx].astype(np.float32)

    # Probe for top-K.
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr, weight=sample_weight_fold,
                         feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()
    ws_va = df_va["wind_speed_120m"].to_numpy()

    # === Fold-5 evaluation ===
    print("\n=== Fold-5 evaluation ===")
    # Base model (10 seeds, global sample_weight from compute_training_weights).
    print("Training base (10 seeds)...")
    preds_base, iters_base = train_seed_ensemble_valid(
        X_tr, y_tr, X_va, y_va, top_k, SEEDS_BASE, weights=sample_weight_fold
    )
    nmae_base = normalized_mae(y_va, preds_base)
    print(f"  Base: {nmae_base:.4f}%")

    # 3 regime specialists (5 seeds each). Multiply regime weights by the
    # global sample_weight so impossible rows still receive weight 0.0.
    regime_preds = {}
    regime_iters = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight_fold).astype(np.float32)
        print(f"Training specialist {name}...")
        preds, iters = train_seed_ensemble_valid(X_tr, y_tr, X_va, y_va, top_k, SEEDS_SPEC, weights)
        regime_preds[name] = preds
        regime_iters[name] = iters
        print(f"  {name}: {normalized_mae(y_va, preds):.4f}%")

    # Blend strategies.
    preds_avg3 = np.mean(list(regime_preds.values()), axis=0)
    preds_base_avg3 = 0.5 * preds_base + 0.5 * preds_avg3
    preds_base_avg3_60 = 0.6 * preds_base + 0.4 * preds_avg3

    print("\n=== Fold-5 blend comparison ===")
    print(f"  Base (10 seeds):           {nmae_base:.4f}%")
    print(f"  Avg 3 specialists:         {normalized_mae(y_va, preds_avg3):.4f}%")
    print(f"  Base + avg3 (50/50):       {normalized_mae(y_va, preds_base_avg3):.4f}%")
    print(f"  Base + avg3 (60/40):       {normalized_mae(y_va, preds_base_avg3_60):.4f}%")

    # === Full-fit ===
    print("\n=== Full-fit ===")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    wake_full = fit_wake_lookup(df_train_clean, n_sectors=16)
    df_train_full = _add_pc(df_train, pc_sector_full, pc_global_full)
    df_train_full = add_wake_features(df_train_full, wake_full)
    X_full = df_train_full[top_k].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)
    ws_full = df_train_full["wind_speed_120m"].to_numpy()

    df_vp = _add_pc(df_valid_sorted, pc_sector_full, pc_global_full)
    df_vp = add_wake_features(df_vp, wake_full)
    for c in set(top_k) - set(df_vp.columns):
        df_vp[c] = 0.0
    X_valid = df_vp[top_k].to_numpy(dtype=np.float32)

    n_rounds_base = max(int(np.median(iters_base) * 1.2), 2000)
    print(f"  Base: training {len(SEEDS_BASE)} seeds, {n_rounds_base} rounds each")
    preds_valid_base = train_seed_ensemble_full(
        X_full, y_full, top_k, SEEDS_BASE, n_rounds_base, X_valid,
        weights=sample_weight_full,
    )

    # Specialists.
    specialist_valid_preds = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_full >= lo) & (ws_full < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight_full).astype(np.float32)
        n_rounds_spec = max(int(np.median(regime_iters[name]) * 1.2), 2000)
        print(f"  {name}: {len(SEEDS_SPEC)} seeds, {n_rounds_spec} rounds")
        p = train_seed_ensemble_full(X_full, y_full, top_k, SEEDS_SPEC, n_rounds_spec, X_valid, weights)
        specialist_valid_preds[name] = p

    preds_valid_avg3 = np.mean(list(specialist_valid_preds.values()), axis=0)
    preds_valid_final = 0.5 * preds_valid_base + 0.5 * preds_valid_avg3
    preds_valid_final = np.clip(preds_valid_final, 0, CAPACITY_MW)

    order = df_vp["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_valid_final)
    po[order] = preds_valid_final
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))
    print(f"\n  Submission saved: {SUBMISSION_PATH}")
    print(f"  Mean: {preds_valid_final.mean():.2f}")

    # Also save the pure-base (v14 equivalent) and pure-avg3 for blending.
    preds_valid_base_only = np.clip(preds_valid_base, 0, CAPACITY_MW)
    po_base = np.empty_like(preds_valid_base_only)
    po_base[order] = preds_valid_base_only
    write_submission(po_base, _ROOT / "submissions" / "archive" / "v15.1_base_only.csv", expected_rows=len(df_valid))

    preds_valid_avg3_clipped = np.clip(preds_valid_avg3, 0, CAPACITY_MW)
    po_avg3 = np.empty_like(preds_valid_avg3_clipped)
    po_avg3[order] = preds_valid_avg3_clipped
    write_submission(po_avg3, _ROOT / "submissions" / "archive" / "v15.2_specialists.csv", expected_rows=len(df_valid))
    print("Done.")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(residual_clean=args.residual_clean)
