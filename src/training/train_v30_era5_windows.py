"""V30: Ablation of additional ERA5 rolling windows (TODO item 4).

Current windows: 3, 6, 12, 24h.
Candidates: 2h (very-short, rapid ramps), 48h and 72h (synoptic regime).

The 6h window is the #1 feature by gain importance. Hypothesis:
- 2h captures sub-hourly ramp-up/down that 3h misses.
- 48h/72h captures multi-day wind regimes (blocking highs, frontal passages)
  that 24h only partially sees.

Four configs tested on Fold-5 with v29 HPs (trial 23, CF target, K=70):
  baseline   — current 3/6/12/24h windows
  +2h        — add 2h mean + std
  +48_72h    — add 48h and 72h mean + std
  +all       — add 2h + 48h + 72h

Stop condition: only ship windows that improve Fold-5 by ≥ 0.02 pp.

Usage:
    python -m src.training.train_v30_era5_windows
"""

from __future__ import annotations

import json
import sys
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
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.pipeline import build_features, feature_columns
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
REPORT_PATH = _ROOT / "data" / "processed" / "v30_era5_windows_ablation.json"

TURBINE_RATED_MW = 3.465
K = 70
SEEDS_SPEC = [42, 123, 456, 789, 2026]

# v29 / Optuna trial-23 HPs
CONFIG = LGBMConfig(
    num_leaves=55,
    min_data_in_leaf=43,
    learning_rate=0.0259476307435388,
    feature_fraction=0.49455788842926784,
    bagging_fraction=0.4229521099231461,
    bagging_freq=3,
    lambda_l1=1.599611931632972,
    lambda_l2=0.0003476780244211917,
    num_boost_round=5000,
    early_stopping_rounds=250,
    log_period=0,
)

# Window sets to test.  Each entry is (label, extra_windows).
# extra_windows is a list of integer hours to add on top of the base 3/6/12/24.
WINDOW_CONFIGS = [
    ("baseline",  []),
    ("+2h",       [2]),
    ("+48_72h",   [48, 72]),
    ("+all",      [2, 48, 72]),
]


# ============================================================ feature helpers


def _load_train_raw(path: str | Path) -> pd.DataFrame:
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


def _add_era5_rolling(df, extra_windows: list[int] | None = None):
    """Add ERA5 rolling features.

    Base windows: 3, 6, 12, 24h (always included).
    extra_windows: additional window sizes in hours (e.g. [2, 48, 72]).
    """
    df = df.copy()
    ws = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]

    all_windows = [3, 6, 12, 24] + (extra_windows or [])
    for w in all_windows:
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


# ============================================================ ensemble


def _run_fold5(combined, extra_windows: list[int], label: str) -> dict:
    """Run Fold-5 evaluation for a given set of extra rolling windows."""
    combined_w = _add_era5_rolling(combined, extra_windows=extra_windows)

    df_train = combined_w[combined_w["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    sw_full = (~impossible.to_numpy()).astype(np.float32)

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
        c for c in feature_columns(df_tr)
        if c not in ("_is_impossible", "_split", TARGET_COL)
    ]
    active_tr = df_tr["active_turbines"].to_numpy()
    active_va = df_va["active_turbines"].to_numpy()
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf = _to_cf(y_tr_mw, active_tr)
    y_va_cf = _to_cf(y_va_mw, active_va)
    sw_fold = sw_full[train_idx]

    # Probe for top-K.
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    probe_cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain_probe = lgb.Dataset(
        X_tr_all, label=y_tr_cf, weight=sw_fold,
        feature_name=feat_cols_all, free_raw_data=False,
    )
    dval_probe = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        probe_cfg.to_params(), dtrain_probe, num_boost_round=5000,
        valid_sets=[dval_probe], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp, strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    # Report which new windows made top-K.
    new_cols = [f"era5_ws100_roll_mean_{w}h" for w in extra_windows] + \
               [f"era5_ws100_roll_std_{w}h" for w in extra_windows]
    in_topk = [(c, top_k.index(c) + 1) for c in new_cols if c in top_k]
    if in_topk:
        for col, rank in in_topk:
            print(f"    [{label}] {col}: rank {rank}")
    else:
        print(f"    [{label}] no new window features in top-{K}")

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    # 3 regime specialists × 5 seeds.
    regime_preds = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sw_fold).astype(np.float32)
        seed_preds = []
        for s in SEEDS_SPEC:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dtrain = lgb.Dataset(
                X_tr, label=y_tr_cf, weight=weights,
                feature_name=top_k, free_raw_data=False,
            )
            dval = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dtrain, num_boost_round=5000,
                valid_sets=[dval], valid_names=["val"],
                callbacks=[lgb.early_stopping(250, verbose=False)],
            )
            seed_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds[name] = np.mean(seed_preds, axis=0)

    avg_cf = np.mean(list(regime_preds.values()), axis=0)
    avg_mw = np.clip(_from_cf(avg_cf, active_va), 0.0, CAPACITY_MW)
    nmae = float(normalized_mae(y_va_mw, avg_mw))
    print(f"  [{label}] Fold-5 nMAE: {nmae:.4f}%")
    return {"label": label, "extra_windows": extra_windows, "nmae": nmae, "n_features": len(feat_cols_all)}


# ============================================================ main


def main() -> None:
    set_global_seed(42)
    print("V30: ERA5 rolling window ablation")
    print("=" * 60)

    # Build the base combined frame once (without rolling — rolling is
    # applied per config inside _run_fold5).
    print("\nLoading and building base features (no rolling yet)...")
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
    # NOTE: rolling is NOT added here — each config adds its own windows.

    results = []
    for label, extra_windows in WINDOW_CONFIGS:
        print(f"\n--- {label} ---")
        res = _run_fold5(combined, extra_windows=extra_windows, label=label)
        results.append(res)

    # Decision table.
    baseline_nmae = next(r["nmae"] for r in results if r["label"] == "baseline")
    print("\n" + "=" * 60)
    print(f"  {'config':15s}  {'nMAE':>8s}  {'Δ vs baseline':>14s}  {'ship?':>6s}")
    print("  " + "-" * 50)
    for r in results:
        delta = baseline_nmae - r["nmae"]
        ship = "YES" if delta >= 0.02 else ("~" if delta >= 0.005 else "no")
        print(f"  {r['label']:15s}  {r['nmae']:8.4f}%  {delta:+14.4f}pp  {ship:>6s}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
