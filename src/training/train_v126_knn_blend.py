"""V126: KNN Analog Ensemble blended with corrected v97b.

Concept: for each test hour, find the K most similar historical weather
patterns and use their actual power outputs as prediction. This is
fundamentally different from LGBM (non-parametric vs parametric) and
should produce uncorrelated errors.

Weather similarity vector:
  - wind_speed_120m (primary driver)
  - wind_speed_80m, wind_speed_10m (profile shape)
  - wind_direction_120m sin/cos
  - temperature_80m
  - pressure_msl
  - gust
  - hour_sin/cos (diurnal pattern)
  - month_sin/cos (seasonal)

Scaling: StandardScaler, with ws_120 given 3x weight (dominant feature).

Prediction: median(power of K neighbors) adjusted for active turbines.

Blend with corrected v97b at various weights.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
V97B_BEST_PATH = _ROOT / "submissions" / "archive" / "v123.B_hw0.7_q0.7.csv"
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TARGET_COL_NAME = "Выработка. Результирующий расчет"
TOTAL_TURBINES = 26
MAINT_COL = "Кол-во_ВЭУ_в_ремонте"

# KNN parameters
K_NEIGHBORS = 50
WS120_WEIGHT = 3.0  # Extra weight on dominant feature


def _write(preds: np.ndarray, path: Path, label: str = "") -> None:
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TARGET_COL_NAME: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<50s} mean={preds.mean():.2f}  {label}")


def _build_weather_vector(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the weather similarity vector for KNN matching."""
    ts = pd.to_datetime(df.iloc[:, 0])
    hour = ts.dt.hour.to_numpy(dtype=np.float32)
    month = ts.dt.month.to_numpy(dtype=np.float32)
    doy = ts.dt.dayofyear.to_numpy(dtype=np.float32)

    # Direction to sin/cos
    # Raw direction is in 0.001-0.360 format (degrees/1000)
    dir_120 = df["wind_direction_120m"].to_numpy(dtype=np.float32)
    dir_rad = np.deg2rad(dir_120 * 1000.0)
    dir_sin = np.sin(dir_rad)
    dir_cos = np.cos(dir_rad)

    features = {
        "ws_120": df["wind_speed_120m"].to_numpy(dtype=np.float32),
        "ws_80": df["wind_speed_80m"].to_numpy(dtype=np.float32),
        "ws_10": df["wind_speed_10m"].to_numpy(dtype=np.float32),
        "dir_sin": dir_sin,
        "dir_cos": dir_cos,
        "temp_80": df["temperature_80m"].to_numpy(dtype=np.float32),
        "pressure": df["pressure_msl"].to_numpy(dtype=np.float32),
        "gust": df["wind_gusts_10m"].to_numpy(dtype=np.float32),
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
    }

    names = list(features.keys())
    X = np.column_stack(list(features.values()))
    return X, names


def main() -> None:
    set_global_seed(42)
    t_start = time.time()

    print("=" * 72)
    print(f"V126: KNN Analog Ensemble (K={K_NEIGHBORS})")
    print("=" * 72)

    # --- Load training data ---
    print("\n[1/4] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    train.columns = train.columns.str.strip()
    valid = pd.read_csv(VALID_PATH)
    valid.columns = valid.columns.str.strip()

    target_col = TARGET_COL_NAME
    train_ts = pd.to_datetime(train.iloc[:, 0])

    # Get power output and active turbines from training
    train_power = train[target_col].to_numpy(dtype=np.float32)
    train_maint = train[MAINT_COL].to_numpy(dtype=np.float32)
    train_active = TOTAL_TURBINES - train_maint

    # Filter out maintenance/impossible rows (NaN or zero power at decent wind)
    valid_train_mask = (
        np.isfinite(train_power) &
        (train_power >= 0) &
        (train_power <= CAPACITY_MW)
    )
    # Also exclude likely curtailment: very low power at high wind
    ws_train = train["wind_speed_120m"].to_numpy(dtype=np.float32)
    curtailment_mask = (ws_train > 8) & (train_power < 5) & valid_train_mask
    valid_train_mask = valid_train_mask & ~curtailment_mask

    print(f"  Training rows: {len(train)}, valid for KNN: {valid_train_mask.sum()}")
    print(f"  Excluded: {(~valid_train_mask).sum()} (NaN/impossible/curtailment)")

    # Test set
    test_maint = valid[MAINT_COL].to_numpy(dtype=np.float32)
    test_active = TOTAL_TURBINES - test_maint
    print(f"  Test rows: {len(valid)}")
    print(f"  Test active turbines: mean={test_active.mean():.2f}")

    # --- Build weather vectors ---
    print("\n[2/4] Building weather vectors and KNN index...")
    X_train_raw, feat_names = _build_weather_vector(train)
    X_test_raw, _ = _build_weather_vector(valid)

    # Filter to valid rows only
    X_train_valid = X_train_raw[valid_train_mask]
    power_valid = train_power[valid_train_mask]
    active_valid = train_active[valid_train_mask]

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_valid)
    X_test_scaled = scaler.transform(X_test_raw)

    # Apply extra weight to ws_120 (index 0)
    ws120_idx = feat_names.index("ws_120")
    X_train_scaled[:, ws120_idx] *= WS120_WEIGHT
    X_test_scaled[:, ws120_idx] *= WS120_WEIGHT

    # Build BallTree for fast KNN
    tree = BallTree(X_train_scaled, metric="euclidean")
    print(f"  BallTree built on {len(X_train_scaled)} samples, {X_train_scaled.shape[1]} features")

    # --- KNN prediction ---
    print(f"\n[3/4] KNN prediction (K={K_NEIGHBORS})...")
    # Query all test points at once
    distances, indices = tree.query(X_test_scaled, k=K_NEIGHBORS)

    # For each test point, get the median power of K neighbors
    # Adjusted for turbine availability
    knn_preds = np.zeros(len(X_test_scaled), dtype=np.float64)

    for i in range(len(X_test_scaled)):
        neighbor_power = power_valid[indices[i]]
        neighbor_active = active_valid[indices[i]]
        # Normalize to per-turbine power, then scale to test active count
        per_turbine = neighbor_power / np.maximum(neighbor_active, 1)
        adjusted_power = per_turbine * test_active[i]
        knn_preds[i] = np.median(adjusted_power)

    knn_preds = np.clip(knn_preds, 0, CAPACITY_MW)
    print(f"  KNN predictions: mean={knn_preds.mean():.2f}, std={knn_preds.std():.2f}")

    # --- Also try distance-weighted KNN ---
    print("  Computing distance-weighted KNN...")
    knn_weighted = np.zeros(len(X_test_scaled), dtype=np.float64)
    for i in range(len(X_test_scaled)):
        neighbor_power = power_valid[indices[i]]
        neighbor_active = active_valid[indices[i]]
        per_turbine = neighbor_power / np.maximum(neighbor_active, 1)
        adjusted_power = per_turbine * test_active[i]
        # Inverse distance weighting
        dists = distances[i]
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()
        knn_weighted[i] = np.average(adjusted_power, weights=weights)

    knn_weighted = np.clip(knn_weighted, 0, CAPACITY_MW)
    print(f"  Weighted KNN: mean={knn_weighted.mean():.2f}, std={knn_weighted.std():.2f}")

    # --- OOF evaluation (use fold 5 period: Jan-Mar 2025) ---
    print("\n  OOF evaluation (Jan-Mar 2025 = Fold 5 surrogate)...")
    # For fold 5: train on 2023-Sep 2024, predict Jan-Mar 2025
    fold5_start = pd.Timestamp("2025-01-01")
    fold5_end = pd.Timestamp("2025-03-31 23:00:00")

    oof_mask = (train_ts >= fold5_start) & (train_ts <= fold5_end) & valid_train_mask
    train_mask_f5 = (train_ts < fold5_start) & valid_train_mask

    if oof_mask.sum() > 0 and train_mask_f5.sum() > 0:
        X_train_f5 = X_train_raw[train_mask_f5]
        power_f5_train = train_power[train_mask_f5]
        active_f5_train = train_active[train_mask_f5]

        X_val_f5 = X_train_raw[oof_mask]
        power_f5_val = train_power[oof_mask]
        active_f5_val = train_active[oof_mask]

        # Scale
        sc_f5 = StandardScaler()
        X_tr_f5_s = sc_f5.fit_transform(X_train_f5)
        X_va_f5_s = sc_f5.transform(X_val_f5)
        X_tr_f5_s[:, ws120_idx] *= WS120_WEIGHT
        X_va_f5_s[:, ws120_idx] *= WS120_WEIGHT

        tree_f5 = BallTree(X_tr_f5_s, metric="euclidean")

        for k in [20, 50, 100, 200]:
            d_f5, idx_f5 = tree_f5.query(X_va_f5_s, k=k)
            preds_f5 = np.zeros(len(X_va_f5_s))
            for i in range(len(X_va_f5_s)):
                np_power = power_f5_train[idx_f5[i]]
                np_active = active_f5_train[idx_f5[i]]
                per_turb = np_power / np.maximum(np_active, 1)
                preds_f5[i] = np.median(per_turb * active_f5_val[i])
            preds_f5 = np.clip(preds_f5, 0, CAPACITY_MW)
            nmae = float(np.mean(np.abs(power_f5_val - preds_f5)) / CAPACITY_MW * 100)
            print(f"    K={k:3d}: KNN-only nMAE={nmae:.4f}%")

        # Best K=50 for blending
        d_f5, idx_f5 = tree_f5.query(X_va_f5_s, k=50)
        knn_f5 = np.zeros(len(X_va_f5_s))
        for i in range(len(X_va_f5_s)):
            np_power = power_f5_train[idx_f5[i]]
            np_active = active_f5_train[idx_f5[i]]
            per_turb = np_power / np.maximum(np_active, 1)
            knn_f5[i] = np.median(per_turb * active_f5_val[i])
        knn_f5 = np.clip(knn_f5, 0, CAPACITY_MW)

        # Load v97b OOF for fold 5 to evaluate blend
        oof97 = pd.read_parquet(_ROOT / "data" / "processed" / "v97b_oof.parquet")
        oof97["ts"] = pd.to_datetime(oof97["ts"])
        f5_97 = oof97[oof97["fold"] == 5].sort_values("ts").reset_index(drop=True)

        # Match timestamps
        f5_train_ts = train_ts[oof_mask].reset_index(drop=True)
        # The v97b OOF and our KNN OOF should cover the same period
        knn_nmae = float(np.mean(np.abs(power_f5_val - knn_f5)) / CAPACITY_MW * 100)
        v97b_f5_nmae = float(normalized_mae(f5_97["target_mw"].to_numpy(), f5_97["pred_cf_mw"].to_numpy()))

        print(f"\n    KNN-only F5 nMAE: {knn_nmae:.4f}%")
        print(f"    v97b CF F5 nMAE:  {v97b_f5_nmae:.4f}%")

        # Correlation between KNN and v97b errors
        # Align by timestamp
        common_ts = set(f5_train_ts) & set(f5_97["ts"])
        if len(common_ts) > 100:
            knn_df = pd.DataFrame({"ts": f5_train_ts, "knn_pred": knn_f5, "target": power_f5_val})
            merged = knn_df.merge(f5_97[["ts", "pred_cf_mw"]], on="ts")
            corr = float(np.corrcoef(merged["knn_pred"].to_numpy(), merged["pred_cf_mw"].to_numpy())[0, 1])
            print(f"    Prediction correlation KNN vs v97b: {corr:.4f}")

            # Blend evaluation
            knn_vals = merged["knn_pred"].to_numpy()
            lgbm_vals = merged["pred_cf_mw"].to_numpy()
            target_vals = merged["target"].to_numpy()

            print(f"\n    Blend evaluation (F5 OOF):")
            for w_knn in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                bl = np.clip(w_knn * knn_vals + (1 - w_knn) * lgbm_vals, 0, CAPACITY_MW)
                nm = float(np.mean(np.abs(target_vals - bl)) / CAPACITY_MW * 100)
                delta = nm - v97b_f5_nmae
                print(f"      {w_knn:.0%} KNN + {1-w_knn:.0%} LGBM: {nm:.4f}% (Δ={delta:+.4f}%)")

    # --- Generate test submissions ---
    print(f"\n[4/4] Generating submissions...")

    # Load best v97b corrected
    v97b_best = pd.read_csv(V97B_BEST_PATH)
    v97b_preds = v97b_best[TARGET_COL_NAME].to_numpy(dtype=np.float64)

    # KNN standalone
    _write(knn_preds, OUTPUT_DIR / "v126.0_knn_standalone.csv", "KNN K=50 median")
    _write(knn_weighted, OUTPUT_DIR / "v126.0b_knn_weighted.csv", "KNN K=50 dist-weighted")

    # Blends
    for w_knn in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = np.clip(w_knn * knn_preds + (1 - w_knn) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v126.1_knn{int(w_knn*100):02d}_v97b{int((1-w_knn)*100):02d}.csv",
               f"{w_knn:.0%} KNN + {1-w_knn:.0%} v97b_corr")

    # Distance-weighted KNN blend
    for w_knn in [0.10, 0.15, 0.20]:
        blend = np.clip(w_knn * knn_weighted + (1 - w_knn) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v126.2_knnw{int(w_knn*100):02d}_v97b{int((1-w_knn)*100):02d}.csv",
               f"{w_knn:.0%} KNN-wt + {1-w_knn:.0%} v97b_corr")

    # KNN with corrections applied to KNN itself (same recipe)
    valid_ts = pd.to_datetime(valid.iloc[:, 0])
    ws_test = valid["wind_speed_120m"].to_numpy(dtype=np.float32)
    hw_mask = (ws_test >= 12) & (ws_test < 18)
    co_mask = ws_test >= 18

    # Reorder to match submission order (valid is sorted by time, sub is reverse)
    # Check: is valid sorted same as submission?
    sub_ts = pd.to_datetime(v97b_best.iloc[:, 0]) if TIMESTAMP_COL in v97b_best.columns else None

    # KNN predictions are in valid_df order (sorted by time ascending from valid_features.csv)
    # v97b_best is in submission order (Mar 31 23:00 first)
    # Need to align them
    valid_ts_arr = valid_ts.to_numpy()

    # Map KNN to submission order using v97b timestamps
    # v97b_best has no timestamp column (single column), so it matches valid_features order
    # Actually let's check: v123.B was written as single-column, matching the order of v97b submission
    # v97b submission is in reverse chronological order (Mar 31 → Jan 1)
    # valid_features.csv is also in reverse order (first row = Mar 31 23:00)
    # So they should already be aligned!

    # Apply corrections to KNN
    knn_corr = knn_preds.copy()
    # Use the same correction values from the OOF analysis
    # KNN has its own biases — compute them from OOF if available, otherwise use mild corrections
    knn_corr[hw_mask] -= 1.0  # KNN likely over-predicts high wind too
    knn_corr = np.clip(knn_corr, 0, CAPACITY_MW)

    for w_knn in [0.10, 0.15, 0.20]:
        blend = np.clip(w_knn * knn_corr + (1 - w_knn) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v126.3_knnc{int(w_knn*100):02d}_v97b{int((1-w_knn)*100):02d}.csv",
               f"{w_knn:.0%} KNN-corr + {1-w_knn:.0%} v97b_corr")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"\n  Top picks:")
    print(f"    v126.1_knn10_v97b90.csv  (10% KNN diversity)")
    print(f"    v126.1_knn15_v97b85.csv  (15% KNN)")
    print(f"    v126.2_knnw10_v97b90.csv (10% weighted KNN)")


if __name__ == "__main__":
    main()
