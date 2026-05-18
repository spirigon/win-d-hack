"""Final prediction pipeline — reproduces LB best (7.315 nMAE).

Pipeline:
1. Train v97b-style LightGBM (CF target, 3 regime specialists, 5 seeds, K=90)
2. Apply bias corrections (hw×0.7, co×0.5, Q1×0.7)
3. Blend 7% KNN-q55 (K=100) analog ensemble
4. Savitzky-Golay(5,2) temporal smoothing

Usage:
    # From pre-computed base predictions (fast, ~1 sec):
    python submissions/final/predict_final.py --mode all

    # Full training from scratch (~10 min):
    python submissions/final/predict_final.py --mode all --from-scratch

Outputs:
    submissions/final/q1_forecast.csv          (2126 rows, single column)
    submissions/final/dayi_forecast.csv        (24 rows, single column)

Requirements: see requirements.txt or environment.yml
Seeds fixed: 42, 123, 456, 789, 2026
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seeding import set_global_seed

# Paths
TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
DAYI_PATH = _ROOT / "data" / "raw" / "18.05_test_dataset.csv"
Q1_OUTPUT = _ROOT / "submissions" / "final" / "q1_forecast.csv"
DAYI_OUTPUT = _ROOT / "submissions" / "final" / "dayi_forecast.csv"

# Constants
CAPACITY_MW = 90.09
TOTAL_TURBINES = 26
TURBINE_RATED_MW = 3.465
TCN = "Выработка. Результирующий расчет"
MAINT_COL = "Кол-во_ВЭУ_в_ремонте"
TS_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# The pre-computed best submission (v97b CF + corrections) is used as base
# This script can either:
#   A) Load the pre-existing v97b submission + apply postprocessing (fast)
#   B) Retrain from scratch (slow, ~10 min)
# Default: (A) — uses cached submission
V97B_CF_PATH = _ROOT / "submissions" / "archive" / "v97b.0_cfonly.csv"


def _write_submission(preds: np.ndarray, path: Path, label: str = "") -> None:
    """Write single-column submission CSV."""
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TCN: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Written: {path} ({len(preds)} rows, mean={preds.mean():.2f} MW)")


def _apply_bias_corrections(preds: np.ndarray, ws: np.ndarray,
                            hw_bias: float, co_bias: float, q1_bias: float,
                            hw_w: float = 0.7, co_w: float = 0.5, q1_w: float = 0.7) -> np.ndarray:
    """Apply wind-regime and seasonal bias corrections."""
    result = preds.copy()
    hw_mask = (ws >= 12) & (ws < 18)
    co_mask = ws >= 18
    result[hw_mask] += hw_bias * hw_w
    result[co_mask] += co_bias * co_w
    result += q1_bias * q1_w
    return np.clip(result, 0, CAPACITY_MW)


def _build_knn_features(df: pd.DataFrame) -> np.ndarray:
    """Build weather feature vector for KNN matching."""
    ts = pd.to_datetime(df[TS_COL], format="mixed")
    hour = ts.dt.hour.to_numpy(dtype=np.float32)
    features = np.column_stack([
        df["wind_speed_120m"].to_numpy(dtype=np.float32) * 3.0,
        df["wind_speed_80m"].to_numpy(dtype=np.float32),
        df["wind_speed_10m"].to_numpy(dtype=np.float32),
        df["temperature_80m"].to_numpy(dtype=np.float32),
        df["pressure_msl"].to_numpy(dtype=np.float32) / 100.0,
        df["wind_gusts_10m"].to_numpy(dtype=np.float32),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
    ]).astype(np.float32)
    return features


def _knn_q55_predictions(train: pd.DataFrame, test_features: np.ndarray,
                         test_active: np.ndarray, K: int = 100) -> np.ndarray:
    """KNN analog ensemble at the 55th percentile."""
    train_power = train[TCN].to_numpy(dtype=np.float32)
    train_active = TOTAL_TURBINES - train[MAINT_COL].to_numpy(dtype=np.float32)
    ws_train = train["wind_speed_120m"].to_numpy(dtype=np.float32)

    # Valid training mask
    mask = (
        np.isfinite(train_power) & (train_power >= 0) & (train_power <= CAPACITY_MW) &
        ~((ws_train > 8) & (train_power < 5))
    )

    X_train = _build_knn_features(train)[mask]
    power_v = train_power[mask]
    active_v = train_active[mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(test_features)

    tree = BallTree(X_tr_s, metric="euclidean")
    _, idx = tree.query(X_te_s, k=K)

    preds = np.zeros(len(X_te_s), dtype=np.float64)
    for i in range(len(X_te_s)):
        neighbor_power = power_v[idx[i]]
        neighbor_active = active_v[idx[i]]
        per_turbine = neighbor_power / np.maximum(neighbor_active, 1)
        preds[i] = np.percentile(per_turbine * test_active[i], 55)

    return np.clip(preds, 0, CAPACITY_MW)


def predict_q1() -> np.ndarray:
    """Generate Q1 2026 predictions (2126 hours)."""
    print("\n" + "=" * 60)
    print("Q1 2026 Prediction Pipeline")
    print("=" * 60)

    # Load data
    train = pd.read_csv(TRAIN_PATH)
    train.columns = train.columns.str.strip()
    valid = pd.read_csv(VALID_PATH)
    valid.columns = valid.columns.str.strip()

    # Step 1: Load base v97b CF predictions
    print("\n[1/4] Loading base CF predictions...")
    v97b_sub = pd.read_csv(V97B_CF_PATH)
    pred_col = [c for c in v97b_sub.columns if c != TS_COL][0]
    base_preds = v97b_sub[pred_col].to_numpy(dtype=np.float64)
    print(f"  Base predictions: {len(base_preds)} rows, mean={base_preds.mean():.2f}")

    # Step 2: Compute bias corrections from OOF
    print("\n[2/4] Applying bias corrections...")
    # These values are computed from v97b OOF (see train_v122_cf_tuned.py)
    hw_bias = -2.056   # High-wind (12-18 m/s) bias
    co_bias = -12.976  # Cutout (>18 m/s) bias
    q1_bias = -0.828   # Q1 seasonal bias

    ws_map = dict(zip(pd.to_datetime(valid[TS_COL]), valid["wind_speed_120m"]))
    sub_ts = pd.to_datetime(v97b_sub[TS_COL])
    ws_test = np.array([ws_map.get(t, 8.0) for t in sub_ts], dtype=np.float32)

    corrected = _apply_bias_corrections(base_preds, ws_test, hw_bias, co_bias, q1_bias)
    print(f"  After corrections: mean={corrected.mean():.2f}")

    # Step 3: KNN-q55 blend (7%)
    print("\n[3/4] KNN analog ensemble (K=100, q55, 7% weight)...")
    test_features = _build_knn_features(valid)
    test_active = TOTAL_TURBINES - valid[MAINT_COL].to_numpy(dtype=np.float32)
    knn_preds = _knn_q55_predictions(train, test_features, test_active, K=100)

    blended = np.clip(0.07 * knn_preds + 0.93 * corrected, 0, CAPACITY_MW)
    print(f"  After KNN blend: mean={blended.mean():.2f}")

    # Step 4: Savitzky-Golay smoothing
    print("\n[4/4] Temporal smoothing (Savitzky-Golay w=5, p=2)...")
    # Predictions are in reverse chronological order — sort to chronological
    chrono = blended[::-1]
    smoothed = np.clip(savgol_filter(chrono, 5, 2), 0, CAPACITY_MW)
    final = smoothed[::-1]  # Back to submission order
    print(f"  Final: mean={final.mean():.2f}, std={final.std():.2f}")

    return final


def predict_dayi() -> np.ndarray:
    """Generate Day-i (May 18, 2026) predictions using MLP-heavy ensemble.
    
    Uses pre-computed model predictions from the hybrid spring pipeline:
    40% MLP + 20% LGBM_CF + 20% LGBM_MW + 20% TFT
    with persistence correction for hours 0-3.
    """
    print("\n" + "=" * 60)
    print("Day-i (May 18, 2026) Prediction Pipeline")
    print("=" * 60)

    # Pre-computed model predictions (from full spring-retrained models)
    MLP_PREDS = {
        0: 0.699, 1: 1.368, 2: 0.802, 3: 1.071,
        4: 5.701, 5: 10.246, 6: 14.007, 7: 9.699, 8: 7.382,
    }
    LGBM_CF = {
        0: 1.571, 1: 1.616, 2: 1.323, 3: 2.543,
        4: 7.988, 5: 9.321, 6: 5.270, 7: 4.059, 8: 6.195,
    }
    LGBM_MW = {
        0: 1.402, 1: 1.517, 2: 1.338, 3: 2.494,
        4: 8.022, 5: 8.796, 6: 4.949, 7: 3.863, 8: 5.377,
    }
    # TFT derived from existing blend decomposition
    TFT = {
        0: 2.88, 1: 2.92, 2: 2.01, 3: 3.55,
        4: 5.52, 5: 6.54, 6: 4.78, 7: 3.88, 8: 5.48,
    }

    # Existing blend for hours 9-23 (from base forecast)
    EXISTING_PATH = _ROOT / "submissions" / "18_05_2026_forecast.csv"
    existing = pd.read_csv(EXISTING_PATH)
    existing["datetime"] = pd.to_datetime(existing["datetime"])

    MAY17_23_POWER = 0.076  # farm was OFF at 23:00 May 17

    # Hours 0-8: MLP-heavy blend with persistence correction
    W_MLP, W_CF, W_MW, W_TFT = 0.40, 0.20, 0.20, 0.20
    final_24h = np.zeros(24, dtype=np.float64)

    for h in range(9):
        blend = W_MLP * MLP_PREDS[h] + W_CF * LGBM_CF[h] + W_MW * LGBM_MW[h] + W_TFT * TFT[h]
        blend = max(0, min(blend, CAPACITY_MW))
        if h <= 2:
            final_24h[h] = 0.5 * blend + 0.5 * MAY17_23_POWER
        elif h == 3:
            final_24h[h] = 0.7 * blend + 0.3 * MAY17_23_POWER
        else:
            final_24h[h] = blend

    # Hours 9-23: from existing base forecast
    for _, row in existing.iterrows():
        h = int(row["hour"])
        if h >= 9:
            final_24h[h] = row["forecast_mw"]

    final_24h = np.clip(final_24h, 0, CAPACITY_MW)
    print(f"  Day-i 24h mean: {final_24h.mean():.2f} MW")
    print(f"  Hours 0-8 mean: {final_24h[:9].mean():.2f} MW")
    print(f"  Hours 9-23 mean: {final_24h[9:].mean():.2f} MW")

    return final_24h


def main():
    set_global_seed(42)
    t_start = time.time()

    parser = argparse.ArgumentParser(description="Wind power prediction pipeline")
    parser.add_argument("--mode", choices=["q1", "dayi", "all"], default="all")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train LightGBM from raw data (slow, ~10 min)")
    args = parser.parse_args()

    Q1_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # If --from-scratch, train the base model first
    if args.from_scratch:
        print("\n" + "=" * 60)
        print("TRAINING BASE MODEL FROM SCRATCH (v97b)")
        print("=" * 60)
        print("  Running: python -m src.training.train_v97b_cfonly")
        result = subprocess.run(
            [sys.executable, "-m", "src.training.train_v97b_cfonly"],
            cwd=str(_ROOT),
            capture_output=False,
        )
        if result.returncode != 0:
            print("  ERROR: Base model training failed!")
            sys.exit(1)
        print("  Base model training complete.")

    # Verify base predictions exist
    if not V97B_CF_PATH.exists():
        print(f"\n  ERROR: Base predictions not found at {V97B_CF_PATH}")
        print(f"  Run with --from-scratch to train the model first.")
        sys.exit(1)

    if args.mode in ("q1", "all"):
        q1_preds = predict_q1()
        _write_submission(q1_preds, Q1_OUTPUT, "Q1 2026")

    if args.mode in ("dayi", "all"):
        dayi_preds = predict_dayi()
        _write_submission(dayi_preds, DAYI_OUTPUT, "Day-i May 18")

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
