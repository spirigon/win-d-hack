"""V127: Quantile-blended LGBM + direction-stratified KNN.

Two new ideas combined with the winning v97b-corrected base:

1. Quantile averaging: v97b uses regression_l1 (median). But the error
   distribution is asymmetric (skew=0.89). Training at q=0.45 might give
   a slightly better MAE-minimizing point predictor. Since we can't
   retrain easily, we APPROXIMATE this by shifting the prediction slightly
   toward the under-prediction side (since over-prediction contributes
   more to MAE due to the positive skew).

2. Direction-stratified KNN: instead of matching ALL weather features,
   first bin by wind direction sector (8 bins), then find nearest
   neighbors WITHIN the same sector. This captures direction-dependent
   wake effects and power curve variations.

Also: KNN with K=100 (scored better standalone on OOF).
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
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
V97B_BEST_PATH = _ROOT / "submissions" / "archive" / "v123.B_hw0.7_q0.7.csv"
V126_BEST_PATH = _ROOT / "submissions" / "archive" / "v126.1_knn05_v97b95.csv"
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TARGET_COL_NAME = "Выработка. Результирующий расчет"
TOTAL_TURBINES = 26
MAINT_COL = "Кол-во_ВЭУ_в_ремонте"


def _write(preds: np.ndarray, path: Path, label: str = "") -> None:
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TARGET_COL_NAME: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<55s} mean={preds.mean():.2f}  {label}")


def _direction_sector(dir_raw: np.ndarray, n_sectors: int = 8) -> np.ndarray:
    """Convert raw direction (0.001-0.360 format) to sector index."""
    deg = dir_raw * 1000.0
    sector = (deg / (360.0 / n_sectors)).astype(int) % n_sectors
    return sector


def main() -> None:
    set_global_seed(42)
    t_start = time.time()

    print("=" * 72)
    print("V127: Quantile shift + Direction-stratified KNN")
    print("=" * 72)

    # --- Load data ---
    train = pd.read_csv(TRAIN_PATH)
    train.columns = train.columns.str.strip()
    valid = pd.read_csv(VALID_PATH)
    valid.columns = valid.columns.str.strip()

    train_power = train[TARGET_COL_NAME].to_numpy(dtype=np.float32)
    train_maint = train[MAINT_COL].to_numpy(dtype=np.float32)
    train_active = TOTAL_TURBINES - train_maint
    test_maint = valid[MAINT_COL].to_numpy(dtype=np.float32)
    test_active = TOTAL_TURBINES - test_maint

    ws_train = train["wind_speed_120m"].to_numpy(dtype=np.float32)
    ws_test = valid["wind_speed_120m"].to_numpy(dtype=np.float32)

    # Valid mask
    valid_mask = np.isfinite(train_power) & (train_power >= 0) & (train_power <= CAPACITY_MW)
    curtailment = (ws_train > 8) & (train_power < 5) & valid_mask
    valid_mask = valid_mask & ~curtailment

    # Direction sectors
    dir_train = train["wind_direction_120m"].to_numpy(dtype=np.float32)
    dir_test = valid["wind_direction_120m"].to_numpy(dtype=np.float32)
    sector_train = _direction_sector(dir_train)
    sector_test = _direction_sector(dir_test)

    # Load current best
    v97b_preds = pd.read_csv(V97B_BEST_PATH)[TARGET_COL_NAME].to_numpy(dtype=np.float64)
    v126_preds = pd.read_csv(V126_BEST_PATH)[TARGET_COL_NAME].to_numpy(dtype=np.float64)

    # --- 1. Direction-stratified KNN ---
    print("\n[1/3] Direction-stratified KNN (K=50, 8 sectors)...")

    # Build per-sector KNN
    # Features: ws_120, ws_80, ws_10, temp, pressure, gust, hour_sin/cos
    ts_train = pd.to_datetime(train.iloc[:, 0])
    ts_test = pd.to_datetime(valid.iloc[:, 0])

    hour_train = ts_train.dt.hour.to_numpy(dtype=np.float32)
    hour_test = ts_test.dt.hour.to_numpy(dtype=np.float32)

    def _build_features(ws120, ws80, ws10, temp, pressure, gust, hour):
        return np.column_stack([
            ws120 * 3.0,  # heavy weight on primary wind speed
            ws80,
            ws10,
            temp,
            pressure / 100.0,  # scale to similar range
            gust,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
        ]).astype(np.float32)

    X_train_all = _build_features(
        ws_train, train["wind_speed_80m"].to_numpy(dtype=np.float32),
        train["wind_speed_10m"].to_numpy(dtype=np.float32),
        train["temperature_80m"].to_numpy(dtype=np.float32),
        train["pressure_msl"].to_numpy(dtype=np.float32),
        train["wind_gusts_10m"].to_numpy(dtype=np.float32),
        hour_train,
    )
    X_test_all = _build_features(
        ws_test, valid["wind_speed_80m"].to_numpy(dtype=np.float32),
        valid["wind_speed_10m"].to_numpy(dtype=np.float32),
        valid["temperature_80m"].to_numpy(dtype=np.float32),
        valid["pressure_msl"].to_numpy(dtype=np.float32),
        valid["wind_gusts_10m"].to_numpy(dtype=np.float32),
        hour_test,
    )

    knn_sector_preds = np.zeros(len(valid), dtype=np.float64)
    K = 50

    for s in range(8):
        train_s_mask = valid_mask & (sector_train == s)
        test_s_mask = sector_test == s

        if train_s_mask.sum() < K or test_s_mask.sum() == 0:
            # Fallback to global
            continue

        X_tr_s = X_train_all[train_s_mask]
        power_s = train_power[train_s_mask]
        active_s = train_active[train_s_mask]
        X_te_s = X_test_all[test_s_mask]
        active_te_s = test_active[test_s_mask]

        scaler_s = StandardScaler()
        X_tr_s_scaled = scaler_s.fit_transform(X_tr_s)
        X_te_s_scaled = scaler_s.transform(X_te_s)

        tree_s = BallTree(X_tr_s_scaled, metric="euclidean")
        d, idx = tree_s.query(X_te_s_scaled, k=K)

        for i in range(len(X_te_s_scaled)):
            np_power = power_s[idx[i]]
            np_active = active_s[idx[i]]
            per_turb = np_power / np.maximum(np_active, 1)
            knn_sector_preds[test_s_mask][i] = np.median(per_turb * active_te_s[i])

        print(f"    Sector {s}: {train_s_mask.sum()} train, {test_s_mask.sum()} test")

    knn_sector_preds = np.clip(knn_sector_preds, 0, CAPACITY_MW)
    print(f"  Sector KNN: mean={knn_sector_preds.mean():.2f}, std={knn_sector_preds.std():.2f}")

    # --- 2. KNN K=100 (global, scored better standalone) ---
    print("\n[2/3] Global KNN K=100...")
    scaler_g = StandardScaler()
    X_tr_g = scaler_g.fit_transform(X_train_all[valid_mask])
    X_te_g = scaler_g.transform(X_test_all)
    tree_g = BallTree(X_tr_g, metric="euclidean")
    d100, idx100 = tree_g.query(X_te_g, k=100)

    knn100_preds = np.zeros(len(valid), dtype=np.float64)
    power_v = train_power[valid_mask]
    active_v = train_active[valid_mask]
    for i in range(len(X_te_g)):
        np_power = power_v[idx100[i]]
        np_active = active_v[idx100[i]]
        per_turb = np_power / np.maximum(np_active, 1)
        knn100_preds[i] = np.median(per_turb * test_active[i])
    knn100_preds = np.clip(knn100_preds, 0, CAPACITY_MW)
    print(f"  KNN-100: mean={knn100_preds.mean():.2f}, std={knn100_preds.std():.2f}")

    # --- 3. Quantile-shifted predictions ---
    print("\n[3/3] Quantile-shifted predictions...")
    # The error skew is +0.89 (more large positive errors = under-predictions)
    # Shifting predictions DOWN slightly (toward q=0.48) could help
    # But our bias corrections already ADD to predictions (Q1 under-predicts)
    # So the optimal shift direction depends on the regime.
    # Simple approach: instead of shifting globally, use KNN to estimate the
    # conditional quantile. For each test point, use q=0.45 of neighbors instead of median.

    knn_q45 = np.zeros(len(valid), dtype=np.float64)
    knn_q55 = np.zeros(len(valid), dtype=np.float64)
    # Use K=100 global tree already built
    for i in range(len(X_te_g)):
        np_power = power_v[idx100[i]]
        np_active = active_v[idx100[i]]
        per_turb = np_power / np.maximum(np_active, 1)
        adjusted = per_turb * test_active[i]
        knn_q45[i] = np.percentile(adjusted, 45)
        knn_q55[i] = np.percentile(adjusted, 55)
    knn_q45 = np.clip(knn_q45, 0, CAPACITY_MW)
    knn_q55 = np.clip(knn_q55, 0, CAPACITY_MW)

    # --- Generate submissions ---
    print("\n  Generating submissions...")

    # A: Sector KNN blends with v97b-corrected
    for w in [0.05, 0.10]:
        blend = np.clip(w * knn_sector_preds + (1 - w) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v127.A_sector_knn{int(w*100):02d}.csv",
               f"{w:.0%} sector-KNN + {1-w:.0%} v97b")

    # B: KNN K=100 blends with v97b-corrected
    for w in [0.05, 0.07, 0.10]:
        blend = np.clip(w * knn100_preds + (1 - w) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v127.B_knn100_{int(w*100):02d}.csv",
               f"{w:.0%} KNN-100 + {1-w:.0%} v97b")

    # C: KNN q45 blend (tilted toward lower predictions)
    for w in [0.05, 0.10]:
        blend = np.clip(w * knn_q45 + (1 - w) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v127.C_q45_{int(w*100):02d}.csv",
               f"{w:.0%} KNN-q45 + {1-w:.0%} v97b")

    # D: KNN q55 blend (tilted toward higher predictions — might help since model under-predicts Q1)
    for w in [0.05, 0.10]:
        blend = np.clip(w * knn_q55 + (1 - w) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v127.D_q55_{int(w*100):02d}.csv",
               f"{w:.0%} KNN-q55 + {1-w:.0%} v97b")

    # E: Replace v97b with v126 (already includes 5% KNN) as base, add more diversity
    for w in [0.05, 0.10]:
        # This stacks: v126 = 95% v97b + 5% KNN50, now add more KNN100
        blend = np.clip(w * knn100_preds + (1 - w) * v126_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v127.E_v126base_knn100_{int(w*100):02d}.csv",
               f"{w:.0%} KNN-100 on v126 base")

    # F: Sector KNN on v126 base
    blend = np.clip(0.05 * knn_sector_preds + 0.95 * v126_preds, 0, CAPACITY_MW)
    _write(blend, OUTPUT_DIR / "v127.F_v126base_sector05.csv", "5% sector-KNN on v126")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"\n  Top picks:")
    print(f"    v127.B_knn100_05.csv      (K=100 scored better standalone)")
    print(f"    v127.A_sector_knn05.csv   (direction-aware diversity)")
    print(f"    v127.D_q55_05.csv         (quantile tilt toward higher preds)")
    print(f"    v127.E_v126base_knn100_05 (add KNN100 on top of current best)")


if __name__ == "__main__":
    main()
