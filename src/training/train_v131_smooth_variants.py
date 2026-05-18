"""V131: Savgol smoothing applied to various bases + order-of-operations."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OUTPUT_DIR = _ROOT / "submissions" / "archive"
TCN = "Выработка. Результирующий расчет"


def _write(preds, path, label=""):
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    pd.DataFrame({TCN: preds}).to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<55s} mean={preds.mean():.2f}  {label}")


def _smooth(preds_sub_order, w=7, p=2):
    """Apply savgol to submission-order preds (reverse chrono → chrono → smooth → reverse)."""
    chrono = preds_sub_order[::-1]
    smoothed = savgol_filter(chrono, w, p)
    return np.clip(smoothed[::-1], 0, CAPACITY_MW)


def main():
    print("=" * 72)
    print("V131: Smoothing variants + order of operations")
    print("=" * 72)

    # Load bases
    v97b = pd.read_csv(OUTPUT_DIR / "v123.B_hw0.7_q0.7.csv")[TCN].values
    v126 = pd.read_csv(OUTPUT_DIR / "v126.1_knn05_v97b95.csv")[TCN].values
    v127d = pd.read_csv(OUTPUT_DIR / "v127.D_q55_05.csv")[TCN].values
    v128a = pd.read_csv(OUTPUT_DIR / "v128.A_q55_07_v97b.csv")[TCN].values

    # 1. Smooth each base individually with savgol(7,2)
    print("\n  1. Savgol(7,2) on individual bases:")
    for name, preds in [("v97b_corr", v97b), ("v126", v126), ("v127d", v127d)]:
        s = _smooth(preds)
        _write(s, OUTPUT_DIR / f"v131.{name}_s72.csv", f"{name} + savgol(7,2)")

    # 2. Different smooth widths on v128.A (LB best before smooth = 7.340)
    print("\n  2. Width sweep on v128.A:")
    for w, p in [(5, 2), (5, 3), (9, 2), (9, 3), (11, 2), (13, 3)]:
        if w < 2 * p + 1:
            continue
        s = _smooth(v128a, w, p)
        _write(s, OUTPUT_DIR / f"v131.v128_s{w}{p}.csv", f"v128 + savgol({w},{p})")

    # 3. Smooth v97b FIRST, then blend with KNN-q55 (different order from v128+smooth)
    print("\n  3. Smooth-then-blend (vs blend-then-smooth):")
    v97b_smooth = _smooth(v97b)

    # Build KNN-q55 predictions
    train = pd.read_csv(TRAIN_PATH)
    train.columns = train.columns.str.strip()
    valid = pd.read_csv(VALID_PATH)
    valid.columns = valid.columns.str.strip()

    tp = train[TCN].values.astype(np.float32)
    ta = 26 - train["Кол-во_ВЭУ_в_ремонте"].values.astype(np.float32)
    va_turb = 26 - valid["Кол-во_ВЭУ_в_ремонте"].values.astype(np.float32)
    ws = train["wind_speed_120m"].values.astype(np.float32)
    mask = np.isfinite(tp) & (tp >= 0) & (tp <= CAPACITY_MW) & ~((ws > 8) & (tp < 5))

    ht = pd.to_datetime(train.iloc[:, 0]).dt.hour.values.astype(np.float32)
    hv = pd.to_datetime(valid.iloc[:, 0]).dt.hour.values.astype(np.float32)

    Xtr = np.column_stack([
        train["wind_speed_120m"].values * 3, train["wind_speed_80m"].values,
        train["wind_speed_10m"].values, train["temperature_80m"].values,
        train["pressure_msl"].values / 100, train["wind_gusts_10m"].values,
        np.sin(2 * np.pi * ht / 24), np.cos(2 * np.pi * ht / 24),
    ]).astype(np.float32)[mask]

    Xte = np.column_stack([
        valid["wind_speed_120m"].values * 3, valid["wind_speed_80m"].values,
        valid["wind_speed_10m"].values, valid["temperature_80m"].values,
        valid["pressure_msl"].values / 100, valid["wind_gusts_10m"].values,
        np.sin(2 * np.pi * hv / 24), np.cos(2 * np.pi * hv / 24),
    ]).astype(np.float32)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    tree = BallTree(Xtr_s)
    _, idx = tree.query(Xte_s, k=100)
    pv, av = tp[mask], ta[mask]

    # KNN-q55 predictions (in valid_features.csv order = reverse chrono)
    q55 = np.array([
        np.percentile(pv[idx[i]] / np.maximum(av[idx[i]], 1) * va_turb[i], 55)
        for i in range(len(Xte_s))
    ])
    q55 = np.clip(q55, 0, CAPACITY_MW)

    # Also smooth the KNN predictions
    q55_smooth = np.clip(savgol_filter(q55, 7, 2), 0, CAPACITY_MW)

    # Order A: smooth v97b → blend with raw q55
    for w_knn in [0.05, 0.07]:
        blend = np.clip(w_knn * q55 + (1 - w_knn) * v97b_smooth, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v131.smooth_then_blend_q{int(w_knn*100):02d}.csv",
               f"smooth(v97b) + {w_knn:.0%} raw-q55")

    # Order B: smooth v97b → blend with smooth q55
    for w_knn in [0.05, 0.07]:
        blend = np.clip(w_knn * q55_smooth + (1 - w_knn) * v97b_smooth, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v131.both_smooth_blend_q{int(w_knn*100):02d}.csv",
               f"smooth(v97b) + {w_knn:.0%} smooth(q55)")

    # 4. Smooth v126 and v127d (they already have some KNN)
    print("\n  4. Smooth existing LB winners:")
    v126_s = _smooth(v126)
    v127d_s = _smooth(v127d)
    _write(v126_s, OUTPUT_DIR / "v131.v126_s72.csv", "v126(LB=7.345) + savgol(7,2)")
    _write(v127d_s, OUTPUT_DIR / "v131.v127d_s72.csv", "v127d(LB=7.342) + savgol(7,2)")

    # 5. Average of smoothed submissions
    print("\n  5. Ensembles of smoothed predictions:")
    avg_3 = np.clip((v97b_smooth + v126_s + v127d_s) / 3, 0, CAPACITY_MW)
    _write(avg_3, OUTPUT_DIR / "v131.avg3_smooth.csv", "avg(smooth v97b, v126, v127d)")

    v128_s = _smooth(v128a)
    avg_2 = np.clip((v128_s + v127d_s) / 2, 0, CAPACITY_MW)
    _write(avg_2, OUTPUT_DIR / "v131.avg2_v128s_v127ds.csv", "avg(smooth v128, smooth v127d)")

    print(f"\n  Top picks:")
    print(f"    v131.v127d_s72.csv              (smooth LB=7.342 winner)")
    print(f"    v131.both_smooth_blend_q07.csv  (both smooth, 7% q55)")
    print(f"    v131.avg2_v128s_v127ds.csv      (ensemble of 2 smoothed)")
    print(f"    v131.v97b_corr_s72.csv          (smooth v97b-corrected)")


if __name__ == "__main__":
    main()
