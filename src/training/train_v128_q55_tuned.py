"""V128: Tune the q55 KNN blend (v127.D = 7.3415 on LB).

Winning recipe: 5% KNN-q55 + 95% v97b-corrected
Now try:
  - Higher percentiles (q57, q60) 
  - Use v126 as base instead of v97b (v126 already has 5% KNN-50)
  - Vary the blend weight (3%, 7%, 10%)
  - Combine q55 KNN with the existing v126 blend
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TIMESTAMP_COL

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


def main() -> None:
    print("=" * 72)
    print("V128: Tune q55 KNN blend (LB best = 7.3415)")
    print("=" * 72)

    # --- Load ---
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

    valid_mask = (
        np.isfinite(train_power) & (train_power >= 0) & (train_power <= CAPACITY_MW) &
        ~((ws_train > 8) & (train_power < 5))
    )

    # Build features
    ts_train = pd.to_datetime(train.iloc[:, 0])
    ts_test = pd.to_datetime(valid.iloc[:, 0])
    hour_train = ts_train.dt.hour.to_numpy(dtype=np.float32)
    hour_test = ts_test.dt.hour.to_numpy(dtype=np.float32)

    def _feats(df, hour):
        return np.column_stack([
            df["wind_speed_120m"].to_numpy(dtype=np.float32) * 3.0,
            df["wind_speed_80m"].to_numpy(dtype=np.float32),
            df["wind_speed_10m"].to_numpy(dtype=np.float32),
            df["temperature_80m"].to_numpy(dtype=np.float32),
            df["pressure_msl"].to_numpy(dtype=np.float32) / 100.0,
            df["wind_gusts_10m"].to_numpy(dtype=np.float32),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
        ]).astype(np.float32)

    X_train = _feats(train, hour_train)[valid_mask]
    X_test = _feats(valid, hour_test)
    power_v = train_power[valid_mask]
    active_v = train_active[valid_mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Build BallTree with K=100
    tree = BallTree(X_tr_s, metric="euclidean")
    _, idx = tree.query(X_te_s, k=100)

    # Compute various quantiles
    quantile_preds = {}
    for q in [45, 50, 55, 57, 60, 65]:
        preds_q = np.zeros(len(X_test), dtype=np.float64)
        for i in range(len(X_test)):
            np_power = power_v[idx[i]]
            np_active = active_v[idx[i]]
            per_turb = np_power / np.maximum(np_active, 1)
            adjusted = per_turb * test_active[i]
            preds_q[i] = np.percentile(adjusted, q)
        quantile_preds[q] = np.clip(preds_q, 0, CAPACITY_MW)
        print(f"  q{q}: mean={quantile_preds[q].mean():.2f}")

    # Load bases
    v97b_preds = pd.read_csv(V97B_BEST_PATH)[TARGET_COL_NAME].to_numpy(dtype=np.float64)
    v126_preds = pd.read_csv(V126_BEST_PATH)[TARGET_COL_NAME].to_numpy(dtype=np.float64)

    print(f"\n  Generating blends...")

    # A: q55 blends with v97b at various weights
    for w in [0.03, 0.05, 0.07, 0.10]:
        blend = np.clip(w * quantile_preds[55] + (1 - w) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v128.A_q55_{int(w*100):02d}_v97b.csv",
               f"{w:.0%} q55 + {1-w:.0%} v97b")

    # B: q55 blends with v126 (v126 = 5% KNN-50-median + 95% v97b)
    for w in [0.03, 0.05, 0.07]:
        blend = np.clip(w * quantile_preds[55] + (1 - w) * v126_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v128.B_q55_{int(w*100):02d}_v126.csv",
               f"{w:.0%} q55 + {1-w:.0%} v126")

    # C: Higher quantiles (q57, q60) with v97b
    for q in [57, 60]:
        for w in [0.05, 0.07]:
            blend = np.clip(w * quantile_preds[q] + (1 - w) * v97b_preds, 0, CAPACITY_MW)
            _write(blend, OUTPUT_DIR / f"v128.C_q{q}_{int(w*100):02d}_v97b.csv",
                   f"{w:.0%} q{q} + {1-w:.0%} v97b")

    # D: q55 on v126 base (stacking: v126 has KNN-50 median, now add KNN-100 q55)
    for w in [0.05, 0.07]:
        blend = np.clip(w * quantile_preds[55] + (1 - w) * v126_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v128.D_q55_{int(w*100):02d}_on_v126.csv",
               f"{w:.0%} q55-K100 on v126 base")

    # E: Mix of quantiles (q50 + q55 + q60 averaged, then blended)
    qmix = np.clip((quantile_preds[50] + quantile_preds[55] + quantile_preds[60]) / 3, 0, CAPACITY_MW)
    for w in [0.05, 0.07, 0.10]:
        blend = np.clip(w * qmix + (1 - w) * v97b_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v128.E_qmix_{int(w*100):02d}_v97b.csv",
               f"{w:.0%} q-mix + {1-w:.0%} v97b")

    # F: Directly blend v126 with v127.D winner (they use different KNN variants)
    # v126 = 5% KNN-50-median + 95% v97b_corr
    # v127.D = 5% KNN-100-q55 + 95% v97b_corr
    # Their average = 2.5% KNN-50-median + 2.5% KNN-100-q55 + 95% v97b_corr
    v127d_preds = pd.read_csv(OUTPUT_DIR / "v127.D_q55_05.csv")[TARGET_COL_NAME].to_numpy(dtype=np.float64)
    for w126 in [0.4, 0.5, 0.6]:
        blend = np.clip(w126 * v126_preds + (1 - w126) * v127d_preds, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v128.F_v126_{int(w126*100)}_v127d_{int((1-w126)*100)}.csv",
               f"{w126:.0%} v126 + {1-w126:.0%} v127.D")

    print(f"\n  Top picks:")
    print(f"    v128.B_q55_05_v126.csv    (q55 on v126 base — stacks two KNN types)")
    print(f"    v128.C_q57_05_v97b.csv    (higher quantile, might push further)")
    print(f"    v128.F_v126_50_v127d_50   (average of two LB-best submissions)")
    print(f"    v128.A_q55_07_v97b.csv    (more q55 weight)")


if __name__ == "__main__":
    main()
