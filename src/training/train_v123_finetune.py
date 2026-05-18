"""V123: Fine-tune around v122.D_q1bias_0.7 = 7.354 on LB.

Winning recipe: CF + hw_bias*0.8 + co_bias*0.5 + q1_bias*0.7
Now fine-tune: sweep q1 weight [0.6..1.0] in small steps, combine with
other corrections that might stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TIMESTAMP_COL

V97B_OOF_PATH = _ROOT / "data" / "processed" / "v97b_oof.parquet"
V97B_SUB_CF_PATH = _ROOT / "submissions" / "archive" / "v97b.0_cfonly.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TARGET_COL_NAME = "Выработка. Результирующий расчет"


def _write(preds: np.ndarray, path: Path, label: str = "") -> None:
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TARGET_COL_NAME: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<50s} mean={preds.mean():.2f}  {label}")


def main() -> None:
    print("=" * 72)
    print("V123: Fine-tune around v122.D q1=0.7 (LB=7.354)")
    print("=" * 72)

    # --- Load ---
    oof = pd.read_parquet(V97B_OOF_PATH)
    oof["error_cf"] = oof["target_mw"] - oof["pred_cf_mw"]
    oof["ts"] = pd.to_datetime(oof["ts"])

    v97b_cf_sub = pd.read_csv(V97B_SUB_CF_PATH)
    pred_col = [c for c in v97b_cf_sub.columns if c != TIMESTAMP_COL][0]
    test_pred_cf = v97b_cf_sub[pred_col].to_numpy(dtype=np.float64)
    sub_ts = pd.to_datetime(v97b_cf_sub[TIMESTAMP_COL])

    valid_df = pd.read_csv(VALID_PATH)
    valid_df[TIMESTAMP_COL] = pd.to_datetime(valid_df.iloc[:, 0])
    ws_map = dict(zip(valid_df[TIMESTAMP_COL], valid_df["wind_speed_120m"]))
    test_ws = np.array([ws_map.get(t, 8.0) for t in sub_ts], dtype=np.float32)
    test_month = sub_ts.dt.month.to_numpy()
    test_hour = sub_ts.dt.hour.to_numpy()

    # --- Biases (CF model) ---
    hw_b = float(oof[(oof["ws_120"] >= 12) & (oof["ws_120"] < 18)]["error_cf"].mean())
    co_b = float(oof[oof["ws_120"] >= 18]["error_cf"].mean())
    mid_b = float(oof[(oof["ws_120"] >= 8) & (oof["ws_120"] < 12)]["error_cf"].mean())
    low_b = float(oof[(oof["ws_120"] >= 4) & (oof["ws_120"] < 8)]["error_cf"].mean())
    calm_b = float(oof[oof["ws_120"] < 4]["error_cf"].mean())
    q1_b = float(oof[oof["ts"].dt.month.isin([1, 2, 3])]["error_cf"].mean())
    night_b = float(oof[oof["ts"].dt.hour.isin([20,21,22,23,0,1,2,3])]["error_cf"].mean())
    day_b = float(oof[~oof["ts"].dt.hour.isin([20,21,22,23,0,1,2,3])]["error_cf"].mean())
    m1_b = float(oof[oof["ts"].dt.month == 1]["error_cf"].mean())
    m2_b = float(oof[oof["ts"].dt.month == 2]["error_cf"].mean())
    m3_b = float(oof[oof["ts"].dt.month == 3]["error_cf"].mean())

    print(f"  hw={hw_b:+.3f} co={co_b:+.3f} mid={mid_b:+.3f} low={low_b:+.3f} calm={calm_b:+.3f}")
    print(f"  q1={q1_b:+.3f} night={night_b:+.3f} day={day_b:+.3f}")
    print(f"  m1={m1_b:+.3f} m2={m2_b:+.3f} m3={m3_b:+.3f}")

    # --- Masks ---
    hw_mask = (test_ws >= 12) & (test_ws < 18)
    co_mask = test_ws >= 18
    mid_mask = (test_ws >= 8) & (test_ws < 12)
    low_mask = (test_ws >= 4) & (test_ws < 8)
    calm_mask = test_ws < 4
    night_mask = np.isin(test_hour, [20, 21, 22, 23, 0, 1, 2, 3])

    def _base_correction(cf: np.ndarray, hw_w=0.8, co_w=0.5) -> np.ndarray:
        """Apply the base high-wind + cutout correction."""
        out = cf.copy()
        out[hw_mask] += hw_b * hw_w
        out[co_mask] += co_b * co_w
        return out

    print(f"\n  --- A: Q1 bias sweep (fine grain) ---")
    for q1_w in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        v = _base_correction(test_pred_cf)
        v += q1_b * q1_w
        _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / f"v123.A_q1_{q1_w:.2f}.csv")

    print(f"\n  --- B: Q1=0.7 + high-wind strength sweep ---")
    for hw_w in [0.6, 0.7, 0.9, 1.0, 1.2]:
        v = _base_correction(test_pred_cf, hw_w=hw_w)
        v += q1_b * 0.7
        _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / f"v123.B_hw{hw_w:.1f}_q0.7.csv")

    print(f"\n  --- C: Q1=0.7 + additional mid-wind correction ---")
    for mid_w in [0.3, 0.5, 0.7]:
        v = _base_correction(test_pred_cf)
        v[mid_mask] += mid_b * mid_w
        v += q1_b * 0.7
        _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / f"v123.C_mid{mid_w:.1f}_q0.7.csv")

    print(f"\n  --- D: Q1=0.7 + per-month fine-tuning ---")
    # Instead of uniform Q1 shift, apply month-specific
    for m_w in [0.5, 0.7, 0.9]:
        v = _base_correction(test_pred_cf)
        v[test_month == 1] += m1_b * m_w
        v[test_month == 2] += m2_b * m_w
        v[test_month == 3] += m3_b * m_w
        _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / f"v123.D_month{m_w:.1f}.csv")

    print(f"\n  --- E: Q1=0.7 + night correction ---")
    for n_w in [0.2, 0.3, 0.5]:
        v = _base_correction(test_pred_cf)
        v += q1_b * 0.7
        v[night_mask] += night_b * n_w
        v[~night_mask] += day_b * n_w
        _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / f"v123.E_night{n_w:.1f}_q0.7.csv")

    print(f"\n  --- F: Best combo candidates ---")
    # F1: Q1=0.8 + hw=1.0 + co=0.7 (stronger all around)
    v = _base_correction(test_pred_cf, hw_w=1.0, co_w=0.7)
    v += q1_b * 0.8
    _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / "v123.F1_strong.csv")

    # F2: Q1=0.7 + all regimes mild
    v = _base_correction(test_pred_cf)
    v[mid_mask] += mid_b * 0.3
    v[low_mask] += low_b * 0.2
    v += q1_b * 0.7
    _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / "v123.F2_all_mild.csv")

    # F3: Month-specific + hw stronger
    v = _base_correction(test_pred_cf, hw_w=1.0, co_w=0.7)
    v[test_month == 1] += m1_b * 0.7
    v[test_month == 2] += m2_b * 0.7
    v[test_month == 3] += m3_b * 0.7
    _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / "v123.F3_month_hw_strong.csv")

    # F4: Conservative Q1=0.75 + hw=0.9
    v = _base_correction(test_pred_cf, hw_w=0.9)
    v += q1_b * 0.75
    _write(np.clip(v, 0, CAPACITY_MW), OUTPUT_DIR / "v123.F4_q0.75_hw0.9.csv")

    print(f"\n  Top picks:")
    print(f"    v123.A_q1_0.80.csv   (Q1 push to 0.8)")
    print(f"    v123.A_q1_0.75.csv   (midpoint between 0.7 and 0.8)")
    print(f"    v123.F1_strong.csv   (all corrections stronger)")
    print(f"    v123.D_month0.7.csv  (per-month replaces uniform Q1)")
    print(f"    v123.F4_q0.75_hw0.9  (balanced increase)")


if __name__ == "__main__":
    main()
