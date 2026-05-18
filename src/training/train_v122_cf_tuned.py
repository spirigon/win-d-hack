"""V122: CF-base correction sweep — v121.F (CF + corrections) scored 7.388.

CF-only base beats blend50 base on LB. Now tune correction strengths
on the CF base to push lower.

v121.F recipe: CF + hw_bias*0.8 + co_bias*0.5 + global*0.3
  hw_bias = -1.841 MW, co_bias = -11.911 MW, global = +0.185 MW

Generate variants sweeping:
  - High-wind correction strength
  - Global shift strength
  - Adding mid/low wind corrections
  - Month-specific corrections
  - Q1-specific bias
  - Night/day correction
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
from src.utils.seeding import set_global_seed

# --- Paths ---
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
    print(f"  {path.name:<45s} mean={preds.mean():.2f}  {label}")


def main() -> None:
    set_global_seed(42)

    print("=" * 72)
    print("V122: CF-base correction sweep (v121.F = 7.388 on LB)")
    print("=" * 72)

    # --- Load data ---
    oof = pd.read_parquet(V97B_OOF_PATH)
    oof["pred_cf"] = oof["pred_cf_mw"]
    oof["error_cf"] = oof["target_mw"] - oof["pred_cf_mw"]
    oof["ts"] = pd.to_datetime(oof["ts"])

    # Load CF test predictions
    v97b_cf_sub = pd.read_csv(V97B_SUB_CF_PATH)
    pred_col = [c for c in v97b_cf_sub.columns if c != TIMESTAMP_COL][0]
    test_pred_cf = v97b_cf_sub[pred_col].to_numpy(dtype=np.float64)
    sub_ts = pd.to_datetime(v97b_cf_sub[TIMESTAMP_COL])

    # Load wind speed
    valid_df = pd.read_csv(VALID_PATH)
    valid_df[TIMESTAMP_COL] = pd.to_datetime(valid_df.iloc[:, 0])
    ws_map = dict(zip(valid_df[TIMESTAMP_COL], valid_df["wind_speed_120m"]))
    test_ws = np.array([ws_map.get(t, 8.0) for t in sub_ts], dtype=np.float32)
    test_month = sub_ts.dt.month.to_numpy()
    test_hour = sub_ts.dt.hour.to_numpy()

    # --- OOF bias stats (CF model) ---
    print("\n  CF model OOF error stats:")
    global_bias_cf = float(oof["error_cf"].mean())
    print(f"  Global CF bias: {global_bias_cf:+.3f} MW")

    ws_biases = {}
    for lo, hi, name in [(0, 4, "calm"), (4, 8, "low"), (8, 12, "mid"),
                         (12, 18, "high"), (18, 100, "cutout")]:
        sub = oof[(oof["ws_120"] >= lo) & (oof["ws_120"] < hi)]
        b = float(sub["error_cf"].mean()) if len(sub) > 5 else 0
        ws_biases[name] = b
        print(f"  ws {lo:2d}-{hi:3d} ({name:6s}): bias={b:+.3f} MW  n={len(sub)}")

    q1_bias_cf = float(oof[oof["ts"].dt.month.isin([1, 2, 3])]["error_cf"].mean())
    print(f"  Q1-months CF bias: {q1_bias_cf:+.3f} MW")

    night_bias_cf = float(oof[oof["ts"].dt.hour.isin([20,21,22,23,0,1,2,3])]["error_cf"].mean())
    day_bias_cf = float(oof[~oof["ts"].dt.hour.isin([20,21,22,23,0,1,2,3])]["error_cf"].mean())
    print(f"  Night CF bias: {night_bias_cf:+.3f} MW")
    print(f"  Day CF bias: {day_bias_cf:+.3f} MW")

    m1_cf = float(oof[oof["ts"].dt.month == 1]["error_cf"].mean())
    m2_cf = float(oof[oof["ts"].dt.month == 2]["error_cf"].mean())
    m3_cf = float(oof[oof["ts"].dt.month == 3]["error_cf"].mean())
    print(f"  Jan={m1_cf:+.3f} Feb={m2_cf:+.3f} Mar={m3_cf:+.3f}")

    # --- Test masks ---
    hw_mask = (test_ws >= 12) & (test_ws < 18)
    co_mask = test_ws >= 18
    mid_mask = (test_ws >= 8) & (test_ws < 12)
    low_mask = (test_ws >= 4) & (test_ws < 8)
    calm_mask = test_ws < 4
    night_mask = np.isin(test_hour, [20, 21, 22, 23, 0, 1, 2, 3])

    hw_b = ws_biases["high"]
    co_b = ws_biases["cutout"]
    mid_b = ws_biases["mid"]
    low_b = ws_biases["low"]
    calm_b = ws_biases["calm"]

    print(f"\n  Generating variants on CF base...")

    # --- A: v121.F exact (baseline) ---
    va = test_pred_cf.copy()
    va[hw_mask] += hw_b * 0.8
    va[co_mask] += co_b * 0.5
    va += global_bias_cf * 0.3
    _write(np.clip(va, 0, CAPACITY_MW), OUTPUT_DIR / "v122.A_baseline.csv", "(= v121.F)")

    # --- B: Sweep high-wind strength ---
    for hw_w in [0.5, 0.6, 1.0, 1.2]:
        vb = test_pred_cf.copy()
        vb[hw_mask] += hw_b * hw_w
        vb[co_mask] += co_b * 0.5
        vb += global_bias_cf * 0.3
        _write(np.clip(vb, 0, CAPACITY_MW), OUTPUT_DIR / f"v122.B_hw{hw_w:.1f}.csv")

    # --- C: Sweep global strength ---
    for g_w in [0.0, 0.5, 0.7, 1.0]:
        vc = test_pred_cf.copy()
        vc[hw_mask] += hw_b * 0.8
        vc[co_mask] += co_b * 0.5
        vc += global_bias_cf * g_w
        _write(np.clip(vc, 0, CAPACITY_MW), OUTPUT_DIR / f"v122.C_g{g_w:.1f}.csv")

    # --- D: Add Q1-specific shift instead of global ---
    vd = test_pred_cf.copy()
    vd[hw_mask] += hw_b * 0.8
    vd[co_mask] += co_b * 0.5
    vd += q1_bias_cf * 0.3
    _write(np.clip(vd, 0, CAPACITY_MW), OUTPUT_DIR / "v122.D_q1bias_0.3.csv")

    vd2 = test_pred_cf.copy()
    vd2[hw_mask] += hw_b * 0.8
    vd2[co_mask] += co_b * 0.5
    vd2 += q1_bias_cf * 0.5
    _write(np.clip(vd2, 0, CAPACITY_MW), OUTPUT_DIR / "v122.D_q1bias_0.5.csv")

    vd3 = test_pred_cf.copy()
    vd3[hw_mask] += hw_b * 0.8
    vd3[co_mask] += co_b * 0.5
    vd3 += q1_bias_cf * 0.7
    _write(np.clip(vd3, 0, CAPACITY_MW), OUTPUT_DIR / "v122.D_q1bias_0.7.csv")

    # --- E: All regimes + Q1 bias ---
    ve = test_pred_cf.copy()
    ve[calm_mask] += calm_b * 0.3
    ve[low_mask] += low_b * 0.3
    ve[mid_mask] += mid_b * 0.5
    ve[hw_mask] += hw_b * 0.8
    ve[co_mask] += co_b * 0.5
    ve += q1_bias_cf * 0.3
    _write(np.clip(ve, 0, CAPACITY_MW), OUTPUT_DIR / "v122.E_all_regimes_q1.csv")

    # --- F: Month-specific corrections ---
    vf = test_pred_cf.copy()
    vf[hw_mask] += hw_b * 0.8
    vf[co_mask] += co_b * 0.5
    vf[test_month == 1] += m1_cf * 0.3
    vf[test_month == 2] += m2_cf * 0.3
    vf[test_month == 3] += m3_cf * 0.3
    _write(np.clip(vf, 0, CAPACITY_MW), OUTPUT_DIR / "v122.F_month_corr.csv")

    # --- G: Night correction ---
    vg = test_pred_cf.copy()
    vg[hw_mask] += hw_b * 0.8
    vg[co_mask] += co_b * 0.5
    vg[night_mask] += night_bias_cf * 0.3
    vg[~night_mask] += day_bias_cf * 0.3
    _write(np.clip(vg, 0, CAPACITY_MW), OUTPUT_DIR / "v122.G_night.csv")

    # --- H: Simple uniform shifts on CF ---
    for shift in [0.3, 0.5, 0.7, 1.0]:
        vh = np.clip(test_pred_cf + shift, 0, CAPACITY_MW)
        _write(vh, OUTPUT_DIR / f"v122.H_shift_{shift:.1f}.csv")

    # --- I: Best guess combo (strongest corrections that make physical sense) ---
    vi = test_pred_cf.copy()
    vi[hw_mask] += hw_b * 1.0     # full high-wind correction
    vi[co_mask] += co_b * 0.7     # stronger cutout
    vi[mid_mask] += mid_b * 0.3   # mild mid correction
    vi += q1_bias_cf * 0.4        # Q1 bias
    _write(np.clip(vi, 0, CAPACITY_MW), OUTPUT_DIR / "v122.I_best_guess.csv")

    # --- J: Aggressive Q1 bias (the Q1 OOF shows -0.88 CF bias) ---
    vj = test_pred_cf.copy()
    vj[hw_mask] += hw_b * 0.8
    vj[co_mask] += co_b * 0.5
    vj += q1_bias_cf * 1.0  # full Q1 bias correction
    _write(np.clip(vj, 0, CAPACITY_MW), OUTPUT_DIR / "v122.J_full_q1.csv")

    print(f"\n  Top picks for LB:")
    print(f"    v122.D_q1bias_0.5  (Q1 bias targeted, moderate)")
    print(f"    v122.I_best_guess  (hw full + co strong + mid + Q1)")
    print(f"    v122.J_full_q1     (full Q1 bias = {q1_bias_cf:+.3f} MW)")
    print(f"    v122.H_shift_0.5   (simple +0.5 on CF)")
    print(f"    v122.B_hw1.0       (stronger high-wind)")


if __name__ == "__main__":
    main()
