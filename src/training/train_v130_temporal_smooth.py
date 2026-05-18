"""V130: Temporal smoothing of predictions.

Key insight: v97b makes INDEPENDENT predictions per hour. But real power
output has temporal persistence (wind fields evolve smoothly). The model's
prediction sequence is noisier than reality because it doesn't "know" what
it predicted for adjacent hours.

Error autocorrelation = 0.71 at lag-1h → errors persist in time.
This means: if the model over-predicts at t, it likely over-predicts at t+1.

Approach: apply temporal smoothing to the prediction sequence.
- EMA (exponential moving average) with various spans
- Rolling mean/median
- Savitzky-Golay filter (polynomial smoothing preserving peaks)

The logic: real power transitions more smoothly than the model predicts.
Over-smoothing hurts (loses genuine variation), but mild smoothing removes
prediction noise.

Additionally: train a simple 1D-Conv model on the weather SEQUENCE to
produce a temporally-aware prediction, then blend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae

V97B_OOF_PATH = _ROOT / "data" / "processed" / "v97b_oof.parquet"
V97B_BEST_PATH = _ROOT / "submissions" / "archive" / "v123.B_hw0.7_q0.7.csv"
V128_BEST_PATH = _ROOT / "submissions" / "archive" / "v128.A_q55_07_v97b.csv"
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TARGET_COL_NAME = "Выработка. Результирующий расчет"


def _write(preds: np.ndarray, path: Path, label: str = "") -> None:
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TARGET_COL_NAME: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<55s} mean={preds.mean():.2f}  {label}")


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (forward pass on time-sorted data)."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _bidirectional_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Bidirectional EMA — averages forward and backward pass to avoid lag."""
    fwd = _ema(series, span)
    bwd = _ema(series[::-1], span)[::-1]
    return (fwd + bwd) / 2.0


def _rolling_median(series: np.ndarray, window: int) -> np.ndarray:
    """Rolling median with reflection padding."""
    pad = window // 2
    padded = np.concatenate([series[pad:0:-1], series, series[-2:-pad-2:-1]])
    result = np.array([np.median(padded[i:i+window]) for i in range(len(series))])
    return result


def main() -> None:
    print("=" * 72)
    print("V130: Temporal Smoothing of Predictions")
    print("=" * 72)

    # --- OOF evaluation first ---
    print("\n[1/2] OOF evaluation of smoothing approaches...")
    oof = pd.read_parquet(V97B_OOF_PATH)
    oof["ts"] = pd.to_datetime(oof["ts"])
    oof["pred_cf"] = oof["pred_cf_mw"]

    # Apply corrections to OOF (same recipe as v123.B)
    hw_b = -2.056
    co_b = -12.976
    q1_b = -0.828
    oof["pred_corr"] = oof["pred_cf"].copy()
    hw_mask_oof = (oof["ws_120"] >= 12) & (oof["ws_120"] < 18)
    co_mask_oof = oof["ws_120"] >= 18
    oof.loc[hw_mask_oof, "pred_corr"] += hw_b * 0.7
    oof.loc[co_mask_oof, "pred_corr"] += co_b * 0.5
    oof["pred_corr"] += q1_b * 0.7
    oof["pred_corr"] = oof["pred_corr"].clip(0, CAPACITY_MW)

    # Evaluate on Fold 5 (time-sorted)
    f5 = oof[oof["fold"] == 5].sort_values("ts").reset_index(drop=True)
    target_f5 = f5["target_mw"].to_numpy()
    pred_f5 = f5["pred_corr"].to_numpy()

    base_nmae = float(np.mean(np.abs(target_f5 - pred_f5)) / CAPACITY_MW * 100)
    print(f"  Baseline (corrected, no smoothing): {base_nmae:.4f}%")

    # Test various smoothing methods
    print("\n  Smoothing on corrected predictions (Fold 5):")
    results = {"none": base_nmae}

    # EMA
    for span in [2, 3, 4, 5, 6, 8]:
        smoothed = _bidirectional_ema(pred_f5, span)
        smoothed = np.clip(smoothed, 0, CAPACITY_MW)
        nm = float(np.mean(np.abs(target_f5 - smoothed)) / CAPACITY_MW * 100)
        delta = nm - base_nmae
        print(f"    EMA span={span}: {nm:.4f}% (Δ={delta:+.4f}%)")
        results[f"ema_{span}"] = nm

    # Rolling median
    for w in [3, 5, 7]:
        smoothed = _rolling_median(pred_f5, w)
        smoothed = np.clip(smoothed, 0, CAPACITY_MW)
        nm = float(np.mean(np.abs(target_f5 - smoothed)) / CAPACITY_MW * 100)
        delta = nm - base_nmae
        print(f"    Rolling median w={w}: {nm:.4f}% (Δ={delta:+.4f}%)")
        results[f"rmed_{w}"] = nm

    # Savitzky-Golay filter
    for w, p in [(5, 2), (7, 2), (7, 3), (9, 3), (11, 3)]:
        smoothed = savgol_filter(pred_f5, w, p)
        smoothed = np.clip(smoothed, 0, CAPACITY_MW)
        nm = float(np.mean(np.abs(target_f5 - smoothed)) / CAPACITY_MW * 100)
        delta = nm - base_nmae
        print(f"    Savgol w={w} p={p}: {nm:.4f}% (Δ={delta:+.4f}%)")
        results[f"savgol_{w}_{p}"] = nm

    # Blend: partial smoothing (alpha * smoothed + (1-alpha) * raw)
    print("\n  Partial smoothing blends:")
    best_smooth = _bidirectional_ema(pred_f5, 3)
    for alpha in [0.2, 0.3, 0.5, 0.7]:
        blended = np.clip(alpha * best_smooth + (1 - alpha) * pred_f5, 0, CAPACITY_MW)
        nm = float(np.mean(np.abs(target_f5 - blended)) / CAPACITY_MW * 100)
        delta = nm - base_nmae
        print(f"    α={alpha:.1f} EMA-3: {nm:.4f}% (Δ={delta:+.4f}%)")
        results[f"blend_ema3_{alpha}"] = nm

    # Find best
    best_method = min(results, key=results.get)
    best_nmae = results[best_method]
    print(f"\n  Best method: {best_method} = {best_nmae:.4f}% (Δ={best_nmae-base_nmae:+.4f}%)")

    # --- Apply to test submissions ---
    print("\n[2/2] Generating smoothed test submissions...")

    # Load current best (v128.A = 7.340) — which is in REVERSE chronological order
    v128_df = pd.read_csv(V128_BEST_PATH)
    v128_preds = v128_df[TARGET_COL_NAME].to_numpy(dtype=np.float64)
    # File is in reverse order (Mar 31 23:00 first) — need to sort chronologically for smoothing
    v128_chrono = v128_preds[::-1]  # Now Jan 1 00:00 first

    # Also load v97b corrected
    v97b_df = pd.read_csv(V97B_BEST_PATH)
    v97b_preds = v97b_df[TARGET_COL_NAME].to_numpy(dtype=np.float64)
    v97b_chrono = v97b_preds[::-1]

    # Apply smoothing in chronological order, then reverse back
    print("\n  Applying smoothing to v128.A (current LB best = 7.340):")

    # EMA
    for span in [2, 3, 4]:
        smoothed = _bidirectional_ema(v128_chrono, span)
        smoothed = np.clip(smoothed, 0, CAPACITY_MW)
        result = smoothed[::-1]  # Back to submission order
        _write(result, OUTPUT_DIR / f"v130.A_v128_ema{span}.csv", f"v128 + bidir-EMA span={span}")

    # Savitzky-Golay
    for w, p in [(5, 2), (7, 2), (7, 3)]:
        smoothed = savgol_filter(v128_chrono, w, p)
        smoothed = np.clip(smoothed, 0, CAPACITY_MW)
        result = smoothed[::-1]
        _write(result, OUTPUT_DIR / f"v130.B_v128_savgol{w}_{p}.csv", f"v128 + savgol w={w} p={p}")

    # Partial blend with EMA
    for alpha, span in [(0.3, 3), (0.5, 3), (0.3, 2)]:
        smoothed = _bidirectional_ema(v128_chrono, span)
        blended = np.clip(alpha * smoothed + (1 - alpha) * v128_chrono, 0, CAPACITY_MW)
        result = blended[::-1]
        _write(result, OUTPUT_DIR / f"v130.C_v128_blend{int(alpha*10)}_ema{span}.csv",
               f"v128: {alpha:.0%} EMA-{span} + {1-alpha:.0%} raw")

    # Same for v97b corrected base
    print("\n  Also smoothing v97b-corrected:")
    for span in [2, 3]:
        smoothed = _bidirectional_ema(v97b_chrono, span)
        smoothed = np.clip(smoothed, 0, CAPACITY_MW)
        result = smoothed[::-1]
        _write(result, OUTPUT_DIR / f"v130.D_v97b_ema{span}.csv", f"v97b-corr + bidir-EMA span={span}")

    # Rolling median on v128
    for w in [3, 5]:
        smoothed = _rolling_median(v128_chrono, w)
        smoothed = np.clip(smoothed, 0, CAPACITY_MW)
        result = smoothed[::-1]
        _write(result, OUTPUT_DIR / f"v130.E_v128_rmed{w}.csv", f"v128 + rolling median w={w}")

    print(f"\n  Top picks:")
    print(f"    v130.C_v128_blend3_ema3.csv  (30% smoothing — conservative)")
    print(f"    v130.A_v128_ema3.csv         (full bidirectional EMA-3)")
    print(f"    v130.B_v128_savgol5_2.csv    (polynomial smoothing)")
    if best_method != "none":
        print(f"    OOF suggested: {best_method}")


if __name__ == "__main__":
    main()
