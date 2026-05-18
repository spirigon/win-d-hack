"""V60: Chronos-Bolt zero-shot forecast stitched with v34.1 predictions.

Idea (per discussion):

    Chronos-Bolt is a pretrained transformer time-series forecaster. Its
    autoregressive horizon error grows with steps; at 7-14 days ahead
    the recent target history (Dec 2025) is still close enough that the
    transformer's pattern-matching is meaningful. At 90 days (the full
    Q1 2026 window) it would be hallucination.

    Strategy: replace v34.1's predictions for the first ``stitch_hours``
    of Q1 2026 with Chronos's univariate forecast, then keep v34.1 for
    the remainder.

The univariate aspect is on purpose — Chronos doesn't know about NWP. The
diversity comes from it noticing patterns LightGBM can't (e.g. cyclical
behaviour, seasonality). If it's worse than v34.1 in those days, the LB
will tell us; if it's better, we get a free win exclusively on the early
window.

Outputs:
    submissions/archive/v60.{stitch_days}d_chronos_stitch.csv

For each ``stitch_days`` in {7, 14} we write:
  - Pure Chronos for the first ``stitch_days`` × 24 hours
  - v34.1 prediction for the rest
  - Submission file ready to upload

Usage:
    python -m src.training.train_v60_chronos_stitch
    python -m src.training.train_v60_chronos_stitch --stitch-days 7 14
    python -m src.training.train_v60_chronos_stitch --model chronos-bolt-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chronos import ChronosBoltPipeline

from src.data.loaders import load_valid_features
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"

# v34.1 archive submission (used as the "fill" for hours after the stitch window).
V341_PATH = _ROOT / "submissions" / "archive" / "v34.1_no_iso.csv"

# Default Chronos model. Size order: tiny < mini < small < base.
DEFAULT_MODEL = "amazon/chronos-bolt-base"

# Context length: how many hours of history to feed Chronos. Bolt accepts up
# to 2048 tokens; 1 hour ≈ 1 token. We use the FULL training history (~32k
# hours) but only the last 2048 hours go into the prompt — they should be
# the most relevant.
CONTEXT_HOURS = 2048


def _load_target_history() -> pd.Series:
    """Load full chronological target_mw history."""
    df = pd.read_csv(TRAIN_PATH)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    series = df.set_index(TIMESTAMP_COL)[TARGET_COL].astype(float)
    series = series.dropna()  # drop any remaining NaN from the raw CSV
    return series


def _load_v341_test() -> pd.DataFrame:
    """Load the v34.1 final submission and pair predictions with timestamps.

    Returns a DataFrame with columns ``ts`` (datetime64) and ``v341_mw``
    (predicted power, MW).
    """
    sub = pd.read_csv(V341_PATH)
    if TIMESTAMP_COL not in sub.columns:
        raise ValueError(
            f"{V341_PATH} doesn't have {TIMESTAMP_COL} column — "
            f"the archive file format must be the 2-column variant"
        )
    sub[TIMESTAMP_COL] = pd.to_datetime(sub[TIMESTAMP_COL])
    sub = sub.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return pd.DataFrame({
        "ts": sub[TIMESTAMP_COL],
        "v341_mw": sub[TARGET_COL].astype(float),
    })


def _chronos_forecast(
    series: pd.Series,
    *,
    horizon_hours: int,
    model_name: str,
    device: str,
    context_hours: int = CONTEXT_HOURS,
) -> np.ndarray:
    """Zero-shot Chronos-Bolt forecast.

    We feed the most recent ``context_hours`` of history and ask for a
    forecast of length ``horizon_hours``. Returns the median forecast in MW.
    """
    print(f"  Loading model {model_name} on {device}...")
    pipe = ChronosBoltPipeline.from_pretrained(model_name, device_map=device)

    history = series.values[-context_hours:]
    print(f"  History length: {len(history)} hours, "
          f"target stats: mean={history.mean():.2f}, std={history.std():.2f}")

    context = torch.tensor(history, dtype=torch.float32).reshape(1, -1)
    print(f"  Predicting {horizon_hours} hours ahead...")
    with torch.no_grad():
        forecast = pipe.predict(context, prediction_length=horizon_hours)

    # forecast shape: (batch_size=1, num_quantiles=9, prediction_length).
    # Median is the 5th of 9 quantiles, i.e. index 4 (0.5 quantile).
    median = forecast[0, 4, :].cpu().numpy()
    return np.clip(median, 0.0, CAPACITY_MW)


def _make_submission(
    chronos_pred: np.ndarray,
    v341_df: pd.DataFrame,
    *,
    stitch_hours: int,
    blend_alpha: float = 1.0,
    output_path: Path,
):
    """Stitch Chronos and v34.1 predictions, then write submission.

    Parameters
    ----------
    chronos_pred:
        Length ``stitch_hours``, the Chronos median in MW.
    v341_df:
        DataFrame with ``ts`` and ``v341_mw`` columns, sorted by ts ascending.
    stitch_hours:
        Number of hours from the start of Q1 2026 to overwrite with Chronos.
    blend_alpha:
        1.0 = pure Chronos in the stitch window. 0.5 = 50/50 blend with v34.1.
    output_path:
        Where to write the submission CSV.
    """
    # Make sure we know which v34.1 rows align with the chronos hours.
    # v34.1 is sorted ascending; Q1 2026 starts at 2026-01-01 00:00.
    n_total = len(v341_df)
    # Find the row index where the first Q1-2026 hour lives.
    first_q1_idx = int(v341_df["ts"].searchsorted(pd.Timestamp("2026-01-01 00:00")))
    print(f"  v341 rows: {n_total}  first Q1-2026 row index: {first_q1_idx}")

    # Build the stitched MW array.
    final_mw = v341_df["v341_mw"].to_numpy().astype(np.float64).copy()
    end_idx = first_q1_idx + stitch_hours
    if end_idx > n_total:
        raise ValueError(
            f"stitch window [{first_q1_idx}, {end_idx}) extends beyond "
            f"v341_df length {n_total}"
        )
    if blend_alpha == 1.0:
        final_mw[first_q1_idx:end_idx] = chronos_pred
    else:
        original = final_mw[first_q1_idx:end_idx]
        final_mw[first_q1_idx:end_idx] = blend_alpha * chronos_pred + (1 - blend_alpha) * original
    final_mw = np.clip(final_mw, 0.0, CAPACITY_MW)

    # Restore the descending order that ``valid_features.csv`` uses.
    df_valid = load_valid_features(VALID_PATH)
    valid_ts_to_row = {ts: i for i, ts in enumerate(df_valid[TIMESTAMP_COL])}
    po = np.empty(n_total, dtype=np.float64)
    ts_po = np.empty(n_total, dtype="datetime64[ns]")
    for i, ts in enumerate(v341_df["ts"]):
        out_idx = valid_ts_to_row.get(ts)
        if out_idx is None:
            raise ValueError(f"timestamp {ts} not found in valid_features.csv")
        po[out_idx] = final_mw[i]
        ts_po[out_idx] = ts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, output_path, expected_rows=n_total, timestamps=ts_po)
    print(f"  Submission saved: {output_path}")
    print(f"    Stitched window mean: {final_mw[first_q1_idx:end_idx].mean():.2f} MW   "
          f"(Chronos pred mean: {chronos_pred.mean():.2f} MW)")
    print(f"    Tail (post-stitch) mean: {final_mw[end_idx:].mean():.2f} MW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-days", nargs="+", type=int, default=[7, 14],
                    help="how many days at the start of Q1 2026 to overwrite with Chronos")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HuggingFace model id (default: amazon/chronos-bolt-base)")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda (default: cpu)")
    ap.add_argument("--blend", type=float, default=1.0,
                    help="alpha for stitch (1.0=pure chronos, 0.5=50/50 with v341)")
    args = ap.parse_args()

    print("=" * 72)
    print(f"V60: Chronos-Bolt stitch with v34.1   (model={args.model}, "
          f"device={args.device}, blend={args.blend})")
    print("=" * 72)

    # --- 1. Load history + v34.1 baseline ----------------------------
    print("\n[1/3] Loading history...")
    history = _load_target_history()
    print(f"  History: {len(history)} hours from "
          f"{history.index.min()} to {history.index.max()}")
    v341_df = _load_v341_test()
    print(f"  v341 submission: {len(v341_df)} predictions, "
          f"range [{v341_df['v341_mw'].min():.2f}, {v341_df['v341_mw'].max():.2f}] MW")

    # --- 2. Pick max horizon and predict once -----------------------
    max_horizon = max(args.stitch_days) * 24
    print(f"\n[2/3] Forecasting {max_horizon} hours with Chronos...")
    forecast_full = _chronos_forecast(
        history,
        horizon_hours=max_horizon,
        model_name=args.model,
        device=args.device,
    )
    print(f"  Forecast stats: mean={forecast_full.mean():.2f}, "
          f"std={forecast_full.std():.2f}, "
          f"range=[{forecast_full.min():.2f}, {forecast_full.max():.2f}]")

    # --- 3. Build one submission per stitch-days choice ------------
    print("\n[3/3] Building submissions...")
    for d in args.stitch_days:
        h = d * 24
        out_path = _ROOT / "submissions" / "archive" / f"v60.{d}d_chronos_stitch.csv"
        if abs(args.blend - 1.0) > 1e-6:
            out_path = out_path.with_name(
                f"v60.{d}d_chronos_blend{int(args.blend*100)}.csv"
            )
        print(f"\n  -- {d}-day stitch ({h} hours), blend={args.blend} --")
        _make_submission(
            forecast_full[:h],
            v341_df,
            stitch_hours=h,
            blend_alpha=args.blend,
            output_path=out_path,
        )


if __name__ == "__main__":
    main()
