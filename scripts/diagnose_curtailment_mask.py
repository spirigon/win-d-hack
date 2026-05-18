"""Inspect what the curtailment mask flags.

Runs ``identify_curtailment_rows`` against the training set with the v32
feature pipeline applied. Reports:

  - How many rows total are flagged
  - Distribution by year and quarter
  - Top 20 longest curtailment runs (so we can eyeball whether they look
    plausible)
  - Whether March 11-12 2025 (the diagnostic's worst hours) are caught

This is a *diagnostic*. It writes nothing to disk and trains no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.curtailment_mask import (
    CurtailmentConfig,
    identify_curtailment_rows,
    summarize_curtailment,
)
from src.data.outliers import identify_impossible_rows
from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.features.pipeline import build_features
from src.training.train_v32_era5v2 import _load_train_raw

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"


def main():
    print("Loading training set...")
    df = _load_train_raw(TRAIN_PATH)
    df = build_features(df, sort_by_time=False)
    df["_is_impossible_legacy"] = identify_impossible_rows(df).to_numpy()
    print(f"  Rows: {len(df)}  legacy_impossible: {df['_is_impossible_legacy'].sum()}")

    cfg = CurtailmentConfig(
        min_wind_ms=5.0,
        fraction_threshold=0.4,
        min_expected_mw=15.0,
        min_run_length_hours=3,
    )
    print(f"\nConfig: {cfg}")
    flagged, diag = identify_curtailment_rows(
        df,
        cfg=cfg,
        pre_existing_impossible=df["_is_impossible_legacy"],
    )
    summary = summarize_curtailment(diag)
    print(f"\nSummary: {summary}")

    # By year/quarter.
    df["_curtail"] = flagged.to_numpy()
    df["_year"] = df[TIMESTAMP_COL].dt.year
    df["_quarter"] = df[TIMESTAMP_COL].dt.to_period("Q").astype(str)

    print("\nFlagged by year:")
    by_year = df.groupby("_year").agg(
        total=("_curtail", "size"),
        flagged=("_curtail", "sum"),
    )
    by_year["pct"] = (by_year["flagged"] / by_year["total"] * 100).round(2)
    print(by_year.to_string())

    print("\nFlagged by quarter:")
    by_q = df.groupby("_quarter").agg(
        total=("_curtail", "size"),
        flagged=("_curtail", "sum"),
    )
    by_q["pct"] = (by_q["flagged"] / by_q["total"] * 100).round(2)
    print(by_q.to_string())

    # Top runs.
    print("\nTop 20 longest curtailment runs:")
    diag2 = diag.copy()
    diag2["ts"] = df[TIMESTAMP_COL].to_numpy()
    diag2["run_id"] = (diag2["flagged"] != diag2["flagged"].shift()).cumsum()
    runs = (
        diag2[diag2["flagged"]]
        .groupby("run_id")
        .agg(
            start=("ts", "min"),
            end=("ts", "max"),
            length=("ts", "size"),
            mean_ws=("ws", "mean"),
            mean_target=("target", "mean"),
            mean_expected=("expected", "mean"),
        )
        .sort_values("length", ascending=False)
        .head(20)
    )
    runs["mean_ratio"] = (runs["mean_target"] / runs["mean_expected"]).round(3)
    print(runs.to_string())

    # March 11-12 2025 specifically — the worst hours from the diagnostic.
    print("\nMarch 11-12, 2025 inspection:")
    mar_mask = (df[TIMESTAMP_COL] >= "2025-03-11 00:00") & (df[TIMESTAMP_COL] <= "2025-03-12 23:00")
    mar = pd.DataFrame({
        "ts": df.loc[mar_mask, TIMESTAMP_COL].to_numpy(),
        "ws": diag.loc[mar_mask, "ws"].to_numpy(),
        "target": diag.loc[mar_mask, "target"].to_numpy(),
        "expected": diag.loc[mar_mask, "expected"].to_numpy(),
        "ratio": diag.loc[mar_mask, "ratio"].to_numpy(),
        "flagged": diag.loc[mar_mask, "flagged"].to_numpy(),
    })
    print(mar.to_string(index=False))
    n_caught = mar["flagged"].sum()
    print(f"\n  Caught {n_caught} of {len(mar)} hours in March 11-12.")

    # If we mask out flagged rows, how does that change the model's training set?
    n_total_train = len(df)
    n_legacy = int(df["_is_impossible_legacy"].sum())
    n_curtail = int(df["_curtail"].sum())
    n_overlap = int((df["_is_impossible_legacy"] & df["_curtail"]).sum())
    n_remaining = n_total_train - n_legacy - n_curtail + n_overlap
    print(f"\nTraining set size:")
    print(f"  total              : {n_total_train}")
    print(f"  legacy impossible  : {n_legacy} ({n_legacy/n_total_train*100:.2f}%)")
    print(f"  curtailment_mask   : {n_curtail} ({n_curtail/n_total_train*100:.2f}%)")
    print(f"  overlap            : {n_overlap}")
    print(f"  remaining for train: {n_remaining} ({n_remaining/n_total_train*100:.2f}%)")


if __name__ == "__main__":
    main()
