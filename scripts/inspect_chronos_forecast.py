"""Compare Chronos's forecast to v34.1 hour-by-hour for the first 14 days.

Plots / tables both series so we can see whether Chronos is making
plausible hourly predictions or just predicting a flat mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import TARGET_COL, TIMESTAMP_COL

V341 = _ROOT / "submissions" / "archive" / "v34.1_no_iso.csv"
V60_7D = _ROOT / "submissions" / "archive" / "v60.7d_chronos_stitch.csv"
V60_14D = _ROOT / "submissions" / "archive" / "v60.14d_chronos_stitch.csv"


def main():
    v341 = pd.read_csv(V341)
    v60_7 = pd.read_csv(V60_7D)
    v60_14 = pd.read_csv(V60_14D)

    for df in (v341, v60_7, v60_14):
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    v341 = v341.sort_values(TIMESTAMP_COL).set_index(TIMESTAMP_COL)
    v60_7 = v60_7.sort_values(TIMESTAMP_COL).set_index(TIMESTAMP_COL)
    v60_14 = v60_14.sort_values(TIMESTAMP_COL).set_index(TIMESTAMP_COL)

    # First 14 days of Q1 2026 — that's where Chronos is active in v60.14d.
    start = pd.Timestamp("2026-01-01 00:00")
    end = pd.Timestamp("2026-01-15 00:00")

    cmp = pd.DataFrame({
        "v341": v341.loc[start:end - pd.Timedelta("1h"), TARGET_COL],
        "chronos_14d": v60_14.loc[start:end - pd.Timedelta("1h"), TARGET_COL],
    })

    print("Hour-by-hour comparison, first 7 days of Q1 2026:")
    print("=" * 70)
    print(cmp.iloc[:24].to_string())  # First 24 hours
    print("\n..." )
    print(cmp.iloc[164:172].to_string())  # Around hour 168 (7-day boundary)
    print("\nDay-level summary (first 14 days):")

    daily = cmp.copy()
    daily["day"] = (daily.index - start).total_seconds() // 86400
    daily["day"] = daily["day"].astype(int)
    summary = daily.groupby("day").agg(
        v341_mean=("v341", "mean"),
        v341_std=("v341", "std"),
        chronos_mean=("chronos_14d", "mean"),
        chronos_std=("chronos_14d", "std"),
        diff=("v341", lambda s: (s - daily.loc[s.index, "chronos_14d"]).mean()),
    ).round(2)
    print(summary.to_string())

    print("\nOverall stats first 7 days:")
    seven = cmp.iloc[:168]
    print(f"  v341      : mean={seven['v341'].mean():.2f}  std={seven['v341'].std():.2f}  "
          f"range=[{seven['v341'].min():.2f}, {seven['v341'].max():.2f}]")
    print(f"  chronos_14: mean={seven['chronos_14d'].mean():.2f}  std={seven['chronos_14d'].std():.2f}  "
          f"range=[{seven['chronos_14d'].min():.2f}, {seven['chronos_14d'].max():.2f}]")
    print(f"  MAE between them: {(seven['v341'] - seven['chronos_14d']).abs().mean():.2f} MW")
    print(f"  Correlation     : {seven['v341'].corr(seven['chronos_14d']):.3f}")

    print("\nOverall stats hours 8-14 days:")
    second = cmp.iloc[168:336]
    print(f"  v341      : mean={second['v341'].mean():.2f}  std={second['v341'].std():.2f}")
    print(f"  chronos_14: mean={second['chronos_14d'].mean():.2f}  std={second['chronos_14d'].std():.2f}")
    print(f"  Correlation     : {second['v341'].corr(second['chronos_14d']):.3f}")


if __name__ == "__main__":
    main()
