"""Analyze training outliers (impossible data) and NWP calibration signals.

Two questions:
1. Are there impossible training rows (high power / zero wind) that pollute the
   power curve? How many?
2. Does the weather appear to systematically under/over-predict certain regimes
   (a proxy for NWP bias that we can correct)?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.schema import (
    CAPACITY_MW,
    TARGET_COL,
    TIMESTAMP_COL,
    TURBINES_IN_MAINTENANCE_COL,
)

TRAIN = _ROOT / "data" / "raw" / "train_dataset.csv"


def main() -> None:
    df = pd.read_csv(TRAIN)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    print("=" * 70)
    print("OUTLIER ANALYSIS: physically impossible training rows")
    print("=" * 70)

    # Case A: low wind, high power (physically impossible).
    low_wind = df["wind_speed_120m"] < 4.0
    high_power = df[TARGET_COL] > 30.0
    impossible_a = low_wind & high_power
    print(f"\nLow wind (<4 m/s at 120m) + high power (>30 MW): {impossible_a.sum()} rows")
    if impossible_a.sum() > 0:
        sub = df[impossible_a].sort_values(TARGET_COL, ascending=False).head(20)
        print(sub[[TIMESTAMP_COL, TARGET_COL, "wind_speed_120m", "wind_speed_80m", "wind_gusts_10m", TURBINES_IN_MAINTENANCE_COL]].to_string(index=False))

    # Case B: very low wind, moderate power.
    vlow_wind = df["wind_speed_120m"] < 3.0
    mod_power = df[TARGET_COL] > 10.0
    impossible_b = vlow_wind & mod_power
    print(f"\nVery low wind (<3 m/s at 120m) + moderate power (>10 MW): {impossible_b.sum()} rows")

    # Stricter: any level should agree; use REWS proxy.
    # If ALL of ws_10, ws_80, ws_120, ws_180 are low but power is high → really impossible.
    ws_max = df[["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]].max(axis=1)
    really_impossible = (ws_max < 4.0) & (df[TARGET_COL] > 20.0)
    print(f"\nAll heights <4 m/s + power >20 MW: {really_impossible.sum()} rows")
    if really_impossible.sum() > 0:
        sub = df[really_impossible][[TIMESTAMP_COL, TARGET_COL, "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_gusts_10m"]]
        print(sub.to_string(index=False))

    # Gust disagreement: gusts are much higher than wind speed.
    gust_anomaly = (df["wind_gusts_10m"] > df["wind_speed_10m"] * 3) & (df[TARGET_COL] > 20)
    print(f"\nGust anomaly (gust > 3x wind) + power >20 MW: {gust_anomaly.sum()} rows")

    # Case C: high wind but low power (ramp-down / curtailment).
    # Already analyzed.

    print("\n" + "=" * 70)
    print("CURTAILMENT SIGNAL EXPLORATION")
    print("=" * 70)

    # Look for lagged signals of curtailment.
    # "Is the farm underproducing RIGHT NOW vs what wind conditions predict?"
    from sklearn.isotonic import IsotonicRegression
    mask_clean = df[TARGET_COL].notna() & (df[TARGET_COL] > 0.5) & ~impossible_a & ~gust_anomaly
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=CAPACITY_MW)
    ir.fit(df.loc[mask_clean, "wind_speed_120m"], df.loc[mask_clean, TARGET_COL])
    df["p_expected_clean"] = ir.predict(df["wind_speed_120m"])
    df["underprod_clean"] = df["p_expected_clean"] - df[TARGET_COL]
    df["curtail_flag"] = ((df["p_expected_clean"] > 15) & (df[TARGET_COL] < df["p_expected_clean"] * 0.5)).astype(int)

    # Does curtailment have autocorrelation? i.e., does lag-1 curtailment flag predict current?
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"curtail_lag_{lag}"] = df["curtail_flag"].shift(lag)

    # Correlation between current curtailment and recent history.
    print(f"\nAuto-correlation of curtailment flag:")
    for lag in [1, 2, 3, 6, 12, 24]:
        c = df["curtail_flag"].corr(df[f"curtail_lag_{lag}"])
        print(f"  lag {lag:2d}h: {c:.4f}")

    # Distribution of maintenance count right before/during curtailment
    print(f"\nMaintenance count distribution during curtailment:")
    print(df.loc[df['curtail_flag']==1, TURBINES_IN_MAINTENANCE_COL].describe().round(3))
    print(f"\nMaintenance count in normal hours:")
    print(df.loc[df['curtail_flag']==0, TURBINES_IN_MAINTENANCE_COL].describe().round(3))

    # Test: is there a "cut-out" at some wind speed we haven't exploited?
    # Bin by wind_gusts_10m and look at mean power.
    print(f"\n\nMean power by wind_gusts_10m bins:")
    df["gust_bin"] = pd.cut(df["wind_gusts_10m"], [0, 5, 10, 15, 18, 20, 22, 25, 30])
    print(df.groupby("gust_bin", observed=True).agg(
        n=(TARGET_COL, "size"),
        mean_power=(TARGET_COL, "mean"),
        median_power=(TARGET_COL, "median"),
        mean_ws80=("wind_speed_80m", "mean"),
        p_zero_rate=(TARGET_COL, lambda x: (x < 2).mean()),
    ).round(2))

    # Is there a wind_gusts_10m cut-out we're missing?
    print(f"\n\nHigh-wind shutdown analysis:")
    high_gust = df["wind_gusts_10m"] > 15
    print(f"  Rows with gusts > 15 m/s: {high_gust.sum()}")
    print(f"  Of those, rows with power < 5 MW: {(high_gust & (df[TARGET_COL] < 5)).sum()}")
    print(f"  Of those, rows with power > 60 MW: {(high_gust & (df[TARGET_COL] > 60)).sum()}")

    # Rate of power drops > 40 MW in 1 hour
    df["power_diff1h"] = df[TARGET_COL].diff(1)
    big_drops = df["power_diff1h"] < -40
    print(f"\n\nBig power drops (>40 MW in 1h): {big_drops.sum()} events")
    print(f"Mean gust at drop events: {df.loc[big_drops, 'wind_gusts_10m'].mean():.2f}")
    print(f"Mean maintenance at drop events: {df.loc[big_drops, TURBINES_IN_MAINTENANCE_COL].mean():.2f}")

    # NWP bias test: does our current model have systematic bias by wind direction sector?
    # We use residuals from the current LGBM OOF if available.
    oof_path = _ROOT / "data" / "processed" / "oof_lgbm_tuned.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        oof_merged = oof.merge(df[[TIMESTAMP_COL, "wind_direction_120m", "wind_speed_120m", "wind_gusts_10m"]], on=TIMESTAMP_COL)
        oof_merged["dir_deg"] = oof_merged["wind_direction_120m"] * 1000
        oof_merged["residual"] = oof_merged["y_true"] - oof_merged["y_pred"]
        oof_merged["dir_sector"] = pd.cut(oof_merged["dir_deg"], np.arange(0, 361, 45), include_lowest=True)

        print(f"\n\n=== OOF residual bias by direction sector (should be ~0) ===")
        print(oof_merged.groupby(["fold", "dir_sector"], observed=True)["residual"].agg(["mean", "median", "std", "count"]).round(3))

        print(f"\n=== OOF residual bias by wind speed bin ===")
        oof_merged["ws_bin"] = pd.cut(oof_merged["wind_speed_120m"], [0, 3, 5, 7, 10, 14, 18, 25])
        print(oof_merged.groupby(["fold", "ws_bin"], observed=True)["residual"].agg(["mean", "median", "count"]).round(3))


if __name__ == "__main__":
    main()
