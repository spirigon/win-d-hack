"""Deep-dive analysis of curtailment / maintenance events.

Questions:
1. How much of the error mass is caused by curtailment events?
2. Can we detect them from features we have?
3. Are they temporal (clustered in time)?
4. Is there a relationship with the maintenance counter that we haven't exploited?
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
from src.features.physics import compute_air_density, compute_v_eff

TRAIN = _ROOT / "data" / "raw" / "train_dataset.csv"


def main() -> None:
    df = pd.read_csv(TRAIN)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Air density + v_eff for physical expected power.
    density = compute_air_density(df["pressure_msl"], df["temperature_80m"])
    df["v_eff"] = compute_v_eff(df["wind_speed_120m"], density)

    # Fit a simple KDE power curve on the whole training set.
    from sklearn.isotonic import IsotonicRegression

    mask_fit = df[TARGET_COL].notna() & (df[TARGET_COL] > 0.5)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=CAPACITY_MW)
    ir.fit(df.loc[mask_fit, "v_eff"], df.loc[mask_fit, TARGET_COL])
    df["p_expected"] = ir.predict(df["v_eff"])

    # Residual = expected - actual. Positive = underproduction (curtailment candidate).
    df["underprod"] = df["p_expected"] - df[TARGET_COL]
    df["underprod_norm"] = df["underprod"] / CAPACITY_MW * 100  # %

    total_err = (df["underprod"].abs()).sum()
    print("=" * 70)
    print("CURTAILMENT / MAINTENANCE DETECTION")
    print("=" * 70)

    # Define "curtailment candidate": p_expected > 30 MW AND actual < p_expected * 0.3
    # (i.e., producing less than 30% of what wind conditions suggest)
    curtail_mask = (df["p_expected"] > 20) & (df[TARGET_COL] < df["p_expected"] * 0.4)
    print(f"\nCurtailment candidates (p_exp>20 AND actual<40% of exp):")
    print(f"  Count: {curtail_mask.sum()} / {len(df)} ({curtail_mask.mean()*100:.2f}%)")
    print(f"  Mean underproduction: {df.loc[curtail_mask, 'underprod'].mean():.2f} MW")
    print(f"  Sum of |error| from these rows: {df.loc[curtail_mask, 'underprod'].abs().sum():.1f} MW")
    print(f"  Share of total error mass: {df.loc[curtail_mask, 'underprod'].abs().sum() / total_err * 100:.1f}%")

    # Maintenance count distribution for curtailment events
    print(f"\nMaintenance count during curtailment events:")
    print(df.loc[curtail_mask, TURBINES_IN_MAINTENANCE_COL].value_counts().sort_index())

    print(f"\nMaintenance count in normal hours:")
    print(df.loc[~curtail_mask, TURBINES_IN_MAINTENANCE_COL].value_counts().sort_index())

    # Is curtailment temporally clustered?
    print(f"\nCurtailment clustering:")
    curtail_run = curtail_mask.astype(int)
    # Consecutive hours of curtailment.
    groups = (curtail_run.diff() != 0).cumsum()
    runs = curtail_run.groupby(groups).agg(["sum", "count"])
    curtail_runs = runs[runs["sum"] > 0]["count"]
    if len(curtail_runs) > 0:
        print(f"  Number of curtailment 'events' (contiguous runs): {len(curtail_runs)}")
        print(f"  Mean run length: {curtail_runs.mean():.2f} hours")
        print(f"  Max run length: {curtail_runs.max()} hours")
        print(f"  Distribution: {curtail_runs.describe().to_dict()}")

    # Time-of-day pattern for curtailment?
    df["hour"] = df[TIMESTAMP_COL].dt.hour
    hour_curtail = df.groupby("hour").apply(
        lambda g: curtail_mask[g.index].mean() * 100
    )
    print(f"\nCurtailment rate by hour of day (%):")
    print(hour_curtail.round(2).to_dict())

    # Year-over-year trend
    df["year"] = df[TIMESTAMP_COL].dt.year
    print(f"\nCurtailment rate by year:")
    print(df.groupby("year").apply(lambda g: curtail_mask[g.index].mean() * 100).round(2).to_dict())

    # Is curtailment predictable from any feature?
    print(f"\n\n=== Can we predict curtailment from features? ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit

    features_for_curtail = [
        "wind_speed_80m", "wind_speed_120m", "wind_gusts_10m",
        "pressure_msl", "temperature_80m", TURBINES_IN_MAINTENANCE_COL,
        "wind_direction_120m", "cloud_cover_low", "hour",
    ]
    df["month"] = df[TIMESTAMP_COL].dt.month
    features_for_curtail.append("month")

    X = df[features_for_curtail].values
    y = curtail_mask.astype(int).values

    # Time-series CV to check if curtailment is predictable.
    from sklearn.metrics import roc_auc_score
    tscv = TimeSeriesSplit(n_splits=3)
    aucs = []
    for fold_idx, (tr, va) in enumerate(tscv.split(X)):
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, class_weight="balanced")
        rf.fit(X[tr], y[tr])
        prob = rf.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y[va], prob)
        aucs.append(auc)
        print(f"  Fold {fold_idx+1}: AUC = {auc:.4f}")
    print(f"  Mean AUC: {np.mean(aucs):.4f}")

    if np.mean(aucs) > 0.75:
        print("  -> Curtailment IS reasonably predictable from features.")

    # Feature importance for curtailment prediction
    rf_full = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42, class_weight="balanced")
    rf_full.fit(X, y)
    print(f"\n  Feature importance for curtailment detection:")
    imp = sorted(zip(features_for_curtail, rf_full.feature_importances_), key=lambda x: -x[1])
    for name, score in imp:
        print(f"    {name:40s} {score:.4f}")

    # === Worst hour residuals: are they curtailment? ===
    print(f"\n\n=== Top 30 worst residual hours ===")
    df["abs_err"] = df["underprod"].abs()
    worst = df.nlargest(30, "abs_err")[[
        TIMESTAMP_COL, TARGET_COL, "v_eff", "p_expected",
        TURBINES_IN_MAINTENANCE_COL, "wind_speed_120m", "wind_gusts_10m"
    ]]
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
