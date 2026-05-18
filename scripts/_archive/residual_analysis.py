"""Where does the tuned model fail? Diagnose by regime."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

oof = pd.read_parquet(_ROOT / "data" / "processed" / "oof_lgbm_tuned.parquet")
train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["METEOFORECASTHOUR_OPENM_Datetime"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
# Merge weather info into OOF for regime analysis.
merged = oof.merge(
    train[[
        "METEOFORECASTHOUR_OPENM_Datetime",
        "wind_speed_80m",
        "wind_direction_80m",
        "wind_gusts_10m",
    ]],
    on="METEOFORECASTHOUR_OPENM_Datetime",
    how="left",
)
merged["abs_err"] = (merged["y_true"] - merged["y_pred"]).abs()
merged["signed_err"] = merged["y_pred"] - merged["y_true"]

CAPACITY = 90.09

print("=== Per-fold nMAE ===")
per_fold = merged.groupby("fold")["abs_err"].mean() / CAPACITY * 100
print(per_fold.round(4))

print("\n=== Per-wind-speed regime (Fold-5 only) ===")
f5 = merged[merged["fold"] == "fold5_2025Q1"].copy()
f5["ws_bin"] = pd.cut(f5["wind_speed_80m"], [0, 3, 5, 7, 10, 14, 25])
print(f5.groupby("ws_bin", observed=True).agg(
    n=("abs_err", "size"),
    mean_y_true=("y_true", "mean"),
    mean_y_pred=("y_pred", "mean"),
    mean_abs_err=("abs_err", "mean"),
    bias=("signed_err", "mean"),
    rmse=("signed_err", lambda x: np.sqrt(np.mean(x**2))),
).round(3))

print("\n=== Per-month regime (Fold-5 Q1-2025) ===")
f5["month"] = pd.to_datetime(f5["METEOFORECASTHOUR_OPENM_Datetime"]).dt.month
print(f5.groupby("month").agg(
    n=("abs_err", "size"),
    mean_abs_err=("abs_err", "mean"),
    bias=("signed_err", "mean"),
).round(3))

print("\n=== Per-direction sector (Fold-5, 8 bins) ===")
f5["dir_deg"] = f5["wind_direction_80m"] * 1000
f5["dir_bin"] = pd.cut(f5["dir_deg"], np.arange(0, 361, 45), include_lowest=True)
print(f5.groupby("dir_bin", observed=True).agg(
    n=("abs_err", "size"),
    mean_abs_err=("abs_err", "mean"),
    bias=("signed_err", "mean"),
).round(3))

print("\n=== Worst single-hour errors (Fold-5) ===")
worst = f5.nlargest(10, "abs_err")[[
    "METEOFORECASTHOUR_OPENM_Datetime", "y_true", "y_pred", "abs_err", "wind_speed_80m", "wind_gusts_10m"
]]
print(worst.to_string(index=False))
