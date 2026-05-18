"""Format the May 18 prediction into the competition format.

Output: single column CSV with header, 24 rows (hours 0-23), values in MW.
Same format as Q1 submission.
"""
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

INPUT  = _ROOT / "submissions" / "18_05_2026_mlp_heavy_0_8.csv"
OUTPUT = _ROOT / "submissions" / "final" / "may18_2026_predictions.csv"

TARGET_COL = "Выработка. Результирующий расчет"

df = pd.read_csv(INPUT)
df = df.sort_values("hour").reset_index(drop=True)

# Single column output with Cyrillic header
out = pd.DataFrame({TARGET_COL: df["forecast_mw"].to_numpy()})
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False, float_format="%.6f")

print(f"May 18 submission: {OUTPUT}")
print(f"  Rows: {len(out)}")
print(f"  Mean: {out[TARGET_COL].mean():.3f} MW")
print(f"  Range: [{out[TARGET_COL].min():.3f}, {out[TARGET_COL].max():.3f}]")
print(f"\nFirst 9 hours (scored):")
for i, row in out.head(9).iterrows():
    print(f"  h{i}: {row[TARGET_COL]:.3f} MW")
