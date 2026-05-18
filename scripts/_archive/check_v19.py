import pandas as pd
from pathlib import Path

root = Path("submissions/archive")
for f in ["v19.0_deep_blend.csv", "v19.1_ftt_only.csv", "v19.2_lgbm_only.csv"]:
    df = pd.read_csv(root / f, header=None)
    print(f"{f}: rows={len(df)}, mean={df[0].mean():.2f}, range=[{df[0].min():.3f}, {df[0].max():.3f}]")
