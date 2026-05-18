"""Compare Q1-2025 (train) vs Q1-2026 (valid) weather distributions."""
import pandas as pd
import numpy as np

train = pd.read_csv(r"f:\Claude\win_d\data\raw\train_dataset.csv")
valid = pd.read_csv(r"f:\Claude\win_d\data\raw\valid_features.csv")

train["ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
valid["ts"] = pd.to_datetime(valid["METEOFORECASTHOUR_OPENM_Datetime"])

# Q1 2025 from train
q1_2025 = train[(train["ts"] >= "2025-01-01") & (train["ts"] <= "2025-03-31")]
q1_2026 = valid  # all of valid is Q1 2026

print(f"Q1-2025: {len(q1_2025)} rows, Q1-2026: {len(q1_2026)} rows")

cols = ["wind_speed_80m", "wind_speed_120m", "wind_gusts_10m", "temperature_80m", "pressure_msl"]
for c in cols:
    s25 = q1_2025[c]
    s26 = q1_2026[c]
    print(f"\n{c}:")
    print(f"  2025: mean={s25.mean():.3f} std={s25.std():.3f} median={s25.median():.3f}")
    print(f"  2026: mean={s26.mean():.3f} std={s26.std():.3f} median={s26.median():.3f}")
    print(f"  Shift: {s26.mean() - s25.mean():.3f}")

# Direction distribution
print("\n\nDirection sector distribution (80m):")
for df, label in [(q1_2025, "2025"), (q1_2026, "2026")]:
    dir_deg = df["wind_direction_80m"] * 1000
    sectors = (dir_deg // 45).astype(int) % 8
    print(f"  {label}: {sectors.value_counts().sort_index().to_dict()}")

# Power distribution in Q1 2025 (our target surrogate)
TARGET = "\u0412\u044b\u0440\u0430\u0431\u043e\u0442\u043a\u0430. \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0440\u0430\u0441\u0447\u0435\u0442"
print(f"\nQ1-2025 power: mean={q1_2025[TARGET].mean():.2f} std={q1_2025[TARGET].std():.2f}")
print(f"  P10={q1_2025[TARGET].quantile(0.1):.2f} P50={q1_2025[TARGET].quantile(0.5):.2f} P90={q1_2025[TARGET].quantile(0.9):.2f}")

# Maintenance
print(f"\nMaintenance Q1-2025: {q1_2025['\u041a\u043e\u043b-\u0432\u043e_\u0412\u042d\u0423_\u0432_\u0440\u0435\u043c\u043e\u043d\u0442\u0435'].value_counts().sort_index().to_dict()}")
print(f"Maintenance Q1-2026: {q1_2026['\u041a\u043e\u043b-\u0432\u043e_\u0412\u042d\u0423_\u0432_\u0440\u0435\u043c\u043e\u043d\u0442\u0435'].value_counts().sort_index().to_dict()}")
