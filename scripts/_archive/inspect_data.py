"""Quick inspection of the raw data provided by the organizer."""
from __future__ import annotations

import pandas as pd

RAW = r"f:\Claude\win_d\data\raw"


def main() -> None:
    train = pd.read_csv(f"{RAW}\\train_dataset.csv")
    valid = pd.read_csv(f"{RAW}\\valid_features.csv")

    print("=== TRAIN ===")
    print("shape:", train.shape)
    print("columns:", list(train.columns))
    print("dtypes:\n", train.dtypes)
    print("head:\n", train.head(3))
    print("tail:\n", train.tail(3))
    ts = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
    print("ts min/max:", ts.min(), "->", ts.max())
    print("n unique ts:", ts.nunique(), "/ rows:", len(train))
    print("null per col:\n", train.isna().sum())
    target = "\u0412\u044b\u0440\u0430\u0431\u043e\u0442\u043a\u0430. \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0440\u0430\u0441\u0447\u0435\u0442"
    print("target describe:\n", train[target].describe())
    print("target<0:", (train[target] < 0).sum(), "target>90.09:", (train[target] > 90.09).sum())

    print("\n=== VALID ===")
    print("shape:", valid.shape)
    print("columns:", list(valid.columns))
    vts = pd.to_datetime(valid["METEOFORECASTHOUR_OPENM_Datetime"])
    print("ts min/max:", vts.min(), "->", vts.max())
    print("n unique ts:", vts.nunique(), "/ rows:", len(valid))
    print("null per col:\n", valid.isna().sum())

    # Check hour continuity
    train_sorted = train.sort_values("METEOFORECASTHOUR_OPENM_Datetime")
    diffs = pd.to_datetime(train_sorted["METEOFORECASTHOUR_OPENM_Datetime"]).diff().dropna()
    print("\nTrain diff value counts:\n", diffs.value_counts().head(5))

    valid_sorted = valid.sort_values("METEOFORECASTHOUR_OPENM_Datetime")
    vdiffs = pd.to_datetime(valid_sorted["METEOFORECASTHOUR_OPENM_Datetime"]).diff().dropna()
    print("Valid diff value counts:\n", vdiffs.value_counts().head(5))


if __name__ == "__main__":
    main()
