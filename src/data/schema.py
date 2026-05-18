"""Column-name constants and capacity.

The organizer CSVs use Cyrillic column names for the target and for the
"turbines in maintenance" counter. Collect every literal here so callers
never have to paste Cyrillic in model / inference code.
"""

from __future__ import annotations

# Installed capacity (MW). Used for the official normalized MAE metric and
# for clipping final predictions.
CAPACITY_MW: float = 90.09

# Column literals as they appear in the provided CSV files.
TIMESTAMP_COL: str = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET_COL: str = "\u0412\u044b\u0440\u0430\u0431\u043e\u0442\u043a\u0430. \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0440\u0430\u0441\u0447\u0435\u0442"
TURBINES_IN_MAINTENANCE_COL: str = "\u041a\u043e\u043b-\u0432\u043e_\u0412\u042d\u0423_\u0432_\u0440\u0435\u043c\u043e\u043d\u0442\u0435"

# Farm static metadata (from the technical brief).
TOTAL_TURBINES: int = 26
HUB_HEIGHT_M: float = 80.0
LAT_DEG: float = 46.8268455973
LON_DEG: float = 38.7179393185

# Raw weather columns (multi-level).
WS_COLS: tuple[str, ...] = (
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
)
WD_COLS: tuple[str, ...] = (
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
)
GUST_COL: str = "wind_gusts_10m"
TEMP_COLS: tuple[str, ...] = ("temperature_80m", "temperature_120m")
PRESSURE_COL: str = "pressure_msl"
PRECIP_COLS: tuple[str, ...] = ("rain", "showers", "snowfall")
CLOUD_COL: str = "cloud_cover_low"

# Columns that must never enter the feature matrix.
LEAKY_COLS: tuple[str, ...] = (TARGET_COL,)
