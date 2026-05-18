"""Physics-informed features per IEC 61400-12-1 methodology.

Derived from research guide (Wind Power Forecasting ML Research):
- REWS: Rotor-Equivalent Wind Speed integrating across rotor swept area.
- v_eff: Density-corrected effective wind speed at hub.
- Hellmann shear exponent: atmospheric stability proxy.
- Wind power density.
- Isotonic regression power curve (smoother than binned).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.schema import CAPACITY_MW, TARGET_COL

RHO_STD = 1.225  # standard air density kg/m^3
R_DRY = 287.05  # specific gas constant for dry air J/(kg·K)


def compute_air_density(pressure_hpa: pd.Series, temperature_c: pd.Series) -> pd.Series:
    """Ideal gas air density from pressure (hPa) and temperature (°C)."""
    pa = pressure_hpa * 100.0
    tk = temperature_c + 273.15
    return pa / (R_DRY * tk)


def compute_rews(
    ws_80: pd.Series,
    ws_120: pd.Series,
    ws_180: pd.Series,
) -> pd.Series:
    """Rotor-Equivalent Wind Speed.

    Integrates wind speed across rotor swept area using cubic average.
    REWS = ((v_80^3 + v_120^3 + v_180^3) / 3)^(1/3)
    """
    cube_avg = (ws_80**3 + ws_120**3 + ws_180**3) / 3.0
    return cube_avg.clip(lower=0) ** (1.0 / 3.0)


def compute_v_eff(ws_hub: pd.Series, air_density: pd.Series) -> pd.Series:
    """Density-corrected effective wind speed at hub height.

    v_eff = v_hub × (ρ/ρ_0)^(1/3)
    """
    return ws_hub * (air_density / RHO_STD) ** (1.0 / 3.0)


def compute_hellmann_alpha(
    ws_10: pd.Series,
    ws_120: pd.Series,
    h_low: float = 10.0,
    h_high: float = 120.0,
) -> pd.Series:
    """Hellmann shear exponent.

    α = ln(v_high / v_low) / ln(h_high / h_low)
    Typical range: 0.1 (unstable) - 0.3 (stable). Mean ~0.23 for open terrain.
    """
    ratio = (ws_120.clip(lower=0.1) / ws_10.clip(lower=0.1)).clip(lower=0.1, upper=10.0)
    alpha = np.log(ratio) / np.log(h_high / h_low)
    return alpha.clip(lower=-0.3, upper=0.7)  # physical bounds


def compute_wpd(ws: pd.Series, density: pd.Series) -> pd.Series:
    """Wind Power Density = 0.5 × ρ × v^3 (W/m^2)."""
    return 0.5 * density * ws**3


class IsotonicPowerCurve:
    """Isotonic regression power curve on v_eff.

    Fits on training data; at inference returns the theoretical power given v_eff.
    Smoother and more physically consistent than binned median.
    """

    def __init__(self) -> None:
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=CAPACITY_MW)
        self._fitted = False

    def fit(self, v_eff: pd.Series, power: pd.Series) -> "IsotonicPowerCurve":
        mask = v_eff.notna() & power.notna() & (power >= 0)
        self.ir.fit(v_eff[mask].to_numpy(), power[mask].to_numpy())
        self._fitted = True
        return self

    def predict(self, v_eff: pd.Series | np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicPowerCurve is not fitted.")
        v = np.asarray(v_eff)
        return self.ir.predict(v)


class SectorIsotonicPowerCurve:
    """Per-sector isotonic regression power curve on v_eff.

    Fits one curve per direction sector — captures direction-dependent
    wake effects and terrain exposure.
    """

    def __init__(self, n_sectors: int = 8) -> None:
        self.n_sectors = n_sectors
        self.curves: list[IsotonicPowerCurve] = []
        self.global_curve: IsotonicPowerCurve = IsotonicPowerCurve()
        self._fitted = False

    def fit(
        self,
        v_eff: pd.Series,
        dir_deg: pd.Series,
        power: pd.Series,
    ) -> "SectorIsotonicPowerCurve":
        # Global fallback.
        self.global_curve.fit(v_eff, power)

        # Per-sector.
        sector_width = 360.0 / self.n_sectors
        sector = (dir_deg // sector_width).astype(int) % self.n_sectors

        self.curves = []
        for s in range(self.n_sectors):
            mask = sector == s
            curve = IsotonicPowerCurve()
            if mask.sum() > 50:
                curve.fit(v_eff[mask], power[mask])
            else:
                # Fall back to global curve for sparse sectors.
                curve = self.global_curve
            self.curves.append(curve)

        self._fitted = True
        return self

    def predict(self, v_eff: np.ndarray, dir_deg: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        sector_width = 360.0 / self.n_sectors
        sector = (dir_deg // sector_width).astype(int) % self.n_sectors
        preds = np.zeros_like(v_eff, dtype=float)
        for s in range(self.n_sectors):
            mask = sector == s
            if mask.any():
                preds[mask] = self.curves[s].predict(v_eff[mask])
        return np.clip(preds, 0.0, CAPACITY_MW)


def fit_isotonic_power_curve(df: pd.DataFrame, v_eff_col: str = "v_eff") -> IsotonicPowerCurve:
    """Convenience: fit isotonic power curve from a dataframe with target."""
    mask = df[TARGET_COL].notna() & (df[TARGET_COL] >= 0)
    return IsotonicPowerCurve().fit(df[mask][v_eff_col], df[mask][TARGET_COL])


def fit_sector_isotonic(
    df: pd.DataFrame,
    v_eff_col: str = "v_eff",
    dir_col: str = "wind_direction_120m",
    n_sectors: int = 8,
) -> SectorIsotonicPowerCurve:
    """Fit sector-specific isotonic power curves."""
    mask = df[TARGET_COL].notna() & (df[TARGET_COL] >= 0)
    data = df[mask]
    dir_deg = data[dir_col] * 1000.0
    return SectorIsotonicPowerCurve(n_sectors=n_sectors).fit(
        data[v_eff_col], dir_deg, data[TARGET_COL]
    )
