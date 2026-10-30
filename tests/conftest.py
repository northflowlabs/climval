"""Small synthetic fields for the tests (kept tiny: the CI machines are small)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

Field = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def smooth(lat: np.ndarray, lon: np.ndarray, month: np.ndarray) -> np.ndarray:
    """A smooth periodic field (K) varying in latitude, longitude and season."""
    la = np.deg2rad(lat)[None, :, None]
    lo = np.deg2rad(lon)[None, None, :]
    season = np.cos(2 * np.pi * (month[:, None, None] - 1) / 12.0)
    return (
        288.0 - 30.0 * np.sin(la) ** 2 + 3.0 * np.cos(lo) * np.cos(la) + 6.0 * np.sin(la) * season
    )


def monthly_times(start: str, periods: int, calendar: str = "standard", day: int = 15) -> object:
    if calendar == "standard":
        return pd.date_range(start, periods=periods, freq="MS") + pd.Timedelta(days=day - 1)
    return xr.date_range(
        start, periods=periods, freq="MS", calendar=calendar, use_cftime=True
    ).shift(day - 1, "D")


def make_field(
    lat: np.ndarray,
    lon: np.ndarray,
    times: object,
    func: Field = smooth,
    units: str = "K",
    name: str = "tas",
) -> xr.DataArray:
    t = xr.DataArray(times, dims="time")
    months = np.asarray(t.dt.month.values)
    data = func(np.asarray(lat, float), np.asarray(lon, float), months)
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": t.values, "lat": lat, "lon": lon},
        name=name,
        attrs={"units": units, "standard_name": "air_temperature"},
    )


@pytest.fixture
def ref_grid() -> xr.DataArray:
    """Reference: 6 degree global grid, -180..180, ascending lat, monthly, standard calendar."""
    return make_field(
        np.arange(-87.0, 90.0, 6.0), np.arange(-177.0, 180.0, 6.0), monthly_times("2000-01-01", 24)
    )
