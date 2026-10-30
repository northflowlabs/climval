# SPDX-License-Identifier: Apache-2.0
"""Write the two small synthetic files the gate tests and the Action workflow use.

Run from the repository root: python tests/data/gate/make_data.py

reference.nc: 2 degree grid over Europe, longitude -16..46, monthly means
2000 to 2002 in the standard calendar, K.
candidate.nc: 5 degree grid, longitude 0..360, latitude 20..80 descending,
monthly means 2000 to 2002 in the noleap calendar, degC, with a warm bias of
0.8 K and noise. Both are synthetic and say so in their attributes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).parent


def field(lat: np.ndarray, lon: np.ndarray, months: np.ndarray) -> np.ndarray:
    la = np.deg2rad(lat)[None, :, None]
    lo = np.deg2rad(lon)[None, None, :]
    season = np.cos(2 * np.pi * (months[:, None, None] - 1) / 12.0)
    return 288.0 - 30.0 * np.sin(la) ** 2 + 3.0 * np.cos(lo) * np.cos(la) - 8.0 * season


def main() -> None:
    rng = np.random.default_rng(42)
    attrs = {"title": "synthetic test data for climval check", "synthetic_data": "yes"}
    # NetCDF3 so that SciPy reads it: the core install has no netCDF4 or h5netcdf.
    enc = {"dtype": "float32"}
    fmt = {"format": "NETCDF3_64BIT", "engine": "scipy"}

    lat = np.arange(31.0, 76.0, 2.0)
    lon = np.arange(-15.0, 46.0, 2.0)
    time = pd.date_range("2000-01-01", periods=36, freq="MS") + pd.Timedelta(days=14)
    data = field(lat, lon, np.asarray(time.month)) + rng.normal(0.0, 0.3, (36, lat.size, lon.size))
    ref = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), data, {"units": "K", "long_name": "2 m temperature"})},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs=attrs,
    )
    ref.to_netcdf(HERE / "reference.nc", encoding={"t2m": enc}, **fmt)

    clat = np.arange(80.0, 19.0, -5.0)
    clon = np.arange(2.5, 360.0, 5.0)
    ctime = xr.date_range(
        "2000-01-01", periods=36, freq="MS", calendar="noleap", use_cftime=True
    ).shift(15, "D")
    months = np.asarray([t.month for t in ctime])
    cdata = field(clat, clon, months) + 0.8 - 273.15
    cdata = cdata + rng.normal(0.0, 0.6, cdata.shape)
    can = xr.Dataset(
        {
            "tas": (
                ("time", "lat", "lon"),
                cdata,
                {"units": "degC", "standard_name": "air_temperature"},
            )
        },
        coords={"time": ctime, "lat": clat, "lon": clon},
        attrs={**attrs, "source_id": "SYNTHETIC-MODEL"},
    )
    can.to_netcdf(HERE / "candidate.nc", encoding={"tas": enc}, **fmt)


if __name__ == "__main__":
    main()
