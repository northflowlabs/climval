# SPDX-License-Identifier: Apache-2.0
"""CF-aware coordinate detection and normalisation.

Latitude, longitude and time are found through cf_xarray when it is
installed, then through CF attributes (``standard_name``, ``units``,
``axis``), then through common names. They are renamed to ``lat``, ``lon``
and ``time`` so every later step can rely on those names. Only index
coordinates are touched; data values are never loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

_NAMES = {
    "lat": ("lat", "latitude", "nav_lat", "lats", "y_lat", "XLAT"),
    "lon": ("lon", "longitude", "nav_lon", "lons", "x_lon", "XLONG"),
    "time": ("time", "valid_time", "t", "date", "datetime"),
}
_STANDARD_NAMES = {"lat": "latitude", "lon": "longitude", "time": "time"}
_UNIT_HINTS = {
    "lat": {"degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN"},
    "lon": {"degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE"},
    "time": set(),
}


@dataclass
class GridInfo:
    """What was found and changed in one input's coordinates."""

    kind: str  # "rectilinear", "curvilinear", "station" or "none"
    original_names: dict[str, str | None]
    changes: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "grid": self.kind,
            "detected": self.original_names,
            "changes": list(self.changes),
        }


def _is_datetime_like(coord: xr.DataArray) -> bool:
    if coord.dtype.kind == "M":
        return True
    if coord.dtype == object and coord.size > 0:
        first = coord.values.ravel()[0]
        return hasattr(first, "calendar")
    return False


def find_coord(da: xr.DataArray, axis: str) -> str | None:
    """Return the name of the coordinate for ``axis`` ("lat", "lon" or "time")."""
    candidates: list[str] = []
    try:
        import cf_xarray  # noqa: F401

        coords = da.cf.coordinates
        candidates += [str(n) for n in coords.get(_STANDARD_NAMES[axis], [])]
        if axis == "time":
            candidates += [str(n) for n in da.cf.axes.get("T", [])]
    except Exception:  # noqa: BLE001  cf_xarray missing or unable to parse
        pass
    for name, coord in da.coords.items():
        attrs = coord.attrs
        if attrs.get("standard_name") == _STANDARD_NAMES[axis]:
            candidates.append(str(name))
        elif attrs.get("units") in _UNIT_HINTS[axis]:
            candidates.append(str(name))
        elif axis == "time" and attrs.get("axis") == "T":
            candidates.append(str(name))
    for name in _NAMES[axis]:
        if name in da.coords:
            candidates.append(name)
    if axis == "time":
        candidates += [str(n) for n, c in da.coords.items() if _is_datetime_like(c)]
    seen: list[str] = []
    for c in candidates:
        if c in da.coords and c not in seen:
            seen.append(c)
    if not seen:
        return None
    # Prefer a coordinate that is also a dimension.
    seen.sort(key=lambda n: n not in da.dims)
    return seen[0]


def normalise_coords(da: xr.DataArray, role: str) -> tuple[xr.DataArray, GridInfo]:
    """Rename coordinates to lat, lon, time and sort latitude ascending."""
    found = {axis: find_coord(da, axis) for axis in ("lat", "lon", "time")}
    info = GridInfo(kind="none", original_names=dict(found))

    rename: dict[str, str] = {}
    for axis, name in found.items():
        if name is not None and name != axis:
            if axis in da.coords or axis in da.dims:
                # Another object already uses the canonical name; drop it so
                # the detected coordinate can take its place.
                if axis in da.coords and axis not in da.dims:
                    da = da.drop_vars(axis)
                else:
                    continue
            rename[name] = axis
    if rename:
        da = da.rename(rename)
        info.changes.append("renamed " + ", ".join(f"{k} to {v}" for k, v in rename.items()))

    if (
        "lat" in da.coords
        and "lon" in da.coords
        and da["lat"].ndim == 1
        and da["lon"].ndim == 1
        and da["lat"].dims == da["lon"].dims
        and da["lat"].dims[0] not in ("lat", "lon", "time")
    ):
        site = str(da["lat"].dims[0])
        if site != "station":
            if "station" in da.dims or "station" in da.coords:
                raise ValueError(f"{role}: cannot use {site!r} as the station dimension")
            da = da.rename({site: "station"})
            info.changes.append(f"dimension {site} renamed to station")
        info.kind = "station"
        if "time" in da.coords and da["time"].ndim == 1 and da["time"].dims[0] != "time":
            da = da.swap_dims({da["time"].dims[0]: "time"})
        unsorted = "time" in da.dims and not da.indexes["time"].is_monotonic_increasing
        if unsorted:
            da = da.sortby("time")
            info.changes.append("time sorted ascending")
        return da, info

    for axis in ("lat", "lon", "time"):
        if axis in da.coords and da[axis].ndim == 1 and da[axis].dims[0] != axis:
            old = da[axis].dims[0]
            da = da.swap_dims({old: axis})
            info.changes.append(f"dimension {old} replaced by coordinate {axis}")

    has_lat = "lat" in da.coords
    has_lon = "lon" in da.coords
    if has_lat and has_lon and (da["lat"].ndim == 2 or da["lon"].ndim == 2):
        info.kind = "curvilinear"
        return da, info
    if (has_lat and "lat" in da.dims) or (has_lon and "lon" in da.dims):
        info.kind = "rectilinear"

    if "lat" in da.dims and da.sizes["lat"] > 1:
        lat = np.asarray(da["lat"].values, dtype=float)
        if np.all(np.diff(lat) < 0):
            da = da.isel(lat=slice(None, None, -1))
            info.changes.append("latitude reversed to ascending order")
        elif not np.all(np.diff(lat) > 0):
            da = da.sortby("lat")
            info.changes.append("latitude sorted ascending")
    if "time" in da.dims and da.sizes["time"] > 1:
        index = da.indexes["time"]
        if not index.is_monotonic_increasing:
            da = da.sortby("time")
            info.changes.append("time sorted ascending")
    return da, info


def spacing(values: np.ndarray) -> float:
    """Median absolute spacing of a 1D coordinate (nan for fewer than 2 points)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float("nan")
    return float(np.median(np.abs(np.diff(v))))


def lon_convention(lon: np.ndarray) -> str:
    return "0..360" if float(np.nanmax(lon)) > 180.0 else "-180..180"


def to_lon_frame(da: xr.DataArray, start: float) -> tuple[xr.DataArray, bool]:
    """Express longitude in [start, start + 360) and sort it ascending.

    Returns the array and whether anything changed. Duplicate longitudes
    produced by the wrap (for example both 0 and 360) are dropped.
    """
    if "lon" not in da.dims:
        return da, False
    lon = np.asarray(da["lon"].values, dtype=float)
    new = start + np.mod(lon - start, 360.0)
    # Snap values that land on start + 360 because of rounding back to start.
    new = np.where(np.isclose(new, start + 360.0), start, new)
    changed = not np.allclose(new, lon)
    if changed:
        da = da.assign_coords(lon=new)
    dup = pd.Index(np.round(new, 9)).duplicated()
    if dup.any():
        da = da.isel(lon=np.nonzero(~dup)[0])
        changed = True
    if da.sizes["lon"] > 1 and not np.all(np.diff(da["lon"].values) > 0):
        da = da.sortby("lon")
        changed = True
    return da, changed


def stations_to_lon_frame(da: xr.DataArray, start: float) -> tuple[xr.DataArray, bool]:
    """Express a station longitude coordinate in [start, start + 360) (order kept)."""
    lon = np.asarray(da["lon"].values, dtype=float)
    new = start + np.mod(lon - start, 360.0)
    new = np.where(np.isclose(new, start + 360.0), start, new)
    changed = not np.allclose(new, lon)
    if changed:
        da = da.assign_coords(lon=("station", new))
    return da, changed


def is_global_lon(lon: np.ndarray) -> bool:
    """True when a sorted 1D longitude axis wraps around the whole globe."""
    v = np.asarray(lon, dtype=float)
    if v.size < 3:
        return False
    d = spacing(v)
    if float(np.max(np.diff(v))) > 1.5 * d:
        return False  # a hole: regional pieces, not a global ring
    span = float(v[-1] - v[0]) + d
    return abs(span - 360.0) < 0.5 * d


def pad_periodic(da: xr.DataArray) -> xr.DataArray:
    """Add one wrapped column at each end of a global longitude axis.

    This lets interpolation work across the seam without extrapolating.
    """
    first = da.isel(lon=[0])
    last = da.isel(lon=[-1])
    first = first.assign_coords(lon=first["lon"].values + 360.0)
    last = last.assign_coords(lon=last["lon"].values - 360.0)
    out: xr.DataArray = xr.concat([last, da, first], dim="lon")
    return out


def summarise_axis(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "min": float(np.min(v)) if v.size else None,
        "max": float(np.max(v)) if v.size else None,
        "spacing": spacing(v) if v.size > 1 else None,
    }
