"""Spatial weights for every spatial mean.

``weights="area"`` (default) uses, in this order,

1. a cell area field (``areacella`` style) supplied with ``area=`` or named
   by the reference's ``cell_measures`` attribute, when it lies on the
   target grid;
2. the area of each cell computed from its latitude and longitude bounds,
   ``R**2 * dlon * (sin(lat_north) - sin(lat_south))``. The bounds are the
   target grid's own (``lat_bnds``, ``lon_bnds``) when the Dataset carries
   them, else midpoints between neighbouring centres (clipped at the
   poles). On a uniform grid this is proportional to cos(latitude); on a
   Gaussian or otherwise non-uniform grid it is not, and it accounts for
   uneven longitude spacing as well.

``weights="cos_lat"`` forces cos(latitude), ``weights=None`` gives every
point the same weight, and a DataArray is used as given (it must not be
negative; for dask arrays this is checked when the values are computed).
Station data get uniform weights unless a DataArray on the ``station``
dimension is given.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from climval.errors import AlignmentError

EARTH_RADIUS_M = 6371000.0


def cos_lat(lat: np.ndarray) -> xr.DataArray:
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))
    return xr.DataArray(np.clip(w, 0.0, None), dims=("lat",), coords={"lat": lat})


def _lat_edges(lat: np.ndarray, bounds: xr.DataArray | None) -> tuple[np.ndarray, np.ndarray, str]:
    from climval.align.regrid import bounds_1d

    lat = np.asarray(lat, dtype=float)
    if bounds is not None:
        try:
            b = bounds.sel(lat=lat, method="nearest", tolerance=1e-6).values
            lo, hi = np.minimum(b[:, 0], b[:, 1]), np.maximum(b[:, 0], b[:, 1])
            return np.clip(lo, -90.0, 90.0), np.clip(hi, -90.0, 90.0), "dataset bounds"
        except (KeyError, ValueError, IndexError):
            pass
    if lat.size < 2:
        raise AlignmentError("cell area needs latitude bounds or at least two latitudes")
    edges = bounds_1d(lat, clip=(-90.0, 90.0))
    return edges[:-1], edges[1:], "midpoints"


def _lon_widths(lon: np.ndarray, bounds: xr.DataArray | None) -> tuple[np.ndarray, str]:
    from climval.align.regrid import bounds_1d

    lon = np.asarray(lon, dtype=float)
    if bounds is not None:
        b = np.asarray(bounds.values, dtype=float)
        centres = np.round(np.mod(np.asarray(bounds[bounds.dims[0]].values, float), 360.0), 6)
        # Bounds may wrap around the seam, for example (358.75, 1.25): the width
        # is the difference modulo 360, and a zero difference is a full circle.
        diff = np.mod(b[:, 1] - b[:, 0], 360.0)
        if b.shape[0] > 1:
            # Descending bounds (b1 < b0) give 360 minus the width modulo 360.
            diff = np.minimum(diff, 360.0 - diff)
        else:
            diff = np.where(np.isclose(diff, 0.0), 360.0, diff)
        width = dict(zip(centres.tolist(), diff.tolist(), strict=True))
        keys = np.round(np.mod(lon, 360.0), 6).tolist()
        if all(k in width for k in keys):
            return np.asarray([width[k] for k in keys]), "dataset bounds"
    if lon.size < 2:
        return np.ones(lon.size), "single longitude"
    edges = bounds_1d(lon)
    return np.diff(edges), "midpoints"


def cell_area(
    like: xr.DataArray,
    lat_bounds: xr.DataArray | None = None,
    lon_bounds: xr.DataArray | None = None,
) -> tuple[xr.DataArray, str]:
    """Cell area in m2 on the grid of ``like`` (dims lat and, when present, lon)."""
    lat = np.asarray(like["lat"].values, dtype=float)
    lo, hi, lat_src = _lat_edges(lat, lat_bounds)
    band = np.abs(np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)))
    w = xr.DataArray(band * EARTH_RADIUS_M**2, dims=("lat",), coords={"lat": lat})
    how = f"latitude bounds from {lat_src}"
    if "lon" in like.dims:
        lon = np.asarray(like["lon"].values, dtype=float)
        widths, lon_src = _lon_widths(lon, lon_bounds)
        w = w * xr.DataArray(np.deg2rad(widths), dims=("lon",), coords={"lon": lon})
        how += f", longitude bounds from {lon_src}"
    else:
        w = w * 2.0 * np.pi
    w.name = "cell_area"
    w.attrs["units"] = "m2"
    return w, how


def _covers(source: np.ndarray, target: np.ndarray) -> bool:
    s = np.sort(np.asarray(source, dtype=float))
    t = np.asarray(target, dtype=float)
    if s.size == 0:
        return False
    pos = np.clip(np.searchsorted(s, t), 0, s.size - 1)
    near = np.minimum(np.abs(s[pos] - t), np.abs(s[np.maximum(pos - 1, 0)] - t))
    return bool(np.all(near <= 1e-6))


def _on_grid(area: xr.DataArray, like: xr.DataArray) -> xr.DataArray | None:
    """``area`` on the grid of ``like``, checked on coordinates only (no data read)."""
    dims = [d for d in ("lat", "lon") if d in like.dims]
    if not dims or any(d not in area.dims for d in dims):
        return None
    if not all(_covers(area[d].values, like[d].values) for d in dims):
        return None
    matched = area.reindex({d: like[d].values for d in dims}, method="nearest", tolerance=1e-6)
    # Missing area values (for example masked ocean cells) get zero weight.
    out: xr.DataArray = matched.fillna(0.0).reset_coords(drop=True)
    return out


def _check_block(block: np.ndarray) -> np.ndarray:
    if np.any(np.asarray(block) < 0):
        raise AlignmentError("Weights must not be negative")
    return block


def nonnegative(w: xr.DataArray) -> xr.DataArray:
    """Raise for negative weights: now for NumPy, when computed for dask arrays."""
    if w.chunks is None:
        _check_block(np.asarray(w.values))
        return w
    out: xr.DataArray = w.copy(data=w.data.map_blocks(_check_block, dtype=w.dtype))
    return out


def make_weights(
    spec: str | xr.DataArray | None,
    like: xr.DataArray,
    area: xr.DataArray | None,
    bounds: dict[str, xr.DataArray] | None = None,
    spatial_dims: tuple[str, ...] | None = None,
) -> tuple[xr.DataArray | None, dict[str, Any]]:
    """Spatial weights on the grid of ``like`` (dims lat and/or lon), or None."""
    record: dict[str, Any] = {"step": "weights"}
    bounds = bounds or {}
    stations = spatial_dims == ("station",)
    if spec is None or (isinstance(spec, str) and spec in ("none", "uniform")):
        record["weights"] = "uniform"
        return None, record
    if isinstance(spec, xr.DataArray):
        if any(d in spec.dims for d in ("lat", "lon")) and not stations:
            w = _on_grid(spec, like)
        else:
            w = spec
        if w is None:
            raise AlignmentError("The weights DataArray does not lie on the target grid")
        record["weights"] = "user supplied"
        return nonnegative(w), record
    if spec not in ("area", "cos_lat"):
        raise AlignmentError(
            f"weights must be 'area', 'cos_lat', None or a DataArray, got {spec!r}"
        )
    if stations:
        record["weights"] = "uniform (station data)"
        return None, record
    if spec == "area" and area is not None:
        w = _on_grid(area, like)
        if w is not None:
            record["weights"] = f"cell area ({area.name or 'area'})"
            return nonnegative(w.astype(float)), record
        record["note"] = "cell area supplied but not on the target grid; computed from bounds"
    if "lat" not in like.dims:
        record["weights"] = "uniform (no latitude dimension)"
        return None, record
    if spec == "cos_lat" or (like.sizes["lat"] < 2 and "lat" not in bounds):
        record["weights"] = "cos(latitude)"
        if spec != "cos_lat":
            record["note"] = "one latitude and no latitude bounds: cos(latitude) used"
        return cos_lat(like["lat"].values), record
    w, how = cell_area(like, bounds.get("lat"), bounds.get("lon"))
    record["weights"] = f"cell area from bounds ({how})"
    return w, record
