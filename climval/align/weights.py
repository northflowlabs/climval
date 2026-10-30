"""Spatial weights for every spatial mean.

``weights="area"`` uses a cell area field (``areacella`` style) when one is
supplied with ``area=`` or found through the reference's ``cell_measures``
attribute, provided it lies on the target grid. Otherwise it falls back to
cos(latitude), which is proportional to cell area on a regular
latitude-longitude grid. ``weights="cos_lat"`` forces cos(latitude),
``weights=None`` gives every point the same weight, and a DataArray is used
as given.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from climval.errors import AlignmentError


def cos_lat(lat: np.ndarray) -> xr.DataArray:
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))
    return xr.DataArray(np.clip(w, 0.0, None), dims=("lat",), coords={"lat": lat})


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


def make_weights(
    spec: str | xr.DataArray | None,
    like: xr.DataArray,
    area: xr.DataArray | None,
) -> tuple[xr.DataArray | None, dict[str, Any]]:
    """Spatial weights on the grid of ``like`` (dims lat and/or lon), or None."""
    record: dict[str, Any] = {"step": "weights"}
    has_lat = "lat" in like.dims
    if spec is None or (isinstance(spec, str) and spec in ("none", "uniform")):
        record["weights"] = "uniform"
        return None, record
    if isinstance(spec, xr.DataArray):
        w = _on_grid(spec, like) if any(d in spec.dims for d in ("lat", "lon")) else spec
        if w is None:
            raise AlignmentError("The weights DataArray does not lie on the target grid")
        if w.chunks is None and bool((w < 0).any()):
            raise AlignmentError("Weights must not be negative")
        record["weights"] = "user supplied"
        return w, record
    if spec not in ("area", "cos_lat"):
        raise AlignmentError(
            f"weights must be 'area', 'cos_lat', None or a DataArray, got {spec!r}"
        )
    if spec == "area" and area is not None:
        w = _on_grid(area, like)
        if w is not None:
            record["weights"] = f"cell area ({area.name or 'area'})"
            return w.astype(float), record
        record["note"] = "cell area supplied but not on the target grid; cos(latitude) used"
    if not has_lat:
        record["weights"] = "uniform (no latitude dimension)"
        return None, record
    record["weights"] = "cos(latitude)"
    return cos_lat(like["lat"].values), record
