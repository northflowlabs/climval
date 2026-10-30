"""Horizontal regridding onto a target grid.

* ``bilinear``: xarray ``interp`` with linear interpolation along latitude and
  longitude (no extra dependency). Works lazily on dask arrays.
* ``nearest``: xarray ``interp`` with nearest-neighbour selection.
* ``conservative``: first-order conservative remapping through xESMF
  (``conservative_normed``), the only method that accepts curvilinear source
  grids.

Target points outside the source grid become NaN; climval never extrapolates.
When every target point already exists in the source grid, the points are
selected directly and nothing is interpolated.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from climval.align.coords import summarise_axis
from climval.errors import AlignmentError, MissingDependencyError

METHODS = ("bilinear", "nearest", "conservative")


def _coincident(src: np.ndarray, tgt: np.ndarray) -> bool:
    src = np.sort(np.asarray(src, dtype=float))
    tgt = np.asarray(tgt, dtype=float)
    if src.size == 0 or tgt.size == 0:
        return False
    tol = 1e-6 * max(1.0, float(np.nanmax(np.abs(src))))
    pos = np.clip(np.searchsorted(src, tgt), 1, max(src.size - 1, 1))
    left = src[pos - 1]
    right = src[np.minimum(pos, src.size - 1)]
    dist = np.minimum(np.abs(tgt - left), np.abs(tgt - right))
    return bool(np.all(dist <= tol))


def _outside(src: np.ndarray, tgt: np.ndarray) -> int:
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    lo, hi = float(np.min(src)), float(np.max(src))
    return int(np.sum((tgt < lo - 1e-9) | (tgt > hi + 1e-9)))


def _in_gap(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Target points that fall in a hole of the source grid.

    A hole is an interval between neighbouring source points wider than 1.5
    times the median spacing (for example between two pieces of a regional
    grid split by a longitude convention change). Interpolating there would
    bridge the hole, so those targets are masked. Targets that coincide with
    a source point are kept.
    """
    s = np.sort(np.asarray(src, dtype=float))
    t = np.asarray(tgt, dtype=float)
    if s.size < 2:
        return np.zeros(t.shape, dtype=bool)
    step = float(np.median(np.diff(s)))
    pos = np.clip(np.searchsorted(s, t), 1, s.size - 1)
    left, right = s[pos - 1], s[pos]
    tol = 1e-9 * max(1.0, step)
    on_point = (np.abs(t - left) <= tol) | (np.abs(t - right) <= tol)
    return np.asarray(((right - left) > 1.5 * step) & ~on_point)


def bounds_1d(centres: np.ndarray, clip: tuple[float, float] | None = None) -> np.ndarray:
    """Cell edges from cell centres (midpoints, extrapolated at the ends)."""
    c = np.asarray(centres, dtype=float)
    if c.size < 2:
        raise AlignmentError("Cannot derive cell bounds from fewer than two points")
    mid = 0.5 * (c[1:] + c[:-1])
    edges = np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])
    if clip is not None:
        edges = np.clip(edges, clip[0], clip[1])
    return edges


def regrid(
    source: xr.DataArray,
    target_lat: np.ndarray | None,
    target_lon: np.ndarray | None,
    method: str,
    role: str,
    source_kind: str = "rectilinear",
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Put ``source`` on the target latitude and longitude."""
    if method not in METHODS:
        raise AlignmentError(f"Unknown regrid method {method!r}; use one of {METHODS}")
    target: dict[str, np.ndarray] = {}
    if target_lat is not None and ("lat" in source.dims or source_kind == "curvilinear"):
        target["lat"] = np.asarray(target_lat)
    if target_lon is not None and ("lon" in source.dims or source_kind == "curvilinear"):
        target["lon"] = np.asarray(target_lon)
    record: dict[str, Any] = {"step": "regrid", "source": role, "method": method}
    if not target:
        record["action"] = "none: no horizontal dimensions to regrid"
        return source, record

    record["target_grid"] = {d: summarise_axis(v) for d, v in target.items()}

    if source_kind == "curvilinear":
        if method != "conservative":
            raise AlignmentError("curvilinear sources need regrid='conservative'")
        out = _xesmf(source, target, curvilinear=True)
        record["action"] = "conservative remapping with xESMF (curvilinear source)"
        return out, record

    record["source_grid"] = {d: summarise_axis(source[d].values) for d in target}
    outside = {d: _outside(source[d].values, v) for d, v in target.items()}
    record["target_points_outside_source"] = outside

    if all(_coincident(source[d].values, v) for d, v in target.items()):
        out = source.sel({d: v for d, v in target.items()}, method="nearest")
        out = out.assign_coords({d: v for d, v in target.items()})
        record["action"] = "grids coincide: points selected, nothing interpolated"
        return out, record

    if method in ("bilinear", "nearest"):
        interp_method: Literal["linear", "nearest"] = (
            "linear" if method == "bilinear" else "nearest"
        )
        out = source.interp(
            {d: v for d, v in target.items()},
            method=interp_method,
            assume_sorted=True,
            kwargs={"fill_value": np.nan},
        )
        gaps = {d: _in_gap(source[d].values, v) for d, v in target.items()}
        for d, in_gap in gaps.items():
            if in_gap.any():
                keep = xr.DataArray(~in_gap, dims=(d,), coords={d: target[d]})
                out = out.where(keep)
        record["target_points_in_source_gaps"] = {d: int(g.sum()) for d, g in gaps.items()}
        record["action"] = (
            f"{method} interpolation with xarray.interp; target points outside "
            "the source grid or inside a gap of it set to NaN"
        )
        return out, record

    out = _xesmf(source, target, curvilinear=False)
    record["action"] = "conservative remapping with xESMF (conservative_normed)"
    return out, record


def _xesmf(source: xr.DataArray, target: dict[str, np.ndarray], curvilinear: bool) -> xr.DataArray:
    try:
        import xesmf as xe
    except ImportError as exc:
        raise MissingDependencyError(
            "regrid='conservative' needs xESMF, which depends on the ESMF "
            "library. Install it with conda: conda install -c conda-forge xesmf "
            '(pip install "climval[regrid]" works where ESMF is already present)'
        ) from exc
    if "lat" not in target or "lon" not in target:
        raise AlignmentError("conservative regridding needs latitude and longitude")
    tgt = xr.Dataset(
        coords={
            "lat": ("lat", target["lat"]),
            "lon": ("lon", target["lon"]),
            "lat_b": ("lat_b", bounds_1d(target["lat"], clip=(-90.0, 90.0))),
            "lon_b": ("lon_b", bounds_1d(target["lon"])),
        }
    )
    if curvilinear:
        if "lat_b" not in source.coords or "lon_b" not in source.coords:
            raise AlignmentError(
                "Conservative regridding of a curvilinear grid needs cell corners "
                "as coordinates lat_b and lon_b (shape (ny+1, nx+1)). "
                "cf_xarray.bounds_to_vertices can build them from CMIP "
                "vertices or *_bnds variables."
            )
        src: xr.Dataset | xr.DataArray = source
    else:
        src = xr.Dataset(
            coords={
                "lat": ("lat", np.asarray(source["lat"].values)),
                "lon": ("lon", np.asarray(source["lon"].values)),
                "lat_b": ("lat_b", bounds_1d(source["lat"].values, clip=(-90.0, 90.0))),
                "lon_b": ("lon_b", bounds_1d(source["lon"].values)),
            }
        )
    regridder = xe.Regridder(src, tgt, "conservative_normed", unmapped_to_nan=True)
    out = regridder(source, skipna=True, na_thres=1.0)
    if not isinstance(out, xr.DataArray):
        raise AlignmentError("xESMF returned an unexpected object")
    return out
