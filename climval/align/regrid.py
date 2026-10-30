# SPDX-License-Identifier: Apache-2.0
"""Horizontal regridding onto a target grid.

* ``bilinear``: xarray ``interp`` with linear interpolation along latitude and
  longitude (no extra dependency). Works lazily on dask arrays.
* ``nearest``: xarray ``interp`` with nearest-neighbour selection.
* ``conservative``: first-order conservative remapping through xESMF
  (``conservative_normed``), the only method that accepts curvilinear source
  grids and the only one that averages a fine source over coarse target
  cells. Global sources are remapped with ``periodic=True`` and without the
  wrapped columns the interpolating methods use. Conservative remapping is
  always done by xESMF, also when target and source points coincide.

A source cell reaches half a spacing beyond its centre, so target points
within half a cell beyond the outermost source centre (clipped at the poles)
take the edge value; points farther out become NaN, and climval never
extrapolates beyond the source cells. For bilinear and nearest, when the
target grid has the same spacing as the source and every target point is a
source point, the points are selected directly and nothing is interpolated.

Station data (paired latitude and longitude on a ``station`` dimension) are
compared by sampling the grid at the station locations.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr

from climval.align.coords import spacing, summarise_axis
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


def _equal_spacing(src: np.ndarray, tgt: np.ndarray) -> bool:
    s, t = spacing(np.sort(np.asarray(src, dtype=float))), spacing(np.asarray(tgt, dtype=float))
    if not (np.isfinite(s) and np.isfinite(t)):
        return np.size(tgt) <= 1 or np.size(src) <= 1
    return bool(abs(s - t) <= 1e-6 * max(s, t))


def _clamp(src: np.ndarray, tgt: np.ndarray, axis: str) -> tuple[np.ndarray, int, int]:
    """Targets beyond the outermost source centre but inside its cell, moved onto that centre.

    A source cell extends half a spacing beyond its centre (clipped at the
    poles for latitude), so a target there lies inside the source grid and
    takes the edge value. Returns the clamped targets, how many were moved and
    how many lie beyond the half cell (those become NaN).
    """
    s = np.asarray(src, dtype=float)
    t = np.asarray(tgt, dtype=float).copy()
    if s.size == 0:
        return t, 0, int(t.size)
    lo, hi = float(np.min(s)), float(np.max(s))
    half = 0.5 * spacing(np.sort(s)) if s.size > 1 else 0.0
    edge_lo, edge_hi = lo - half, hi + half
    if axis == "lat":
        edge_lo, edge_hi = max(edge_lo, -90.0), min(edge_hi, 90.0)
    eps = 1e-9 * max(1.0, abs(lo), abs(hi))
    below = (t < lo) & (t >= edge_lo - eps)
    above = (t > hi) & (t <= edge_hi + eps)
    t[below] = lo
    t[above] = hi
    beyond = int(np.sum((t < lo - eps) | (t > hi + eps)))
    return t, int(below.sum() + above.sum()), beyond


def regrid(
    source: xr.DataArray,
    target_lat: np.ndarray | None,
    target_lon: np.ndarray | None,
    method: str,
    role: str,
    source_kind: str = "rectilinear",
    *,
    periodic: bool = False,
    points: bool = False,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Put ``source`` on the target latitude and longitude.

    ``points=True`` samples the source at the paired (lat[i], lon[i])
    locations of station data instead of on a grid; the result has a
    ``station`` dimension. ``periodic`` tells xESMF that the source wraps
    around in longitude.
    """
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

    if points:
        return _sample_points(source, target, method, record)

    record["target_grid"] = {d: summarise_axis(v) for d, v in target.items()}

    if source_kind == "curvilinear":
        if method != "conservative":
            raise AlignmentError("curvilinear sources need regrid='conservative'")
        out = _xesmf(source, target, curvilinear=True, periodic=False)
        record["action"] = "conservative remapping with xESMF (curvilinear source)"
        return out, record

    record["source_grid"] = {d: summarise_axis(source[d].values) for d in target}

    if method == "conservative":
        for d, v in target.items():
            _, _, n_beyond = _clamp(source[d].values, v, d)
            if n_beyond == np.size(v):
                record["no_overlap"] = f"every target {d} lies outside the source cells"
                return source, record
        out = _xesmf(source, target, curvilinear=False, periodic=periodic)
        record["periodic"] = periodic
        record["action"] = "conservative remapping with xESMF (conservative_normed)"
        return out, record

    same = all(
        _coincident(source[d].values, v) and _equal_spacing(source[d].values, v)
        for d, v in target.items()
    )
    identical = all(
        source[d].size == v.size and np.allclose(source[d].values, v, rtol=0, atol=1e-9)
        for d, v in target.items()
    )
    if identical:
        out = source.assign_coords({d: v for d, v in target.items()})
        record["action"] = "grids identical: nothing interpolated"
        return out, record
    if same:
        out = source.sel({d: v for d, v in target.items()}, method="nearest")
        out = out.assign_coords({d: v for d, v in target.items()})
        record["action"] = "grids coincide: points selected, nothing interpolated"
        return out, record

    coarser = [
        d
        for d, v in target.items()
        if spacing(np.asarray(v, dtype=float)) > 1.5 * spacing(np.sort(source[d].values))
    ]
    if coarser:
        record["note"] = (
            f"the target grid is coarser than the source along {', '.join(coarser)}: "
            f"{method} samples the source at the target points and does not average "
            "over target cells; regrid='conservative' gives cell means"
        )
        warnings.warn(
            f"The {role} is regridded onto a coarser grid with {method}, which samples it "
            "at the target points and does not average over the target cells. "
            "regrid='conservative' (xESMF) gives cell means.",
            UserWarning,
            stacklevel=3,
        )

    clamped: dict[str, np.ndarray] = {}
    moved: dict[str, int] = {}
    beyond: dict[str, int] = {}
    for d, v in target.items():
        clamped[d], moved[d], beyond[d] = _clamp(source[d].values, v, d)
    record["target_points_outside_source"] = beyond
    record["target_points_in_edge_half_cell"] = moved

    interp_method: Literal["linear", "nearest"] = "linear" if method == "bilinear" else "nearest"
    out = _whole_fields(source, list(clamped)).interp(
        {d: v for d, v in clamped.items()},
        method=interp_method,
        assume_sorted=True,
        kwargs={"fill_value": np.nan},
    )
    out = out.assign_coords({d: target[d] for d in clamped})
    gaps = {d: _in_gap(source[d].values, clamped[d]) for d in clamped}
    for d, in_gap in gaps.items():
        if in_gap.any():
            keep = xr.DataArray(~in_gap, dims=(d,), coords={d: target[d]})
            out = out.where(keep)
    record["target_points_in_source_gaps"] = {d: int(g.sum()) for d, g in gaps.items()}
    record["action"] = (
        f"{method} interpolation with xarray.interp; target points within half a source "
        "cell beyond the outermost source centre take the edge value; points farther "
        "out, or inside a gap of the source grid, are NaN"
    )
    return out, record


def _whole_fields(source: xr.DataArray, dims: list[str]) -> xr.DataArray:
    """A dask source rechunked to whole fields along ``dims`` and automatic chunks along time.

    Interpolating a source chunked in small horizontal tiles makes dask split
    the output into many tiny chunks (one per tile and time step).
    """
    if source.chunks is None:
        return source
    spec: dict[str, Any] = {d: -1 for d in dims if d in source.dims}
    if "time" in source.dims:
        spec["time"] = "auto"
    out: xr.DataArray = source.chunk(spec)
    return out


def _sample_points(
    source: xr.DataArray, target: dict[str, np.ndarray], method: str, record: dict[str, Any]
) -> tuple[xr.DataArray, dict[str, Any]]:
    if method == "conservative":
        raise AlignmentError(
            "Station data are compared by sampling the grid at the station locations; "
            "use regrid='bilinear' or 'nearest' (conservative remapping needs a target grid)"
        )
    n = len(next(iter(target.values())))
    clamped: dict[str, np.ndarray] = {}
    beyond: dict[str, int] = {}
    for d, v in target.items():
        clamped[d], _, beyond[d] = _clamp(source[d].values, v, d)
    interp_method: Literal["linear", "nearest"] = "linear" if method == "bilinear" else "nearest"
    out = _whole_fields(source, list(clamped)).interp(
        {d: xr.DataArray(v, dims="station") for d, v in clamped.items()},
        method=interp_method,
        assume_sorted=True,
        kwargs={"fill_value": np.nan},
    )
    out = out.drop_vars([d for d in clamped if d in out.coords])
    keep = np.ones(n, dtype=bool)
    for d in clamped:
        keep &= ~_in_gap(source[d].values, clamped[d])
    if not keep.all():
        out = out.where(xr.DataArray(keep, dims="station"))
    record["target_stations"] = n
    record["stations_outside_source"] = beyond
    record["action"] = (
        f"grid sampled at the station locations ({method}); stations outside the "
        "source grid, or inside a gap of it, are NaN"
    )
    return out, record


def _xesmf(
    source: xr.DataArray, target: dict[str, np.ndarray], curvilinear: bool, periodic: bool
) -> xr.DataArray:
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
    kwargs: dict[str, Any] = {"unmapped_to_nan": True}
    if periodic:
        kwargs["periodic"] = True
    regridder = xe.Regridder(src, tgt, "conservative_normed", **kwargs)
    out = regridder(source, skipna=True, na_thres=1.0)
    if not isinstance(out, xr.DataArray):
        raise AlignmentError("xESMF returned an unexpected object")
    return out
