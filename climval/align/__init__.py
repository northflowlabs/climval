"""Alignment: put a reference and a candidate on the same points.

:func:`align` runs these steps in order and records each one:

1. ``inputs``: resolve DataArray, Dataset, path or URL (lazily, except CSV).
2. ``coordinates``: find lat, lon and time (CF aware), rename them, sort
   latitude ascending, detect curvilinear grids and station data.
3. ``target_grid``: by default the coarser of the two grids; station
   locations are always the target when one input is station data.
4. ``longitude``: express both inputs in one longitude frame (the target
   grid's convention, or a frame that keeps the region contiguous).
5. ``region``: cut the box (crossing the antimeridian if asked). The target
   grid is cut exactly; the grid that will be regridded keeps a margin of
   one cell and, for bilinear and nearest on a global grid when the box
   reaches the edge of the longitude frame, a wrapped column at each end, so
   interpolation at the edges has neighbours. The wrapped columns are added
   after the time step, so they are aggregated once, not per time step.
   Conservative remapping gets no wrapped columns; xESMF treats a global
   source as periodic instead.
6. ``time``: native frequencies (time bounds, frequency attribute or time
   steps), period, common frequency, aggregation with a coverage threshold
   and period matching across calendars (see :mod:`climval.align.temporal`).
7. ``regrid``: interpolate or remap onto the target grid, or sample the grid
   at station locations. A dask source is rechunked to whole horizontal
   fields first, so interpolation works on a few large chunks.
8. ``region_mask``: apply a regionmask polygon on the target grid.
9. ``units``: convert the candidate into the reference's units.
10. ``missing``: paired mask; a point counts only where both are valid.
11. ``weights``: cell area (given, or computed from cell bounds), cos(latitude)
    on request, or uniform for station data.

Inputs are cast to float64 (lazily for dask arrays), so lazy and eager runs
of float32 data give the same values. The aligned dask arrays are rechunked
to whole horizontal fields and dask's automatic chunk size along time, the
same chunks for both inputs.

:func:`align` reads no data values. It reads coordinates and, when an input
is a Dataset or a file whose time, latitude or longitude bounds are dask
arrays, those bounds variables, all of them in one small ``dask.compute``
call, because the native frequency and the cell areas come from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from climval._chunking import field_chunks
from climval.align.coords import (
    GridInfo,
    is_global_lon,
    lon_convention,
    normalise_coords,
    pad_periodic,
    spacing,
    stations_to_lon_frame,
    to_lon_frame,
)
from climval.align.inputs import InputLike, ResolvedInput, resolve_input
from climval.align.region import (
    Box,
    MaskRegion,
    RegionLike,
    as_region,
    choose_lon_frame,
    needs_wrap,
    subset_box,
)
from climval.align.regrid import METHODS
from climval.align.regrid import regrid as _regrid
from climval.align.temporal import DEFAULT_MIN_COVERAGE, align_time, time_weights
from climval.align.units import harmonise_units
from climval.align.weights import make_weights
from climval.errors import AlignmentError, CurvilinearGridError, NoOverlapError
from climval.provenance import describe_input

__all__ = ["AlignedPair", "align", "Box", "MaskRegion"]

HORIZONTAL = ("lat", "lon")


@dataclass
class AlignedPair:
    """The reference and candidate after alignment, on the same points.

    ``reference`` and ``candidate`` share every coordinate and are NaN
    wherever either one is missing. ``weights`` holds spatial weights on the
    target grid (dims lat and/or lon) or None for uniform weights.
    """

    reference: xr.DataArray
    candidate: xr.DataArray
    weights: xr.DataArray | None
    steps: list[dict[str, Any]]
    inputs: dict[str, dict[str, Any]]
    variable: str
    units: str | None
    spatial_dims: tuple[str, ...]
    has_time: bool
    frequency: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    time_weights: xr.DataArray | None = None
    """Length in days of the period each time step stands for (monthly and yearly data)."""

    @property
    def valid(self) -> xr.DataArray:
        out: xr.DataArray = self.reference.notnull() & self.candidate.notnull()
        return out

    def to_dataset(self) -> xr.Dataset:
        """Both aligned fields (and weights) in one Dataset for inspection."""
        data = {"reference": self.reference, "candidate": self.candidate}
        ds = xr.Dataset(data)
        if self.weights is not None:
            ds["weights"] = self.weights
        return ds


def _split_variable(variable: str | tuple[str, str] | None) -> tuple[str | None, str | None]:
    if variable is None or isinstance(variable, str):
        return variable, variable
    if len(variable) != 2:
        raise AlignmentError("variable must be a name or a (reference, candidate) pair")
    return variable[0], variable[1]


def _cell_size(da: xr.DataArray) -> float:
    sizes = [spacing(da[d].values) for d in HORIZONTAL if d in da.dims]
    sizes = [s for s in sizes if np.isfinite(s)]
    return float(np.prod(sizes)) if sizes else float("nan")


def _target_role(
    target: Any, ref: xr.DataArray, can: xr.DataArray, rg: GridInfo, cg: GridInfo
) -> str:
    if rg.kind == "station" or cg.kind == "station":
        if isinstance(target, (xr.DataArray, xr.Dataset)):
            raise AlignmentError("Station data are compared at the stations; drop target=")
        return "reference" if rg.kind == "station" else "candidate"
    if isinstance(target, (xr.DataArray, xr.Dataset)):
        return "custom"
    if target in ("reference", "candidate"):
        return str(target)
    if target == "coarser":
        if cg.kind != "rectilinear":
            return "reference"
        if rg.kind != "rectilinear":
            return "candidate"
        return "candidate" if _cell_size(can) > _cell_size(ref) else "reference"
    raise AlignmentError(
        "target must be 'reference', 'candidate', 'coarser' or an xarray object "
        f"holding the target grid, got {target!r}"
    )


def _prepare_area(area: xr.DataArray | None, frame: float, box: Box | None) -> xr.DataArray | None:
    if area is None:
        return None
    a, info = normalise_coords(area, "area")
    if info.kind != "rectilinear":
        return None
    a, _ = to_lon_frame(a, frame)
    if box is not None:
        try:
            a = subset_box(a, box, frame, "area")
        except AlignmentError:
            return None
    return a


def align(
    reference: InputLike,
    candidate: InputLike,
    variable: str | tuple[str, str] | None = None,
    *,
    region: RegionLike = None,
    period: tuple[Any, Any] | None = None,
    regrid: str = "bilinear",
    target: str | xr.DataArray | xr.Dataset = "coarser",
    time: str = "intersect",
    freq: str | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    units: str | Mapping[str, str] = "convert",
    weights: str | xr.DataArray | None = "area",
    area: xr.DataArray | None = None,
    open_kwargs: Mapping[str, Any] | None = None,
) -> AlignedPair:
    """Align a candidate with a reference without reading data values; see module docs."""
    if regrid not in METHODS:
        raise AlignmentError(f"regrid must be one of {METHODS}, got {regrid!r}")
    ref_var, can_var = _split_variable(variable)
    r_in: ResolvedInput = resolve_input(reference, ref_var, "reference", open_kwargs)
    c_in: ResolvedInput = resolve_input(candidate, can_var, "candidate", open_kwargs)
    _load_bounds(r_in, c_in)
    inputs = {
        "reference": describe_input(r_in.data, r_in.kind, r_in.location, r_in.dataset_attrs),
        "candidate": describe_input(c_in.data, c_in.kind, c_in.location, c_in.dataset_attrs),
    }
    steps: list[dict[str, Any]] = [
        {
            "step": "inputs",
            "reference_variable": inputs["reference"]["variable"],
            "candidate_variable": inputs["candidate"]["variable"],
        }
    ]

    # 2. coordinates
    ref, rg = normalise_coords(r_in.data, "reference")
    can, cg = normalise_coords(c_in.data, "candidate")
    # Some xarray versions and xESMF drop attributes in reductions and
    # remapping; the variables' own attributes (units) are put back below.
    attrs = {"reference": dict(ref.attrs), "candidate": dict(can.attrs)}
    steps.append({"step": "coordinates", "reference": rg.record(), "candidate": cg.record()})

    role = _target_role(target, ref, can, rg, cg)
    custom: xr.DataArray | None = None
    if role == "custom":
        assert isinstance(target, (xr.DataArray, xr.Dataset))
        custom, tg = normalise_coords(_grid_template(target), "target grid")
        if tg.kind != "rectilinear":
            raise CurvilinearGridError("The target grid must be rectilinear (1D lat and lon)")
        custom = custom.isel({d: 0 for d in custom.dims if d not in HORIZONTAL})

    grids = {"reference": rg, "candidate": cg}
    sources = {"reference": role != "reference", "candidate": role != "candidate"}
    stations = rg.kind == "station" or cg.kind == "station"
    for name, g in grids.items():
        if stations and g.kind == "curvilinear":
            raise CurvilinearGridError(
                "Station data can be compared with rectilinear grids only; "
                f"the {name} grid is curvilinear"
            )
        if g.kind != "curvilinear":
            continue
        if not sources[name]:
            raise CurvilinearGridError(
                f"The {name} grid is curvilinear (2D latitude and longitude) and "
                "cannot be the target grid. Use target='reference' or "
                "target='candidate' so a rectilinear grid is the target, or pass "
                "a rectilinear target grid."
            )
        if regrid != "conservative":
            raise CurvilinearGridError(
                f"The {name} grid is curvilinear (2D latitude and longitude). "
                "climval interpolates only rectilinear grids. Use "
                "regrid='conservative', which remaps curvilinear grids through "
                "xESMF (conda install -c conda-forge xesmf)."
            )
    if rg.kind == "station" and cg.kind == "station":
        sources = {"reference": False, "candidate": False}
    tg_rec: dict[str, Any] = {
        "step": "target_grid",
        "target": role,
        "requested": target if isinstance(target, str) else "user grid",
        "regridded": [k for k, v in sources.items() if v],
    }
    if stations:
        tg_rec["stations"] = "the station locations are the target; a grid is sampled there"
    steps.append(tg_rec)

    # 3. longitude frame
    box_or_mask = as_region(region)
    mask_region = box_or_mask if isinstance(box_or_mask, MaskRegion) else None
    box = mask_region.box() if mask_region is not None else box_or_mask
    assert box is None or isinstance(box, Box)
    tgt_da = custom if custom is not None else (ref if role == "reference" else can)
    tgt_lon: np.ndarray | None = tgt_da["lon"].values if "lon" in tgt_da.dims else None
    frame = choose_lon_frame(box, tgt_lon)
    lon_rec: dict[str, Any] = {"step": "longitude", "frame": [frame, frame + 360.0]}
    arrays = {"reference": ref, "candidate": can}
    for name in arrays:
        da = arrays[name]
        if grids[name].kind == "rectilinear" and "lon" in da.dims:
            before = lon_convention(da["lon"].values)
            da, changed = to_lon_frame(da, frame)
            lon_rec[name] = f"{before} {'re-expressed in frame' if changed else 'unchanged'}"
            arrays[name] = da
        elif grids[name].kind == "station":
            before = lon_convention(da["lon"].values)
            da, changed = stations_to_lon_frame(da, frame)
            lon_rec[name] = f"{before} {'re-expressed in frame' if changed else 'unchanged'}"
            arrays[name] = da
    if custom is not None and "lon" in custom.dims:
        custom, _ = to_lon_frame(custom, frame)
    steps.append(lon_rec)

    # 4. region subset (and periodic padding of sources)
    reg_rec: dict[str, Any] = {"step": "region"}
    reg_rec["region"] = (
        mask_region.to_dict() if mask_region is not None else box.to_dict() if box else None
    )
    if mask_region is not None and stations:
        raise AlignmentError(
            "regionmask regions are not supported with station data; use climval.Box"
        )
    interpolates = regrid in ("bilinear", "nearest") or stations
    # Global sources that need wrapped columns are padded after the time step
    # (and cut in longitude after the padding), so the padding is aggregated
    # once instead of splitting every native time step into narrow chunks.
    pad_after: dict[str, tuple[float, float]] = {}
    for name in arrays:
        da = arrays[name]
        if grids[name].kind == "station":
            if box is not None:
                da = _stations_in_box(da, box, frame, name)
                reg_rec[f"{name}_stations"] = int(da.sizes["station"])
            arrays[name] = da
            continue
        if grids[name].kind != "rectilinear":
            continue
        margin_lat = spacing(da["lat"].values) if sources[name] and "lat" in da.dims else 0.0
        margin_lon = spacing(da["lon"].values) if sources[name] and "lon" in da.dims else 0.0
        margin_lat = 0.0 if np.isnan(margin_lat) else margin_lat
        margin_lon = 0.0 if np.isnan(margin_lon) else margin_lon
        pad = (
            sources[name]
            and interpolates
            and "lon" in da.dims
            and is_global_lon(da["lon"].values)
            and not _same_axis(da["lon"].values, tgt_lon)
            and needs_wrap(da["lon"].values, box, frame, margin_lon)
        )
        if pad:
            pad_after[name] = (margin_lat, margin_lon)
            reg_rec[f"{name}_padded"] = "global longitude: one wrapped column added at each end"
        if box is not None:
            da = subset_box(
                da, box, frame, name, margin_lat=margin_lat, margin_lon=margin_lon, lon=not pad
            )
            if not pad:
                reg_rec[f"{name}_points"] = {
                    d: int(da.sizes[d]) for d in HORIZONTAL if d in da.dims
                }
        arrays[name] = da
    if custom is not None and box is not None:
        custom = subset_box(custom, box, frame, "target grid")
    if box is None:
        reg_rec["action"] = "none: no region"
    steps.append(reg_rec)
    ref, can = arrays["reference"], arrays["candidate"]

    # 5. time
    ref, can, time_rec = align_time(
        ref,
        can,
        freq=freq,
        period=period,
        mode=time,
        min_coverage=min_coverage,
        bounds={
            "reference": _time_bounds(r_in.bounds.get("time"), ref),
            "candidate": _time_bounds(c_in.bounds.get("time"), can),
        },
        declared={"reference": r_in.frequency, "candidate": c_in.frequency},
    )
    target_freq = time_rec.pop("target_freq", None)
    steps.append(time_rec)
    # Aggregated inputs were cast to float64 before aggregating; cast the others.
    ref, can = _as_float64(ref), _as_float64(can)
    for name, (margin_lat, margin_lon) in pad_after.items():
        da = pad_periodic(ref if name == "reference" else can)
        if box is not None:
            da = subset_box(da, box, frame, name, margin_lat=margin_lat, margin_lon=margin_lon)
            reg_rec[f"{name}_points"] = {d: int(da.sizes[d]) for d in HORIZONTAL if d in da.dims}
        if name == "reference":
            ref = da
        else:
            can = da

    # 6. regrid
    if custom is not None:
        tgt_da = custom
    else:
        tgt_da = ref if role == "reference" else can
    tgt_lat: np.ndarray | None
    if stations:
        tgt_lat = np.asarray(tgt_da["lat"].values)
        tgt_lon = np.asarray(tgt_da["lon"].values)
    else:
        tgt_lat = tgt_da["lat"].values if "lat" in tgt_da.dims else None
        tgt_lon = tgt_da["lon"].values if "lon" in tgt_da.dims else None
    for name in ("candidate", "reference"):
        if not sources[name]:
            continue
        src = can if name == "candidate" else ref
        periodic = "lon" in src.dims and is_global_lon(src["lon"].values)
        src, rec = _regrid(
            src,
            tgt_lat,
            tgt_lon,
            regrid,
            name,
            grids[name].kind,
            periodic=periodic,
            points=stations,
        )
        steps.append(rec)
        _check_overlap(src, rec, name)
        if name == "candidate":
            can = src
        else:
            ref = src
    if rg.kind == "station" and cg.kind == "station":
        _check_same_stations(ref, can)

    # 7. regionmask polygon
    if mask_region is not None:
        if tgt_lat is None or tgt_lon is None:
            raise AlignmentError("A regionmask region needs latitude and longitude")
        inside = mask_region.mask(tgt_lat, tgt_lon)
        ref = ref.where(inside)
        can = can.where(inside)
        steps.append(
            {"step": "region_mask", "region": mask_region.name, "cells_inside": int(inside.sum())}
        )

    # Drop auxiliary coordinates so arithmetic aligns on index coordinates only.
    ref = ref.reset_coords(drop=True)
    can = can.reset_coords(drop=True)
    for d in (*HORIZONTAL, "station"):
        if d in ref.dims and d in can.dims:
            if ref.sizes[d] != can.sizes[d]:
                raise AlignmentError(f"After regridding, {d} sizes differ; this is a climval bug")
            owner_da = ref if d in ref.coords else can if d in can.coords else None
            if owner_da is not None:
                values = owner_da[d].values
                ref = ref.assign_coords({d: values})
                can = can.assign_coords({d: values})
    if stations:
        site = {"lat": ("station", tgt_lat), "lon": ("station", tgt_lon)}
        ref = ref.assign_coords(site)
        can = can.assign_coords(site)
    _check_extra_dims(ref, can)
    ref = ref.assign_attrs(attrs["reference"])
    can = can.assign_attrs(attrs["candidate"])

    # 8. units
    ref, can, unit_rec = harmonise_units(ref, can, units)
    steps.append(unit_rec)

    # 9. paired missing-value mask
    ref, can = _rechunk_pair(ref, can)
    valid = ref.notnull() & can.notnull()
    ref = ref.where(valid)
    can = can.where(valid)
    steps.append(
        {
            "step": "missing",
            "action": (
                "paired mask: a point counts only where reference and candidate are both valid"
            ),
        }
    )

    # 10. weights
    spatial = ("station",) if stations else tuple(d for d in HORIZONTAL if d in ref.dims)
    owner = r_in if role == "reference" else c_in if role == "candidate" else None
    area_da = area if area is not None else (owner.area if owner is not None else None)
    area_da = None if stations else _prepare_area(area_da, frame, box)
    grid_bounds = {} if owner is None or stations else _grid_bounds(owner.bounds)
    w, w_rec = make_weights(weights, ref, area_da, grid_bounds, spatial)
    tw = time_weights(ref, target_freq) if "time" in ref.dims else None
    if tw is not None:
        w_rec["time"] = (
            "each time step weighted by the length of its period in days "
            f"({time_rec.get('frequency')}, reference calendar)"
        )
    steps.append(w_rec)
    return AlignedPair(
        reference=ref,
        candidate=can,
        weights=w,
        steps=steps,
        inputs=inputs,
        variable=str(ref.name) if ref.name is not None else "variable",
        units=ref.attrs.get("units"),
        spatial_dims=spatial,
        has_time="time" in ref.dims,
        frequency=time_rec.get("frequency"),
        time_weights=tw,
        options={
            "regrid": regrid,
            "target": role,
            "time": time,
            "freq": freq,
            "min_coverage": min_coverage,
            "period": list(period) if period is not None else None,
            "units": units if isinstance(units, str) else dict(units),
            "weights": weights if isinstance(weights, str) or weights is None else "user supplied",
            "region": reg_rec["region"],
        },
    )


def _load_bounds(*inputs: ResolvedInput) -> None:
    """Read every dask-backed bounds variable of the inputs in one ``dask.compute``."""
    lazy = [(inp, axis) for inp in inputs for axis, b in inp.bounds.items() if b.chunks is not None]
    if not lazy:
        return
    import dask

    values = dask.compute(*[inp.bounds[axis] for inp, axis in lazy])  # type: ignore[attr-defined,no-untyped-call]
    for (inp, axis), loaded in zip(lazy, values, strict=True):
        inp.bounds[axis] = loaded


def _as_float64(da: xr.DataArray) -> xr.DataArray:
    """Numeric data as float64 (lazily for dask), so lazy and eager runs agree."""
    if da.dtype.kind in "fiub" and da.dtype != np.float64:
        out: xr.DataArray = da.astype(np.float64)
        return out
    return da


def _rechunk_pair(ref: xr.DataArray, can: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Rechunk both aligned dask arrays to the same chunks.

    Whole horizontal fields (or all stations) per chunk, when one field fits
    in a quarter of dask's ``array.chunk-size``, and dask's automatic size
    along time. Without this a regridded daily input keeps one chunk of a few
    hundred kilobytes per day, and every metric adds tasks per chunk.
    """
    if ref.chunks is None and can.chunks is None:
        return ref, can
    ref = field_chunks(ref if ref.chunks is not None else ref.chunk())
    can = can.chunk({d: c for d, c in ref.chunksizes.items() if d in can.dims})
    return ref, can


def _check_overlap(src: xr.DataArray, rec: dict[str, Any], role: str) -> None:
    """Raise when no target point lies on the regridded source along an axis."""
    outside = rec.get("target_points_outside_source", {})
    gaps = rec.get("target_points_in_source_gaps", {})
    for d, n_out in outside.items():
        if d in src.dims and src.sizes[d] and max(n_out, gaps.get(d, 0)) >= src.sizes[d]:
            raise NoOverlapError(
                f"The {role} grid does not overlap the target grid along {d}: every target "
                f"{d} lies outside it. Check the region and the extents of the inputs."
            )
    if rec.get("no_overlap"):
        raise NoOverlapError(
            f"The {role} grid does not overlap the target grid: {rec['no_overlap']}"
        )


def _time_bounds(bounds: xr.DataArray | None, da: xr.DataArray) -> xr.DataArray | None:
    """Time bounds renamed to ``time`` and put in the order of ``da``'s time steps."""
    if bounds is None or "time" not in da.dims:
        return None
    b = bounds.rename({bounds.dims[0]: "time"}) if bounds.dims[0] != "time" else bounds
    try:
        out: xr.DataArray = b.sel(time=da["time"].values)
    except (KeyError, ValueError, TypeError):
        return None
    return out


def _grid_bounds(bounds: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    out: dict[str, xr.DataArray] = {}
    for axis in HORIZONTAL:
        b = bounds.get(axis)
        if b is not None:
            out[axis] = b.rename({b.dims[0]: axis}) if b.dims[0] != axis else b
    return out


def _stations_in_box(da: xr.DataArray, box: Box, frame: float, role: str) -> xr.DataArray:
    from climval.errors import NoOverlapError

    lat = np.asarray(da["lat"].values, dtype=float)
    keep = (lat >= box.lat[0] - 1e-9) & (lat <= box.lat[1] + 1e-9)
    if box.subsets_lon:
        lon = np.asarray(da["lon"].values, dtype=float)
        west = frame + np.mod(box.west - frame, 360.0)
        keep &= (lon >= west - 1e-9) & (lon <= west + box.lon_width + 1e-9)
    if not keep.any():
        raise NoOverlapError(f"The region {box.to_dict()} contains no station of the {role}")
    out: xr.DataArray = da.isel(station=np.nonzero(keep)[0])
    return out


def _check_same_stations(ref: xr.DataArray, can: xr.DataArray) -> None:
    for c in ("lat", "lon"):
        a, b = np.asarray(ref[c].values, float), np.asarray(can[c].values, float)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-6):
            raise AlignmentError(
                "Both inputs are station data but their stations differ; select the "
                "same stations, in the same order, in both"
            )


def _grid_template(obj: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """A DataArray carrying the horizontal grid of a user-supplied target."""
    if isinstance(obj, xr.DataArray):
        return obj
    if obj.data_vars:
        return obj[next(iter(obj.data_vars))]
    dims = [str(d) for d in obj.dims]
    return xr.DataArray(np.zeros([obj.sizes[d] for d in dims]), dims=dims, coords=obj.coords)


def _check_extra_dims(ref: xr.DataArray, can: xr.DataArray) -> None:
    core = set(HORIZONTAL) | {"time", "station"}
    for d in set(ref.dims) & set(can.dims):
        if d in core:
            continue
        a = ref[d].values if d in ref.coords else np.arange(ref.sizes[d])
        b = can[d].values if d in can.coords else np.arange(can.sizes[d])
        if a.shape != b.shape or not np.array_equal(a, b):
            raise AlignmentError(
                f"Dimension {d!r} differs between reference and candidate; "
                "select matching values first (for example .sel(plev=85000))"
            )


def _same_axis(a: np.ndarray, b: np.ndarray | None) -> bool:
    """True when two coordinate axes hold the same values (no regridding needed)."""
    if b is None or np.size(a) != np.size(b):
        return False
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-9))
