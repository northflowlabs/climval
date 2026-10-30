"""Alignment: put a reference and a candidate on the same points.

:func:`align` runs these steps in order and records each one:

1. ``inputs``: resolve DataArray, Dataset, path or URL lazily.
2. ``coordinates``: find lat, lon and time (CF aware), rename them, sort
   latitude ascending, detect curvilinear grids.
3. ``longitude``: express both inputs in one longitude frame (the target
   grid's convention, or a frame that keeps the region contiguous).
4. ``region``: cut the box (crossing the antimeridian if asked). The target
   grid is cut exactly; the grid that will be regridded keeps a margin of
   one cell and, when global, a wrapped column at each end, so
   interpolation at the edges has neighbours.
5. ``time``: period, common frequency, aggregation and period matching
   across calendars (see :mod:`climval.align.temporal`).
6. ``regrid``: interpolate or remap onto the target grid.
7. ``region_mask``: apply a regionmask polygon on the target grid.
8. ``units``: convert the candidate into the reference's units.
9. ``missing``: paired mask; a point counts only where both are valid.
10. ``weights``: cell area or cos(latitude).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from climval.align.coords import (
    GridInfo,
    is_global_lon,
    lon_convention,
    normalise_coords,
    pad_periodic,
    spacing,
    to_lon_frame,
)
from climval.align.inputs import InputLike, ResolvedInput, resolve_input
from climval.align.region import (
    Box,
    MaskRegion,
    RegionLike,
    as_region,
    choose_lon_frame,
    subset_box,
)
from climval.align.regrid import METHODS
from climval.align.regrid import regrid as _regrid
from climval.align.temporal import align_time
from climval.align.units import harmonise_units
from climval.align.weights import make_weights
from climval.errors import AlignmentError, CurvilinearGridError
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
    target: str | xr.DataArray | xr.Dataset = "reference",
    time: str = "intersect",
    freq: str | None = None,
    units: str | Mapping[str, str] = "convert",
    weights: str | xr.DataArray | None = "area",
    area: xr.DataArray | None = None,
    open_kwargs: Mapping[str, Any] | None = None,
) -> AlignedPair:
    """Align a candidate with a reference. Nothing is computed; see module docs."""
    if regrid not in METHODS:
        raise AlignmentError(f"regrid must be one of {METHODS}, got {regrid!r}")
    ref_var, can_var = _split_variable(variable)
    r_in: ResolvedInput = resolve_input(reference, ref_var, "reference", open_kwargs)
    c_in: ResolvedInput = resolve_input(candidate, can_var, "candidate", open_kwargs)
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
    for name, g in grids.items():
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
    steps.append(
        {"step": "target_grid", "target": role, "regridded": [k for k, v in sources.items() if v]}
    )

    # 3. longitude frame
    box_or_mask = as_region(region)
    mask_region = box_or_mask if isinstance(box_or_mask, MaskRegion) else None
    box = mask_region.box() if mask_region is not None else box_or_mask
    assert box is None or isinstance(box, Box)
    tgt_da = custom if custom is not None else (ref if role == "reference" else can)
    tgt_lon = tgt_da["lon"].values if "lon" in tgt_da.dims else None
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
    if custom is not None and "lon" in custom.dims:
        custom, _ = to_lon_frame(custom, frame)
    steps.append(lon_rec)

    # 4. region subset (and periodic padding of sources)
    reg_rec: dict[str, Any] = {"step": "region"}
    reg_rec["region"] = (
        mask_region.to_dict() if mask_region is not None else box.to_dict() if box else None
    )
    for name in arrays:
        da = arrays[name]
        if grids[name].kind != "rectilinear":
            continue
        if sources[name] and "lon" in da.dims and is_global_lon(da["lon"].values):
            da = pad_periodic(da)
            reg_rec[f"{name}_padded"] = "global longitude: one wrapped column added at each end"
        if box is not None:
            margin_lat = spacing(da["lat"].values) if sources[name] and "lat" in da.dims else 0.0
            margin_lon = spacing(da["lon"].values) if sources[name] and "lon" in da.dims else 0.0
            da = subset_box(
                da,
                box,
                frame,
                name,
                margin_lat=0.0 if np.isnan(margin_lat) else margin_lat,
                margin_lon=0.0 if np.isnan(margin_lon) else margin_lon,
            )
            reg_rec[f"{name}_points"] = {d: int(da.sizes[d]) for d in HORIZONTAL if d in da.dims}
        arrays[name] = da
    if custom is not None and box is not None:
        custom = subset_box(custom, box, frame, "target grid")
    if box is None:
        reg_rec["action"] = "none: no region"
    steps.append(reg_rec)
    ref, can = arrays["reference"], arrays["candidate"]

    # 5. time
    ref, can, time_rec = align_time(ref, can, freq=freq, period=period, mode=time)
    steps.append(time_rec)

    # 6. regrid
    if custom is not None:
        tgt_da = custom
    else:
        tgt_da = ref if role == "reference" else can
    tgt_lat = tgt_da["lat"].values if "lat" in tgt_da.dims else None
    tgt_lon = tgt_da["lon"].values if "lon" in tgt_da.dims else None
    if sources["candidate"]:
        can, rec = _regrid(can, tgt_lat, tgt_lon, regrid, "candidate", cg.kind)
        steps.append(rec)
    if sources["reference"]:
        ref, rec = _regrid(ref, tgt_lat, tgt_lon, regrid, "reference", rg.kind)
        steps.append(rec)

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
    for d in HORIZONTAL:
        if d in ref.dims and d in can.dims:
            if ref.sizes[d] != can.sizes[d]:
                raise AlignmentError(f"After regridding, {d} sizes differ; this is a climval bug")
            can = can.assign_coords({d: ref[d].values})
    _check_extra_dims(ref, can)

    # 8. units
    ref, can, unit_rec = harmonise_units(ref, can, units)
    steps.append(unit_rec)

    # 9. paired missing-value mask
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
    area_da = area if area is not None else (r_in.area if role == "reference" else None)
    area_da = _prepare_area(area_da, frame, box)
    w, w_rec = make_weights(weights, ref, area_da)
    steps.append(w_rec)

    spatial = tuple(d for d in HORIZONTAL if d in ref.dims)
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
        options={
            "regrid": regrid,
            "target": role,
            "time": time,
            "freq": freq,
            "period": list(period) if period is not None else None,
            "units": units if isinstance(units, str) else dict(units),
            "weights": weights if isinstance(weights, str) or weights is None else "user supplied",
            "region": reg_rec["region"],
        },
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
    core = set(HORIZONTAL) | {"time"}
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
