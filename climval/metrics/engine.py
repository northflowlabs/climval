"""Evaluate metrics on an aligned pair, per sample and per group.

A *sample* says which points a number is computed over:

``space_time``
    Every valid point in space and time pooled, each weighted by its
    spatial weight (cell area or cos latitude). One value per group.
``climatology``
    The spatial pattern of the time-mean fields: both inputs are averaged
    over the (paired) time steps of the group first, then compared over
    space with spatial weights. This is the usual Taylor diagram and
    Gleckler et al. (2008) statistic. One value per group.
``space``
    The spatial pattern at each time step, giving a time series of values
    (dimension ``time``). Not grouped.
``time``
    Temporal statistics at each grid point, every time step weighted
    equally, giving maps (dimensions ``lat`` and ``lon``). One map per group.

Groups come from ``by``: "season" (DJF, MAM, JJA, SON), "month" or "year".
The group "all" holds the whole period and is always present.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from climval.errors import AlignmentError
from climval.metrics.stats import BaseMetric

SAMPLES: dict[str, str] = {
    "space_time": "all valid points in space and time pooled, spatially weighted",
    "climatology": "spatial pattern of the time-mean fields, spatially weighted",
    "space": "spatial pattern at each time step (one value per time step)",
    "time": "temporal statistics at each grid point (one map), equal weight per step",
}
SCALAR_SAMPLES = ("space_time", "climatology")
DEFAULT_SAMPLES = ("space_time", "climatology")
GROUPINGS = ("season", "month", "year")
_SEASONS = ("DJF", "MAM", "JJA", "SON")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def make_groups(
    time: xr.DataArray | None, by: str | None, frequency: str | None
) -> list[tuple[str, np.ndarray | None]]:
    groups: list[tuple[str, np.ndarray | None]] = [("all", None)]
    if by is None:
        return groups
    if by not in GROUPINGS:
        raise AlignmentError(f"by must be one of {GROUPINGS} or None, got {by!r}")
    if time is None:
        raise AlignmentError(f"by={by!r} needs a time dimension")
    if by in ("season", "month") and frequency == "yearly":
        raise AlignmentError(f"by={by!r} needs monthly or finer data, the comparison is yearly")
    if by == "season":
        labels = np.asarray(time.dt.season.values).astype(str)
        order: Sequence[str] = _SEASONS
    elif by == "month":
        labels = np.asarray([_MONTHS[m - 1] for m in time.dt.month.values])
        order = _MONTHS
    else:
        labels = np.asarray(time.dt.year.values).astype(str)
        order = sorted(set(labels.tolist()), key=int)
    for name in order:
        idx = np.nonzero(labels == name)[0]
        if idx.size:
            groups.append((name, idx))
    return groups


def _reduce_all(
    metrics: Sequence[BaseMetric],
    r: xr.DataArray,
    c: xr.DataArray,
    w: xr.DataArray,
    valid: xr.DataArray,
    dims: Sequence[str],
) -> xr.Dataset:
    out: dict[str, xr.DataArray] = {}
    for m in metrics:
        try:
            value = m.reduce(r, c, w, list(dims))
        except Exception as exc:
            raise type(exc)(f"metric {m.name!r} failed: {exc}") from exc
        out[m.name] = value.reset_coords(drop=True) if isinstance(value, xr.DataArray) else value
    out["n_points"] = valid.sum(list(dims))
    return xr.Dataset(out)


def evaluate(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    spatial_weights: xr.DataArray | None,
    metrics: Sequence[BaseMetric],
    spatial_dims: Sequence[str],
    has_time: bool,
    samples: Sequence[str] = DEFAULT_SAMPLES,
    by: str | None = None,
    frequency: str | None = None,
) -> dict[str, xr.Dataset]:
    """Lazy metric values: a Dataset per sample, one variable per metric."""
    unknown = set(samples) - set(SAMPLES)
    if unknown:
        raise ValueError(f"Unknown samples {sorted(unknown)}; choose from {list(SAMPLES)}")
    spatial = list(spatial_dims)
    time_dims = ["time"] if has_time else []
    groups = make_groups(reference["time"] if has_time else None, by, frequency)

    def weights_for(valid: xr.DataArray, spatial_w: bool) -> xr.DataArray:
        w = valid.astype(float)
        if spatial_w and spatial_weights is not None:
            w = w * spatial_weights
        out: xr.DataArray = w
        return out

    results: dict[str, xr.Dataset] = {}
    for sample in samples:
        if sample == "space_time" and not (spatial or time_dims):
            continue
        if sample in ("climatology", "space") and not spatial:
            continue
        if sample in ("space", "time") and not has_time:
            continue
        per_group: list[xr.Dataset] = []
        names: list[str] = []
        for name, idx in groups:
            if sample == "space" and name != "all":
                continue
            r = reference if idx is None else reference.isel(time=idx)
            c = candidate if idx is None else candidate.isel(time=idx)
            if sample == "climatology" and has_time:
                r = r.mean("time")
                c = c.mean("time")
                valid = r.notnull() & c.notnull()
                r, c = r.where(valid), c.where(valid)
            else:
                valid = r.notnull() & c.notnull()
            if sample == "space_time":
                dims = spatial + time_dims
            elif sample in ("climatology", "space"):
                dims = spatial
            else:
                dims = time_dims
            w = weights_for(valid, spatial_w=sample != "time")
            per_group.append(_reduce_all(metrics, r, c, w, valid, dims))
            names.append(name)
        if len(per_group) == 1:
            ds = per_group[0].expand_dims(group=names)
        else:
            ds = xr.concat(per_group, dim="group", join="outer")
            ds = ds.assign_coords(group=names)
        ds.attrs["sample"] = sample
        ds.attrs["description"] = SAMPLES[sample]
        results[sample] = ds
    return results


def scalar_rows(
    samples: dict[str, xr.Dataset], metrics: Sequence[BaseMetric], units: str | None
) -> list[dict[str, Any]]:
    """Long-format rows for the scalar samples (one row per value)."""
    rows: list[dict[str, Any]] = []
    for sample in SCALAR_SAMPLES:
        if sample not in samples:
            continue
        ds = samples[sample]
        for m in [*metrics, None]:
            name = m.name if m is not None else "n_points"
            da = ds[name]
            extra = [d for d in da.dims if d != "group"]
            stacked = da.stack(_point=extra) if extra else da.expand_dims(_point=[0])
            for gi, group in enumerate(ds["group"].values):
                for pi in range(stacked.sizes["_point"]):
                    val = stacked.isel(group=gi, _point=pi).values
                    row: dict[str, Any] = {
                        "sample": sample,
                        "group": str(group),
                        "metric": name,
                        "value": float(val),
                        "units": metric_units(m, units),
                    }
                    for d in extra:
                        coord = stacked[d].values[pi]
                        row[str(d)] = coord.item() if hasattr(coord, "item") else coord
                    rows.append(row)
    return rows


def metric_units(metric: BaseMetric | None, data_units: str | None) -> str:
    if metric is None:
        return "count"
    kind = metric.units_kind
    if kind == "data":
        return data_units or ""
    if kind == "dimensionless":
        return "1"
    if kind == "percent":
        return "%"
    return ""
