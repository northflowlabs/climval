# SPDX-License-Identifier: Apache-2.0
"""Evaluate metrics on an aligned pair, per sample and per group.

A *sample* says which points a number is computed over:

``space_time``
    Every valid point in space and time pooled, each weighted by its
    spatial weight (cell area) times its time weight. One value per group.
``climatology``
    The spatial pattern of the time-mean fields: both inputs are averaged
    over the (paired) time steps of the group first, with time weights,
    then compared over space with spatial weights. This is the usual Taylor
    diagram and Gleckler et al. (2008) statistic. One value per group.
``space``
    The spatial pattern at each time step, giving a time series of values
    (dimension ``time``). Not grouped.
``time``
    Temporal statistics at each grid point, with time weights, giving maps
    (dimensions ``lat`` and ``lon``). One map per group.

Time weights: for monthly and yearly comparisons each time step counts in
proportion to the length of its period in days (in the reference's
calendar), the same weighting climval uses to aggregate months into years.
Daily and sub-daily steps count equally. The ``space_time`` and ``time``
samples and the time mean of ``climatology`` use these time weights.

Groups come from ``by``: "season" (DJF, MAM, JJA, SON), "month" or "year".
The group "all" holds the whole period and is always present. With
``seasons="contiguous"`` (default) a December belongs to the following
year's DJF, and a season of a given year counts only when all three of its
months are present, so partial winters at the start and end of the period
are left out (recorded in the provenance). For daily and sub-daily data a
month counts as present only when at least ``min_coverage`` (default 0.8)
of its time steps are, so one day of December does not complete a DJF.
``seasons="calendar"`` pools every December, January and February in the
period into DJF, whether or not they belong to one winter, as xarray's
``time.season`` does.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from climval.errors import AlignmentError
from climval.metrics.stats import BaseMetric, shared_reductions, wmean

SAMPLES: dict[str, str] = {
    "space_time": (
        "all valid points in space and time pooled, weighted by spatial weight times time weight"
    ),
    "climatology": "spatial pattern of the time-weighted mean fields, spatially weighted",
    "space": "spatial pattern at each time step (one value per time step), spatially weighted",
    "time": "temporal statistics at each grid point (one map), time weighted",
}
SCALAR_SAMPLES = ("space_time", "climatology")
DEFAULT_SAMPLES = ("space_time", "climatology")
GROUPINGS = ("season", "month", "year")
SEASON_MODES = ("contiguous", "calendar")
_SEASONS = ("DJF", "MAM", "JJA", "SON")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


_SEASON_MONTHS = {"DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8}, "SON": {9, 10, 11}}


def _steps_per_day(frequency: str | None) -> float | None:
    if frequency == "daily":
        return 1.0
    if frequency == "hourly":
        return 24.0
    m = re.fullmatch(r"(\d+)-hourly", frequency or "")
    return 24.0 / int(m.group(1)) if m else None


def complete_seasons(
    time: xr.DataArray, frequency: str | None = None, min_coverage: float = 0.8
) -> tuple[np.ndarray, list[str]]:
    """Mask of time steps in complete contiguous seasons, and the seasons left out.

    A December belongs to the DJF of the following year. A season of a year
    is complete when all three of its months are present. For daily and
    sub-daily ``frequency`` ("daily", "hourly", "3-hourly", ...) a month is
    present when at least ``min_coverage`` of its time steps are.
    """
    labels = np.asarray(time.dt.season.values).astype(str)
    month = np.asarray(time.dt.month.values)
    year = np.asarray(time.dt.year.values)
    syear = year + (month == 12)
    present = np.ones(labels.size, dtype=bool)
    per_day = _steps_per_day(frequency)
    if per_day is not None and labels.size:
        key = year * 100 + month
        days = np.asarray(time.dt.days_in_month.values, dtype=float)
        uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        expected = np.zeros(uniq.size)
        expected[inv.ravel()] = days * per_day
        present = (counts >= min_coverage * expected - 1e-9)[inv.ravel()]
    keep = np.zeros(labels.size, dtype=bool)
    dropped: list[str] = []
    for name in _SEASONS:
        for y in np.unique(syear[labels == name]):
            idx = (labels == name) & (syear == y)
            if set(month[idx & present].tolist()) == _SEASON_MONTHS[name]:
                keep |= idx
            else:
                dropped.append(f"{name} {int(y)}")
    return keep, dropped


def make_groups(
    time: xr.DataArray | None,
    by: str | None,
    frequency: str | None,
    seasons: str = "contiguous",
    min_coverage: float = 0.8,
) -> list[tuple[str, np.ndarray | None]]:
    groups: list[tuple[str, np.ndarray | None]] = [("all", None)]
    if by is None:
        return groups
    if by not in GROUPINGS:
        raise AlignmentError(f"by must be one of {GROUPINGS} or None, got {by!r}")
    if seasons not in SEASON_MODES:
        raise AlignmentError(f"seasons must be one of {SEASON_MODES}, got {seasons!r}")
    if time is None:
        raise AlignmentError(f"by={by!r} needs a time dimension")
    if by in ("season", "month") and frequency == "yearly":
        raise AlignmentError(f"by={by!r} needs monthly or finer data, the comparison is yearly")
    if by == "season":
        labels = np.asarray(time.dt.season.values).astype(str)
        if seasons == "contiguous":
            keep, _ = complete_seasons(time, frequency, min_coverage)
            labels = np.where(keep, labels, "")
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
    absolute_offset: float = 0.0,
) -> xr.Dataset:
    out: dict[str, xr.DataArray] = {}
    shifted = (r + absolute_offset, c + absolute_offset) if absolute_offset else (r, c)
    with shared_reductions():
        for m in metrics:
            rr, cc = shifted if m.ratio_scale else (r, c)
            try:
                value = m.reduce(rr, cc, w, list(dims))
            except Exception as exc:
                # Keep the original exception and its traceback; name the metric.
                if sys.version_info >= (3, 11):
                    exc.add_note(f"climval: raised by metric {m.name!r} over dims {list(dims)}")
                raise
            out[m.name] = (
                value.reset_coords(drop=True) if isinstance(value, xr.DataArray) else value
            )
    # Points that count: both values valid and a positive weight.
    out["n_points"] = (valid & (w > 0)).sum(list(dims))
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
    *,
    time_weights: xr.DataArray | None = None,
    seasons: str = "contiguous",
    absolute_offset: float = 0.0,
    min_coverage: float = 0.8,
) -> dict[str, xr.Dataset]:
    """Lazy metric values: a Dataset per sample, one variable per metric.

    ``time_weights`` (dim ``time``) weight each time step, ``absolute_offset``
    is added to both inputs for metrics with ``ratio_scale`` (for example
    273.15 for degC), ``min_coverage`` decides which months of daily or
    sub-daily data count towards a contiguous season; see the module
    docstring for samples and groups.
    """
    unknown = set(samples) - set(SAMPLES)
    if unknown:
        raise ValueError(f"Unknown samples {sorted(unknown)}; choose from {list(SAMPLES)}")
    spatial = list(spatial_dims)
    time_dims = ["time"] if has_time else []
    groups = make_groups(
        reference["time"] if has_time else None, by, frequency, seasons, min_coverage
    )

    def weights_for(valid: xr.DataArray, spatial_w: bool, tw: xr.DataArray | None) -> xr.DataArray:
        w = valid.astype(float)
        if spatial_w and spatial_weights is not None:
            w = w * spatial_weights
        if tw is not None:
            w = w * tw
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
            tw = None
            if time_weights is not None and has_time:
                tw = time_weights if idx is None else time_weights.isel(time=idx)
            if sample == "climatology" and has_time:
                step_w = (r.notnull() & c.notnull()).astype(float)
                if tw is not None:
                    step_w = step_w * tw
                r = wmean(r, step_w, ["time"])
                c = wmean(c, step_w, ["time"])
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
            w = weights_for(
                valid,
                spatial_w=sample != "time",
                tw=tw if sample in ("space_time", "time") else None,
            )
            per_group.append(_reduce_all(metrics, r, c, w, valid, dims, absolute_offset))
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
