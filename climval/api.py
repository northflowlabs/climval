# SPDX-License-Identifier: Apache-2.0
"""The primary API: :func:`compare`, plus :func:`demo` and :func:`demo_data`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climval.align import align
from climval.align.inputs import InputLike
from climval.align.region import RegionLike
from climval.align.temporal import DEFAULT_MIN_COVERAGE
from climval.align.units import absolute_offset
from climval.metrics.engine import DEFAULT_SAMPLES, complete_seasons, evaluate
from climval.metrics.stats import BaseMetric, resolve_metrics
from climval.provenance import environment, jsonable, timestamp
from climval.result import Comparison


def _name(obj: Any, inputs: dict[str, Any], fallback: str) -> str:
    for key in ("source_id", "dataset", "title"):
        value = inputs.get("global_attributes", {}).get(key)
        if value:
            return str(value)
    if inputs.get("location"):
        loc = inputs["location"]
        return str(loc if isinstance(loc, str) else loc[0])
    return fallback


def compare(
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
    by: str | None = None,
    seasons: str = "contiguous",
    metrics: Sequence[str | BaseMetric] | None = None,
    samples: Sequence[str] = DEFAULT_SAMPLES,
    names: tuple[str, str] | None = None,
    compute: bool = True,
    open_kwargs: Mapping[str, Any] | None = None,
) -> Comparison:
    """Compare a candidate with a reference after aligning them.

    Parameters
    ----------
    reference, candidate
        ``xarray.DataArray``, ``xarray.Dataset`` (with ``variable``), a path or
        URL to NetCDF, Zarr or CSV (local, or remote through fsspec, for
        example ``gs://`` or ``s3://``; Zarr stores are recognised by a
        ``.zarr`` suffix or by their metadata), or a list of NetCDF paths.
        NetCDF and Zarr are opened with dask chunks and read chunk by chunk
        when the metrics are computed; CSV is read into memory. A CSV with
        latitude and longitude columns that do not form a full grid is
        station data (see :func:`climval.align.align`).
    variable
        Variable name, or a pair (reference name, candidate name) when they
        differ, for example ``("t2m", "tas")``. Optional for single-variable
        inputs.
    region
        ``climval.Box(lat=(s, n), lon=(w, e))`` (``w > e`` crosses the
        antimeridian), a dict with ``lat`` and ``lon``, or a regionmask
        region such as ``"ar6.land:NEU"``. Applied before regridding.
    period
        ``(start, end)`` as ``"YYYY"``, ``"YYYY-MM"`` or ``"YYYY-MM-DD"``,
        read on calendar fields, so it means the same in every calendar.
    regrid
        "bilinear" (default), "nearest" or "conservative" (needs xESMF; the
        only option for curvilinear grids, and the only one that averages a
        fine grid over coarse cells).
    target
        Grid to compare on: "coarser" (default: the grid with the larger
        cells), "reference", "candidate", or an xarray object holding
        the target grid. With station data the stations are the target.
    time
        "intersect" (default) keeps the periods present in both;
        "exact" requires the same periods.
    freq
        Common time frequency: "h" or "<N>h" (for example "3h"), "D", "MS"
        (monthly) or "YS" (yearly). Defaults to the coarser native frequency
        of the two inputs. Native frequencies come from time bounds, then
        from a ``frequency`` attribute (CMIP vocabulary such as "mon",
        "day", "3hr"), then from the time steps. Declare it on an input with
        ``da.assign_attrs(frequency="mon")`` when the steps alone are
        ambiguous (one month per year, a single time step).
    min_coverage
        Fraction (0 to 1] of a period's native steps that must hold a valid
        value, per grid cell, for its aggregated value to count. Default 0.8.
    units
        "convert" (default) converts the candidate to the reference's units,
        "strict" requires equal units, "ignore" skips the check, or a dict
        {"reference": ..., "candidate": ...} declares units first.
    weights
        "area" (default: the given or cell_measures area, else cell area computed
        from the target grid's latitude and longitude bounds; see
        :mod:`climval.align.weights`),
        "cos_lat", None for uniform weights, or a DataArray.
    area
        Cell area on the target grid (for example CMIP ``areacella``).
    by
        None, "season", "month" or "year": add per-group values.
    seasons
        "contiguous" (default): December belongs to the following year's
        DJF and incomplete seasons are left out. "calendar": every
        December, January and February in the period form DJF, as in
        xarray's ``time.season`` grouping.
    metrics
        Names or BaseMetric instances. Defaults to
        :data:`climval.metrics.DEFAULT_METRICS`.
    samples
        Which point sets to compute over; see :mod:`climval.metrics.engine`.
        Default ("space_time", "climatology").
    names
        Display names (reference, candidate).
    compute
        True (default) computes every sample in one ``dask.compute`` call.
        False returns a lazy Comparison; call ``.compute()`` later.
    open_kwargs
        Passed to the xarray opener for paths and URLs (for example
        ``storage_options``).

    Returns
    -------
    Comparison

    Raises
    ------
    climval.errors.InputError
        A reference or candidate is missing. climval never substitutes
        random data; use :func:`climval.demo` for a demonstration.
    climval.errors.AlignmentError, climval.errors.UnitsError
        The inputs cannot be aligned as requested.
    """
    resolved_metrics = resolve_metrics(metrics)
    pair = align(
        reference,
        candidate,
        variable,
        region=region,
        period=period,
        regrid=regrid,
        target=target,
        time=time,
        freq=freq,
        units=units,
        weights=weights,
        area=area,
        min_coverage=min_coverage,
        open_kwargs=open_kwargs,
    )
    offset = absolute_offset(pair.units)
    lazy = evaluate(
        pair.reference,
        pair.candidate,
        pair.weights,
        resolved_metrics,
        pair.spatial_dims,
        pair.has_time,
        samples=samples,
        by=by,
        frequency=pair.frequency,
        time_weights=pair.time_weights,
        seasons=seasons,
        absolute_offset=offset,
        min_coverage=min_coverage,
    )
    notes: dict[str, Any] = {}
    ratio = [m.name for m in resolved_metrics if m.ratio_scale]
    if ratio and offset:
        notes["absolute_scale"] = (
            f"{', '.join(ratio)} computed with both inputs shifted by {offset:g} "
            f"{pair.units}, so their zero is absolute zero (these metrics divide by a "
            "reference value)"
        )
    methods = {
        sample: {
            str(name): str(var.attrs["climval_method"])
            for name, var in ds.data_vars.items()
            if "climval_method" in var.attrs
        }
        for sample, ds in lazy.items()
    }
    methods = {k: v for k, v in methods.items() if v}
    if by == "season" and pair.has_time:
        dropped = (
            complete_seasons(pair.reference["time"], pair.frequency, min_coverage)[1]
            if seasons == "contiguous"
            else []
        )
        notes["seasons"] = {"mode": seasons, "incomplete_seasons_left_out": dropped}
    ref_name = names[0] if names else _name(reference, pair.inputs["reference"], "reference")
    can_name = names[1] if names else _name(candidate, pair.inputs["candidate"], "candidate")
    provenance = jsonable(
        {
            "created": timestamp(),
            "environment": environment(),
            "inputs": pair.inputs,
            "options": {
                **pair.options,
                "by": by,
                "seasons": seasons,
                "samples": list(lazy),
                "metrics": [m.name for m in resolved_metrics],
            },
            "steps": pair.steps,
            "metric_notes": notes,
            "metric_methods": methods,
        }
    )
    result = Comparison(
        reference_name=ref_name,
        candidate_name=can_name,
        aligned=pair,
        metrics=resolved_metrics,
        samples=lazy,
        provenance=provenance,
        by=by,
    )
    return result.compute() if compute else result


def demo_data(seed: int = 0) -> tuple[xr.DataArray, xr.DataArray]:
    """Small synthetic reference and candidate for demonstrations and tests.

    The reference is a 4 degree global grid, longitude -180..180, latitude
    ascending, monthly means 2000 to 2004 in the standard calendar, in K. The
    candidate is a 5 degree grid, longitude 0..360, latitude descending,
    monthly means 1998 to 2003 in the 360_day calendar, in degC, with a warm
    bias of about 1 K and noise. Every alignment step has work to do.
    """
    rng = np.random.default_rng(seed)

    def field(lat: np.ndarray, lon: np.ndarray, months: np.ndarray) -> np.ndarray:
        la = np.deg2rad(lat)[None, :, None]
        lo = np.deg2rad(lon)[None, None, :]
        season = np.cos(2 * np.pi * (months[:, None, None] - 1) / 12.0)
        values: np.ndarray = (
            288.0
            - 30.0 * np.sin(la) ** 2
            + 3.0 * np.cos(lo) * np.cos(la)
            + 8.0 * np.sin(la) * season
        )
        return values

    ref_lat = np.arange(-88.0, 90.0, 4.0)
    ref_lon = np.arange(-178.0, 180.0, 4.0)
    ref_time = pd.date_range("2000-01-01", "2004-12-01", freq="MS") + pd.Timedelta(days=14)
    ref = field(ref_lat, ref_lon, np.asarray(ref_time.month))
    ref = ref + rng.normal(0.0, 0.5, ref.shape)
    reference = xr.DataArray(
        ref,
        dims=("time", "lat", "lon"),
        coords={"time": ref_time, "lat": ref_lat, "lon": ref_lon},
        name="tas",
        attrs={"units": "K", "standard_name": "air_temperature"},
    )

    can_lat = np.arange(87.5, -90.0, -5.0)
    can_lon = np.arange(2.5, 360.0, 5.0)
    can_time = xr.date_range(
        "1998-01-01", periods=72, freq="MS", calendar="360_day", use_cftime=True
    ).shift(15, "D")
    months = np.asarray([t.month for t in can_time])
    can = field(can_lat, can_lon, months) + 1.0 - 273.15
    can = can + rng.normal(0.0, 0.8, can.shape)
    candidate = xr.DataArray(
        can,
        dims=("time", "lat", "lon"),
        coords={"time": can_time, "lat": can_lat, "lon": can_lon},
        name="tas",
        attrs={"units": "degC", "standard_name": "air_temperature"},
    )
    return reference, candidate


def demo(seed: int = 0, **kwargs: Any) -> Comparison:
    """Run :func:`compare` on :func:`demo_data`. The result is flagged synthetic."""
    reference, candidate = demo_data(seed)
    kwargs.setdefault("names", ("synthetic reference", "synthetic candidate"))
    result = compare(reference, candidate, "tas", **kwargs)
    result.synthetic = True
    result.provenance["synthetic_data"] = True
    return result
