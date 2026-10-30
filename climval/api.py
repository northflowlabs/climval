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
from climval.metrics.engine import DEFAULT_SAMPLES, evaluate
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
    target: str | xr.DataArray | xr.Dataset = "reference",
    time: str = "intersect",
    freq: str | None = None,
    units: str | Mapping[str, str] = "convert",
    weights: str | xr.DataArray | None = "area",
    area: xr.DataArray | None = None,
    by: str | None = None,
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
        example ``gs://`` or ``s3://``), or a list of NetCDF paths. Data are
        opened lazily and never loaded into memory as a whole.
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
        only option for curvilinear grids).
    target
        Grid to compare on: "reference" (default), "candidate", "coarser",
        or an xarray object holding the target grid.
    time
        "intersect" (default) keeps the periods present in both;
        "exact" requires the same periods.
    freq
        Common time frequency: "h", "D", "MS" (monthly) or "YS" (yearly).
        Defaults to the coarser native frequency of the two inputs.
    units
        "convert" (default) converts the candidate to the reference's units,
        "strict" requires equal units, "ignore" skips the check, or a dict
        {"reference": ..., "candidate": ...} declares units first.
    weights
        "area" (default; cell area when available, else cos latitude),
        "cos_lat", None for uniform weights, or a DataArray.
    area
        Cell area on the target grid (for example CMIP ``areacella``).
    by
        None, "season", "month" or "year": add per-group values.
    metrics
        Names or BaseMetric instances. Defaults to
        :data:`climval.metrics.DEFAULT_METRICS`.
    samples
        Which point sets to compute over; see :mod:`climval.metrics.engine`.
        Default ("space_time", "climatology").
    names
        Display names (reference, candidate).
    compute
        True (default) computes now in one dask pass. False returns a lazy
        Comparison; call ``.compute()`` later.
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
        open_kwargs=open_kwargs,
    )
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
    )
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
                "samples": list(lazy),
                "metrics": [m.name for m in resolved_metrics],
            },
            "steps": pair.steps,
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
        return (
            288.0
            - 30.0 * np.sin(la) ** 2
            + 3.0 * np.cos(lo) * np.cos(la)
            + 8.0 * np.sin(la) * season
        )

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
