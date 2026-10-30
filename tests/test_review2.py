# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the second independent review of 0.2.0 (alignment and metrics).

Names follow the review: C critical, M major, m minor, n nit. The scale tests
build dask graphs of ERA5-sized inputs from lazy zeros and count tasks and
chunks; nothing of that size is computed.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import dask
import dask.array as dsa
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import make_field, monthly_times

import climval
from climval.align import align
from climval.align.regrid import regrid
from climval.align.units import harmonise_units
from climval.align.weights import cell_area
from climval.errors import NoOverlapError, UnitsError
from climval.metrics import PercentileBias, get_metric
from climval.metrics.engine import SAMPLES, complete_seasons
from climval.metrics.stats import SpearmanCorrelation, histogram_quantiles


class CountingScheduler:
    """Synchronous scheduler that counts graph executions."""

    def __init__(self) -> None:
        self.computes = 0

    def __call__(self, dsk: Any, keys: Any, **kwargs: Any) -> Any:
        self.computes += 1
        return dask.get(dsk, keys, **kwargs)


def _era5(chunks: dict[str, int], years: int = 1, freq: str = "h", start: int = 2000) -> xr.Dataset:
    """ERA5-shaped lazy zeros: 0.25 degree global, latitude descending, float32."""
    if freq == "h":
        t = pd.date_range(f"{start}-01-01", f"{start + years}-01-01", freq="h", inclusive="left")
    else:
        t = pd.date_range(f"{start}-01-01", periods=12 * years, freq="MS")
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(1440) * 0.25
    shape = (t.size, lat.size, lon.size)
    ch = tuple(
        chunks.get(d, n) for d, n in zip(("time", "latitude", "longitude"), shape, strict=True)
    )
    data = dsa.zeros(shape, chunks=ch, dtype="float32") + np.float32(285.0)
    return xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), data, {"units": "K"})},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )


def _cmip(years: int = 30, freq: str = "D", tchunk: int = 365, start: int = 1985) -> xr.Dataset:
    """CMIP6-shaped lazy zeros: 1 degree, noleap calendar, dask time, lat and lon bounds."""
    n = 365 * years if freq == "D" else 12 * years
    edges = xr.date_range(
        f"{start}-01-01", periods=n + 1, freq=freq, calendar="noleap", use_cftime=True
    )
    lower, upper = np.asarray(edges[:-1]), np.asarray(edges[1:])
    mid = np.asarray([a + (b - a) / 2 for a, b in zip(lower, upper, strict=True)])
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    data = dsa.zeros((n, lat.size, lon.size), chunks=(tchunk, -1, -1), dtype="float32")
    return xr.Dataset(
        {
            "tas": (("time", "lat", "lon"), data + np.float32(286.0), {"units": "K"}),
            "time_bnds": (
                ("time", "bnds"),
                dsa.from_array(np.stack([lower, upper], 1), chunks=(tchunk, 2)),
            ),
            "lat_bnds": (("lat", "bnds"), dsa.from_array(np.stack([lat - 0.5, lat + 0.5], 1))),
            "lon_bnds": (("lon", "bnds"), dsa.from_array(np.stack([lon - 0.5, lon + 0.5], 1))),
        },
        coords={
            "time": ("time", mid, {"bounds": "time_bnds"}),
            "lat": ("lat", lat, {"units": "degrees_north", "bounds": "lat_bnds"}),
            "lon": ("lon", lon, {"units": "degrees_east", "bounds": "lon_bnds"}),
        },
    )


def _build(ref: Any, can: Any, **kwargs: Any) -> tuple[climval.Comparison, int, float]:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = climval.compare(ref, can, ("t2m", "tas"), compute=False, **kwargs)
    seconds = time.perf_counter() - t0
    keys: set[Any] = set()
    for ds in res.samples.values():
        keys.update(dict(ds.__dask_graph__()).keys())
    return res, len(keys), seconds


def _input_tasks(*arrays: xr.DataArray) -> int:
    return sum(len(dict(a.data.__dask_graph__())) for a in arrays)


# ---------------------------------------------------------------------------
# C1, M5, M6: graph size at ERA5 scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunks",
    [
        {"time": 8760, "latitude": 20, "longitude": 20},  # time-series layout
        {"time": 48, "latitude": 145, "longitude": 288},  # spatial tiles
        {"time": 1},  # ARCO-ERA5 layout, one hour per chunk
    ],
    ids=["timeseries", "tiles", "arco"],
)
def test_c1_m5_m6_hourly_year_against_daily_model_builds_a_small_graph(
    chunks: dict[str, int],
) -> None:
    ref, can = _era5(chunks), _cmip()
    res, tasks, seconds = _build(ref, can)
    added = tasks - _input_tasks(ref["t2m"], can["tas"])
    # 0.2.0: 12.0 million tasks and a MemoryError (time series), 2.26 million (tiles),
    # 398 thousand (ARCO). Now each is below 25 thousand tasks on top of the inputs.
    assert added < 25_000, added
    assert seconds < 60
    a = res.aligned
    assert a.reference.chunks == a.candidate.chunks
    assert a.reference.data.npartitions <= 4  # whole fields, automatic size along time
    assert a.reference.dtype == np.float64


def test_m6_thirty_years_daily_from_hourly_builds_in_seconds() -> None:
    ref, can = _era5({"time": 24}, years=30, start=1985), _cmip()
    res, tasks, seconds = _build(ref, can, by="season")
    added = tasks - _input_tasks(ref["t2m"], can["tas"])
    # About six tasks per input chunk of one day (10 950 chunks): latitude sort,
    # two window reductions, and the merge into whole-field chunks. Current
    # xarray, dask and flox add about 67 000 tasks; the lowest supported versions
    # add about 80 000 (older dask rechunks in more pieces and older xarray maps
    # where and isnan over more of them), with the same 43 aligned chunks. A
    # graph with one task per hour (262 800 hours) or per hour and column piece
    # is far above this bound.
    assert added < 90_000, added
    assert seconds < 120
    assert res.aligned.reference.sizes["time"] == 30 * 365


def test_m6_thirty_years_monthly_builds_in_seconds() -> None:
    ref, can = _era5({"time": 1}, years=30, freq="MS", start=1985), _cmip(freq="MS", tchunk=120)
    res, tasks, seconds = _build(ref, can, by="month")
    assert tasks < 20_000, tasks  # 0.2.0: 130 thousand
    assert seconds < 120
    assert res.aligned.reference.sizes["time"] == 360


def test_m6_padding_is_added_after_time_aggregation() -> None:
    ref = _era5({"time": 1})
    pair = align(ref.isel(time=slice(0, 24 * 31)), _cmip(years=1, start=2000), ("t2m", "tas"))
    region = next(s for s in pair.steps if s["step"] == "region")
    assert "reference_padded" in region
    # One chunk per aggregated day at most, never one per hour and column piece.
    assert pair.reference.data.npartitions <= 2


def test_m5_interpolation_works_on_whole_fields() -> None:
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(0.25, 360.0, 0.5)
    src = xr.DataArray(
        dsa.zeros((48, lat.size, lon.size), chunks=(1, 20, 20)),
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2000", periods=48, freq="D"), "lat": lat, "lon": lon},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, _ = regrid(
            src, np.arange(-88.0, 89.0, 2.0), np.arange(1.0, 359.0, 2.0), "bilinear", "x"
        )
    # 0.2.0: 4 272 chunks of (1, 1, 179) and 6.9 million tasks
    assert out.data.npartitions == 1
    assert len(dict(out.data.__dask_graph__())) < 50_000


# ---------------------------------------------------------------------------
# M4: bounds read in one compute
# ---------------------------------------------------------------------------


def test_m4_dataset_with_dask_bounds_reads_them_in_one_compute() -> None:
    ref = _cmip(years=2, start=2000, tchunk=100).rename({"tas": "t2m"})
    ref = ref.isel(lat=slice(0, 40), lon=slice(0, 60))
    can = _cmip(years=3, start=1999, tchunk=100).isel(lat=slice(0, 40), lon=slice(0, 60))
    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        lazy = climval.compare(ref, can, ("t2m", "tas"), compute=False)
    assert scheduler.computes == 1  # 0.2.0: 3 (time bounds twice, latitude bounds once)
    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        lazy.compute()
    assert scheduler.computes == 1
    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        climval.compare(ref, can, ("t2m", "tas"))
    assert scheduler.computes == 2  # 0.2.0: 4
    weights = next(s for s in lazy.provenance["steps"] if s["step"] == "weights")
    assert "dataset bounds" in weights["weights"]


def test_m4_dataarrays_are_not_computed() -> None:
    ref = make_field(
        np.arange(-87.0, 90.0, 6.0), np.arange(3.0, 360.0, 6.0), monthly_times("2000-01-01", 12)
    )
    scheduler = CountingScheduler()
    with dask.config.set(scheduler=scheduler):
        climval.compare(
            ref.chunk({"time": 4}),
            (ref + 1).assign_attrs(units="K").chunk({"time": 4}),
            compute=False,
        )
    assert scheduler.computes == 0


# ---------------------------------------------------------------------------
# M1: constant fields
# ---------------------------------------------------------------------------

RATIO_METRICS = ("pearson_r", "nrmse", "std_ratio", "taylor_skill_score", "kge")


def _monthly(values: np.ndarray) -> xr.DataArray:
    lat = np.arange(-60.0, 61.0, 2.0)
    lon = np.arange(0.0, 360.0, 2.0)
    t = pd.date_range("2000-01-01", periods=24, freq="MS")
    data = np.broadcast_to(values, (24, lat.size, lon.size)).copy()
    return xr.DataArray(
        data, dims=("time", "lat", "lon"), coords={"time": t, "lat": lat, "lon": lon}, name="v"
    )


@pytest.mark.parametrize("chunked", [False, True])
def test_m1_constant_reference_gives_nan_not_1e15(chunked: bool) -> None:
    ref = _monthly(np.asarray(0.9)).assign_attrs(units="1")
    can = ref.copy(data=np.random.default_rng(0).uniform(0, 1, ref.shape))
    if chunked:
        ref, can = ref.chunk({"time": 5}), can.chunk({"time": 5})
    res = climval.compare(ref, can)
    for sample in ("space_time", "climatology"):
        for m in RATIO_METRICS:
            assert np.isnan(res.value(m, sample=sample)), (sample, m)
    assert res.value("rmse") > 0


def test_m1_two_constants_are_not_a_perfect_correlation() -> None:
    ref = _monthly(np.asarray(3.0)).assign_attrs(units="mm/day")
    can = (ref * 1.1 / 86400).assign_attrs(units="kg m-2 s-1")
    res = climval.compare(ref, can)
    for m in RATIO_METRICS:
        assert np.isnan(res.value(m)), m
    assert res.value("mean_bias") == pytest.approx(0.3)


@pytest.mark.parametrize("name", RATIO_METRICS)
def test_m1_compute_path_uses_the_same_tolerance(name: str) -> None:
    rng = np.random.default_rng(1)
    value = get_metric(name).compute(np.full(10, 0.9), rng.uniform(size=10), rng.uniform(1, 2, 10))
    assert np.isnan(value)


def test_m1_small_but_real_variability_is_kept() -> None:
    rng = np.random.default_rng(2)
    r = 280.0 + 1e-6 * rng.normal(size=500)
    c = r + 1e-7 * rng.normal(size=500)
    assert np.isfinite(get_metric("pearson_r").compute(r, c))
    assert np.isfinite(get_metric("nrmse").compute(r, c))


# ---------------------------------------------------------------------------
# m1, m2, m7: percentiles and ranks
# ---------------------------------------------------------------------------


def test_m1_histogram_quantile_stays_inside_the_sample() -> None:
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(0, 100, 999), [100.0]])
    w = rng.uniform(0.1, 1.0, x.size)
    xa, wa = xr.DataArray(x, dims="s").chunk(100), xr.DataArray(w, dims="s").chunk(100)
    for q in (0.0, 0.999, 1.0):
        qr, _ = histogram_quantiles(xa, xa, wa, ["s"], q)
        value = float(qr.compute())
        assert x.min() <= value <= x.max()
    qr, _ = histogram_quantiles(xa, xa, wa, ["s"], 1.0)
    assert float(qr.compute()) == 100.0  # 0.2.0: 100.0000244


def test_m1_histogram_bound_against_the_weighted_ecdf() -> None:
    rng = np.random.default_rng(4)
    for _ in range(50):
        n = int(rng.integers(20, 2000))
        x, w = rng.normal(280, 10, n), rng.uniform(0.05, 1, n)
        order = np.argsort(x)
        cw = np.cumsum(w[order])
        q = 0.95
        inverse = x[order][np.searchsorted(cw, q * cw[-1])]
        xa, wa = (
            xr.DataArray(x, dims="s").chunk(n // 3 + 1),
            xr.DataArray(w, dims="s").chunk(n // 3 + 1),
        )
        value = float(histogram_quantiles(xa, xa, wa, ["s"], q)[0].compute())
        assert abs(value - inverse) <= 2.0**-20 * abs(inverse)


def test_m1_histogram_and_numpy_are_at_most_one_sorted_value_apart_for_equal_weights() -> None:
    rng = np.random.default_rng(7)
    for _ in range(60):
        n = int(rng.integers(20, 2000))
        x = rng.normal(280, 10, n)
        xa = xr.DataArray(x, dims="s").chunk(n // 3 + 1)
        for q in (0.05, 0.5, 0.95):
            exact = np.percentile(x, 100 * q)
            approx = float(histogram_quantiles(xa, xa, xr.ones_like(xa), ["s"], q)[0].compute())
            lo, hi = min(exact, approx), max(exact, approx)
            assert int(np.sum((x > lo) & (x < hi))) <= 1


def test_m2_spearman_ignores_zero_weight_points() -> None:
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)
    w = np.ones(n)
    w[n // 2 :] = 0.0
    y[n // 2 :] = -x[n // 2 :] * 5
    from scipy.stats import spearmanr

    expected = spearmanr(x[: n // 2], y[: n // 2]).statistic
    xa, ya, wa = (xr.DataArray(a, dims="s") for a in (x, y, w))
    assert SpearmanCorrelation().reduce(xa, ya, wa, ["s"]).item() == pytest.approx(
        expected, rel=1e-12
    )
    assert SpearmanCorrelation().compute(x, y, w) == pytest.approx(expected, rel=1e-12)
    lazy = SpearmanCorrelation().reduce(xa.chunk(500), ya.chunk(500), wa.chunk(500), ["s"])
    assert float(lazy.compute()) == pytest.approx(expected, rel=1e-12)


def test_m2_exact_percentile_ignores_zero_weight_points() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(280, 5, 400)
    y = x + 1.0
    w = np.ones(400)
    w[200:] = 0.0
    x[200:] += 100.0
    xa, ya, wa = (xr.DataArray(a, dims="s") for a in (x, y, w))
    got = PercentileBias(95, method="exact").reduce(xa, ya, wa, ["s"]).item()
    kept = np.percentile(x[:200], 95)
    expected = 100 * (np.percentile(y[:200], 95) - kept) / kept
    assert got == pytest.approx(expected, rel=1e-9)


def test_m7_exact_percentile_on_a_large_dask_sample_warns() -> None:
    big = xr.DataArray(dsa.zeros((4000, 100, 100), chunks=(100, 100, 100)), dims=("time", "y", "x"))
    with pytest.warns(UserWarning, match="gathers each sample"):
        PercentileBias(95, method="exact").reduce(big, big, xr.ones_like(big), ["time", "y", "x"])


def test_m15_percentile_bias_near_zero_reference_quantile_is_nan_with_a_note() -> None:
    rng = np.random.default_rng(0)
    values = rng.gamma(0.3, 5, (24, 61, 180))
    values[values < 1] = 0.0
    values += 1e-6  # a tiny positive 5th percentile, as after a unit conversion
    ref = _monthly(np.zeros(1)).copy(data=values).assign_attrs(units="mm/day")
    can = (ref * 1.2 / 86400).assign_attrs(units="kg m-2 s-1")
    res = climval.compare(ref, can)
    assert np.isnan(res.value("percentile_bias_p5"))  # 0.2.0: -18851 %
    assert res.value("percentile_bias_p95") == pytest.approx(20.0, rel=1e-6)
    note = res.provenance["metric_notes"]["percentile_bias_undefined"]
    assert "percentile_bias_p5" in note["groups"]["space_time"]


# ---------------------------------------------------------------------------
# m3, m4, m5, m6, m8: alignment details
# ---------------------------------------------------------------------------


def test_m3_wrapped_longitude_bounds_give_the_right_area() -> None:
    lat = np.arange(-88.75, 90, 2.5)
    lon = np.arange(0, 360, 2.5)
    lb = xr.DataArray(
        np.stack([lat - 1.25, lat + 1.25], 1), dims=("lat", "bnds"), coords={"lat": lat}
    )
    like = xr.DataArray(
        np.zeros((lat.size, lon.size)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    for b in (
        np.stack([lon - 1.25, lon + 1.25], 1),
        np.mod(np.stack([lon - 1.25, lon + 1.25], 1), 360),
    ):
        ob = xr.DataArray(b, dims=("lon", "bnds"), coords={"lon": lon})
        area, how = cell_area(like, lb, ob)
        assert "dataset bounds" in how
        assert float(area.sum()) / (4 * np.pi * 6371000.0**2) == pytest.approx(1.0, rel=1e-12)
        assert float(area.isel(lat=10, lon=0) / area.isel(lat=10, lon=1)) == pytest.approx(1.0)


def test_m4_grids_without_overlap_raise() -> None:
    t = monthly_times("2000-01-01", 12)
    a = make_field(np.arange(30.0, 60.0, 2.0), np.arange(0.0, 40.0, 2.0), t)
    b = make_field(np.arange(30.0, 60.0, 3.0), np.arange(100.0, 140.0, 3.0), t)
    with pytest.raises(NoOverlapError, match="does not overlap"):
        climval.compare(a, b)
    c = make_field(np.arange(-60.0, -30.0, 3.0), np.arange(0.0, 40.0, 3.0), t)
    with pytest.raises(NoOverlapError, match="does not overlap"):
        climval.compare(a, c)


def _daily(calendar: str, days: int = 400) -> xr.DataArray:
    t = xr.date_range(
        "2000-01-01", periods=days, freq="D", calendar=calendar, use_cftime=calendar != "standard"
    )
    lat, lon = np.arange(40.0, 50.0, 2.0), np.arange(0.0, 10.0, 2.0)
    return xr.DataArray(
        np.full((days, lat.size, lon.size), 280.0),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": lat, "lon": lon},
        name="tas",
        attrs={"units": "K"},
    )


def test_m5_daily_360_day_against_standard_warns() -> None:
    with pytest.warns(UserWarning, match="360_day"):
        res = climval.compare(_daily("standard"), _daily("360_day"), metrics=["rmse"])
    step = next(s for s in res.provenance["steps"] if s["step"] == "time")
    assert "freq='MS'" in step["calendar_note"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        climval.compare(_daily("standard"), _daily("360_day"), metrics=["rmse"], freq="MS")


def test_m6_kg_per_square_metre_per_day_converts() -> None:
    ref = _daily("standard").assign_attrs(units="mm/day")
    for units, factor in (("kg m-2 d-1", 1.0), ("kg m-2 h-1", 24.0), ("kg m-2 day-1", 1.0)):
        _, can, rec = harmonise_units(ref, ref.assign_attrs(units=units), "convert")
        assert rec["scale"] == pytest.approx(factor)
        assert rec["via"] == "built-in table"


def test_n_bare_c_is_not_degrees_celsius() -> None:
    ref = _daily("standard")
    with pytest.raises(UnitsError, match="coulomb"):
        harmonise_units(ref, ref.assign_attrs(units="C"), "convert")
    _, can, _ = harmonise_units(ref, (ref - 273.15).assign_attrs(units="degC"), "convert")
    np.testing.assert_allclose(can.values, ref.values)


def test_m8_float32_lazy_and_eager_agree_exactly() -> None:
    rng = np.random.default_rng(6)
    t = pd.date_range("2000-01-01", periods=24 * 60, freq="h")
    lat, lon = np.arange(40.0, 50.0, 0.5), np.arange(0.0, 10.0, 0.5)
    ref = xr.DataArray(
        (280 + 5 * rng.normal(size=(t.size, lat.size, lon.size))).astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": lat, "lon": lon},
        name="tas",
        attrs={"units": "K"},
    )
    can = (ref + np.float32(0.5)).assign_attrs(units="K")
    daily = _daily("standard", 60).sel(lat=slice(40, 50))
    eager = climval.compare(ref, daily, target="candidate")
    lazy = climval.compare(ref.chunk({"time": 100}), daily, target="candidate")
    assert eager.provenance["aligned_values_sha256"] == lazy.provenance["aligned_values_sha256"]
    assert eager.value("rmse") == pytest.approx(lazy.value("rmse"), rel=1e-12)
    assert can.dtype == np.float32


def test_n_coverage_that_empties_a_yearly_input_raises_no_overlap() -> None:
    t = pd.date_range("2000-01-01", periods=24 * 120, freq="h")
    lat, lon = np.arange(50.0, 55.0, 1.0), np.arange(0.0, 5.0, 1.0)
    ref = xr.DataArray(
        np.full((t.size, lat.size, lon.size), 280.0),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": lat, "lon": lon},
        name="tas",
        attrs={"units": "K"},
    )
    with pytest.raises(NoOverlapError, match="min_coverage"):
        climval.compare(ref, ref + 1, freq="YS")


# ---------------------------------------------------------------------------
# m12, m15: samples and seasons, coarser targets
# ---------------------------------------------------------------------------


def test_m12_sample_descriptions_name_the_time_weights() -> None:
    assert "time weight" in SAMPLES["space_time"]
    assert "time weighted" in SAMPLES["time"]
    assert "equal weight" not in SAMPLES["time"]


def test_m15_one_day_of_december_does_not_complete_a_winter() -> None:
    t = xr.DataArray(pd.date_range("2000-12-31", "2001-02-28", freq="D"), dims="time")
    keep, dropped = complete_seasons(t, "daily", 0.8)
    assert not keep.any()
    assert dropped == ["DJF 2001"]
    # Monthly data: one value per month is the whole month.
    m = xr.DataArray(pd.date_range("2000-12-01", periods=3, freq="MS"), dims="time")
    assert complete_seasons(m, "monthly")[0].all()
    # Hourly data with 80 % of every month present.
    h = xr.DataArray(pd.date_range("2000-12-01", "2001-02-28 23:00", freq="h"), dims="time")
    h = h.isel(time=np.nonzero(np.asarray(h.dt.day.values) <= 25)[0])
    assert complete_seasons(h, "hourly", 0.8)[0].all()
    assert not complete_seasons(h, "hourly", 0.9)[0].any()


def test_m15_bilinear_onto_a_coarser_grid_warns() -> None:
    t = monthly_times("2000-01-01", 12)
    fine = make_field(np.arange(-89.5, 90.0, 1.0), np.arange(0.5, 360.0, 1.0), t)
    coarse = make_field(np.arange(-87.5, 90.0, 5.0), np.arange(2.5, 360.0, 5.0), t)
    with pytest.warns(UserWarning, match="conservative"):
        climval.compare(fine, coarse, metrics=["rmse"])
