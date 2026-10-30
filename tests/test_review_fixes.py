"""Regression tests for the findings of the independent review of 0.2.0.

Each test reproduces the case that went wrong (names follow the review:
C critical, M major, N minor) and checks the corrected behaviour.
"""

from __future__ import annotations

import math
import sys
import tracemalloc
import types
import warnings
from pathlib import Path
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
from climval.align.inputs import _is_zarr
from climval.align.units import absolute_offset, canonical
from climval.errors import AlignmentError, MissingDependencyError
from climval.metrics import PercentileBias, get_metric
from climval.metrics.engine import evaluate
from climval.metrics.stats import BaseMetric, histogram_quantiles, resolve_metrics


def _block(times: Any, values: Any, units: str = "K", shape: tuple[int, int] = (2, 2)) -> Any:
    v = np.asarray(values, float)[:, None, None] * np.ones((1, *shape))
    return xr.DataArray(
        v,
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": np.arange(float(shape[0])),
            "lon": np.arange(float(shape[1])),
        },
        name="x",
        attrs={"units": units},
    )


def _time_step(res: climval.Comparison) -> dict[str, Any]:
    return next(s for s in res.provenance["steps"] if s["step"] == "time")


def _with_bounds(da: xr.DataArray, lower: Any, upper: Any) -> xr.Dataset:
    ds = da.to_dataset()
    ds["time_bnds"] = (("time", "bnds"), np.stack([np.asarray(lower), np.asarray(upper)], 1))
    ds["time"].attrs["bounds"] = "time_bnds"
    return ds


class BoxMeanRegridder:
    """Stand-in for xesmf.Regridder: mean of the source centres inside each target cell."""

    calls: list[dict[str, Any]] = []

    def __init__(self, src: xr.Dataset, tgt: xr.Dataset, method: str, **kw: Any) -> None:
        self.src, self.tgt = src, tgt
        BoxMeanRegridder.calls.append({"method": method, "kw": kw, "src_lon": src["lon"].values})

    def __call__(self, da: xr.DataArray, **kw: Any) -> xr.DataArray:
        lat_b, lon_b = self.tgt["lat_b"].values, self.tgt["lon_b"].values
        other = [d for d in da.dims if d not in ("lat", "lon")]
        vals = np.asarray(da.transpose(*other, "lat", "lon").values)
        slat, slon = da["lat"].values, da["lon"].values
        out = np.full((*vals.shape[:-2], lat_b.size - 1, lon_b.size - 1), np.nan)
        for j in range(lat_b.size - 1):
            mlat = (slat >= lat_b[j]) & (slat < lat_b[j + 1])
            for i in range(lon_b.size - 1):
                mlon = (slon >= lon_b[i]) & (slon < lon_b[i + 1])
                cell = vals[..., mlat, :][..., mlon]
                if cell.size:
                    out[..., j, i] = cell.reshape(*cell.shape[:-2], -1).mean(-1)
        coords = {d: da[d].values for d in other if d in da.coords}
        coords.update(lat=self.tgt["lat"].values, lon=self.tgt["lon"].values)
        return xr.DataArray(
            out, dims=(*other, "lat", "lon"), coords=coords, name=da.name, attrs=da.attrs
        )


@pytest.fixture
def fake_xesmf(monkeypatch: pytest.MonkeyPatch) -> type[BoxMeanRegridder]:
    BoxMeanRegridder.calls = []
    monkeypatch.setitem(sys.modules, "xesmf", types.SimpleNamespace(Regridder=BoxMeanRegridder))
    return BoxMeanRegridder


def _fine_and_coarse() -> tuple[xr.DataArray, xr.DataArray]:
    """r1.py: 0.25 degree white noise against a 1 degree grid with integer centres."""
    rng = np.random.default_rng(0)
    flat, flon = np.arange(-10.0, 10.01, 0.25), np.arange(0.0, 360.0, 0.25)
    fine = xr.DataArray(
        rng.normal(size=(flat.size, flon.size)),
        dims=("lat", "lon"),
        coords={"lat": flat, "lon": flon},
        name="x",
        attrs={"units": "K"},
    )
    clat, clon = np.arange(-9.0, 9.1, 1.0), np.arange(0.0, 360.0, 1.0)
    coarse = xr.DataArray(
        np.zeros((clat.size, clon.size)),
        dims=("lat", "lon"),
        coords={"lat": clat, "lon": clon},
        name="x",
        attrs={"units": "K"},
    )
    return fine, coarse


# ---------------------------------------------------------------------------
# C1, M5: conservative regridding
# ---------------------------------------------------------------------------


class TestConservative:
    def test_c1_coincident_points_are_not_point_sampled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fine, coarse = _fine_and_coarse()
        monkeypatch.setitem(sys.modules, "xesmf", None)
        # 0.2.0 selected the coincident points and never imported xESMF.
        with pytest.raises(MissingDependencyError, match="xESMF"):
            climval.compare(fine, coarse, "x", regrid="conservative", metrics=["rmse"])

    def test_c1_conservative_averages(self, fake_xesmf: type[BoxMeanRegridder]) -> None:
        fine, coarse = _fine_and_coarse()
        res = climval.compare(fine, coarse, "x", regrid="conservative", metrics=["rmse"])
        step = next(s for s in res.provenance["steps"] if s["step"] == "regrid")
        assert "conservative remapping" in step["action"]
        # mean of 4 x 4 unit-variance points: rmse about 0.25, point sampling gives about 1
        assert res["rmse"] == pytest.approx(0.25, abs=0.02)

    def test_m5_global_source_is_periodic_without_wrapped_columns(
        self, fake_xesmf: type[BoxMeanRegridder]
    ) -> None:
        fine, coarse = _fine_and_coarse()
        climval.compare(fine, coarse, "x", regrid="conservative", metrics=["rmse"])
        call = fake_xesmf.calls[-1]
        assert call["kw"] == {"unmapped_to_nan": True, "periodic": True}
        lon = call["src_lon"]
        assert lon.size == fine.sizes["lon"] and np.unique(np.mod(lon, 360)).size == lon.size

    def test_m5_regional_source_is_not_periodic(self, fake_xesmf: type[BoxMeanRegridder]) -> None:
        fine, coarse = _fine_and_coarse()
        climval.compare(
            fine.sel(lon=slice(0, 40)),
            coarse.sel(lon=slice(0, 40)),
            "x",
            regrid="conservative",
            metrics=["rmse"],
        )
        assert fake_xesmf.calls[-1]["kw"] == {"unmapped_to_nan": True}

    def test_c1_bilinear_on_coarser_target_is_flagged(self) -> None:
        fine, coarse = _fine_and_coarse()
        res = climval.compare(fine, coarse, "x", metrics=["rmse"])
        step = next(s for s in res.provenance["steps"] if s["step"] == "regrid")
        assert "nothing interpolated" not in step["action"]
        assert "regrid='conservative'" in step["note"]


# ---------------------------------------------------------------------------
# C2, C3, M1, M2, N2, N11: time
# ---------------------------------------------------------------------------


class TestNativeFrequency:
    def _july_and_full(self) -> tuple[xr.DataArray, xr.DataArray]:
        tm = pd.date_range("2000-01-01", periods=60, freq="MS")
        full = _block(tm, 10 * np.cos(2 * np.pi * (tm.month - 7) / 12))
        return full.sel(time=full.time.dt.month == 7), full

    def test_c2_one_month_per_year_is_not_taken_for_yearly(self) -> None:
        july, full = self._july_and_full()
        # 0.2.0 averaged the candidate over whole years: bias -9.962, true 0
        with pytest.raises(AlignmentError, match="attrs\\['frequency'\\]"):
            climval.compare(july, full, "x")

    def test_c2_declared_frequency(self) -> None:
        july, full = self._july_and_full()
        res = climval.compare(july.assign_attrs(frequency="mon"), full, "x")
        assert res["mean_bias"] == pytest.approx(0.0, abs=1e-12)
        assert _time_step(res)["matched_periods"] == 5

    def test_c2_time_bounds(self) -> None:
        july, full = self._july_and_full()
        t = pd.DatetimeIndex(july["time"].values)
        ds = _with_bounds(july, t, t + pd.offsets.MonthBegin(1))
        res = climval.compare(ds, full, "x")
        assert res["mean_bias"] == pytest.approx(0.0, abs=1e-12)
        assert "time bounds" in _time_step(res)["native_frequency"]["reference"]

    def test_c2_yearly_data_stamped_at_year_start_still_works(self) -> None:
        tm = pd.date_range("2000-01-01", periods=24, freq="MS")
        ty = pd.date_range("2000-01-01", periods=2, freq="YS")
        res = climval.compare(_block(ty, [1.0, 1.0]), _block(tm, np.ones(24)), "x")
        assert _time_step(res)["frequency"] == "yearly"

    def test_c2_irregular_steps_raise(self) -> None:
        t = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-02-01", "2000-03-01"])
        with pytest.raises(AlignmentError, match="irregular"):
            climval.compare(_block(t, np.ones(4)), _block(t, np.ones(4)), "x")

    def test_c3_single_step_without_frequency_raises(self) -> None:
        ref = _block(pd.date_range("2000-01-01", periods=31, freq="D"), np.arange(31.0))
        can = _block(pd.DatetimeIndex(["2000-01-01"]), [15.0])
        # 0.2.0 compared the monthly mean with 1 January: bias 15, true 0
        with pytest.raises(AlignmentError, match="single time step"):
            climval.compare(ref, can, "x")

    def test_c3_single_step_with_freq_or_bounds(self) -> None:
        ref = _block(pd.date_range("2000-01-01", periods=31, freq="D"), np.arange(31.0))
        can = _block(pd.DatetimeIndex(["2000-01-16"]), [15.0])
        assert climval.compare(ref, can, "x", freq="MS")["mean_bias"] == pytest.approx(0.0)
        ds = _with_bounds(can, [np.datetime64("2000-01-01")], [np.datetime64("2000-02-01")])
        assert climval.compare(ref, ds, "x")["mean_bias"] == pytest.approx(0.0)

    def test_c3_same_single_timestamp_is_matched(self) -> None:
        t = pd.DatetimeIndex(["2000-01-01"])
        res = climval.compare(_block(t, [1.0]), _block(t, [2.0]), "x")
        assert res["mean_bias"] == pytest.approx(1.0)


class TestSubDaily:
    def test_m1_hourly_against_3_hourly_means(self) -> None:
        th = pd.date_range("2000-01-01", periods=24 * 10, freq="h")
        t3 = pd.date_range("2000-01-01 01:30", periods=8 * 10, freq="3h")
        with pytest.warns(UserWarning, match="different sub-daily steps"):
            res = climval.compare(
                _block(th, np.arange(th.size) % 24), _block(t3, np.full(t3.size, 7.0)), "x"
            )
        step = _time_step(res)
        assert step["frequency"] == "3-hourly"
        assert step["matched_periods"] == 80  # 0.2.0 matched 80 of 240 hourly values, no mean
        assert step["reference_resampled"]["aggregation"] == "mean"
        assert res["mean_bias"] == pytest.approx(7.0 - 11.5)

    def test_m1_bounds_put_end_stamped_values_in_their_day(self) -> None:
        start = pd.Timestamp("2000-01-01")
        lower = pd.date_range(start, periods=72, freq="h")
        upper = lower + pd.Timedelta(hours=1)
        values = np.asarray((lower - start).days, dtype=float)  # day index of each interval
        hourly = _block(upper, values)  # stamped at the end, like ERA5 accumulations
        daily = _block(pd.date_range(start, periods=3, freq="D"), [0.0, 1.0, 2.0])
        with_bounds = climval.compare(daily, _with_bounds(hourly, lower, upper), "x")
        assert with_bounds["mean_bias"] == pytest.approx(0.0, abs=1e-12)
        assert _time_step(with_bounds)["relabelled_to_bound_midpoints"] == ["candidate"]
        without = climval.compare(daily, hourly, "x")
        assert abs(without["mean_bias"]) > 0.01

    def test_m1_freq_accepts_hour_multiples(self) -> None:
        th = pd.date_range("2000-01-01", periods=48, freq="h")
        res = climval.compare(_block(th, np.ones(48)), _block(th, np.ones(48)), "x", freq="6h")
        assert _time_step(res)["frequency"] == "6-hourly"
        assert _time_step(res)["matched_periods"] == 8


class TestCoverage:
    def _inputs(self, valid_days: int) -> tuple[xr.DataArray, xr.DataArray]:
        v = np.arange(31.0)
        v[valid_days:] = np.nan
        ref = _block(pd.date_range("2000-01-01", periods=31, freq="D"), v)
        can = _block(pd.DatetimeIndex(["2000-01-01"]), [15.0])
        return ref, can

    def test_m2_low_coverage_masked(self) -> None:
        ref, can = self._inputs(11)
        res = climval.compare(ref, can, "x", freq="MS")
        # 0.2.0 kept the mean of 11 days (bias 10)
        assert res.value("n_points") == 0
        rec = _time_step(res)["reference_resampled"]
        assert rec["min_coverage"] == 0.8 and "per grid cell" in rec["coverage_rule"]

    def test_m2_threshold(self) -> None:
        ref, can = self._inputs(25)  # 25 of 31 days valid: 0.81
        assert climval.compare(ref, can, "x", freq="MS").value("n_points") == 4
        strict = climval.compare(ref, can, "x", freq="MS", min_coverage=1.0)
        assert strict.value("n_points") == 0
        with pytest.raises(AlignmentError, match="min_coverage"):
            climval.compare(ref, can, "x", freq="MS", min_coverage=0.0)

    def test_n11_seasonal_data_message_names_a_working_freq(self) -> None:
        ts = pd.date_range("2000-03-01", periods=8, freq="QS-MAR")
        with pytest.raises(AlignmentError, match="freq='YS'"):
            climval.compare(_block(ts, np.ones(8)), _block(ts, np.ones(8)), "x")
        res = climval.compare(_block(ts, np.ones(8)), _block(ts, np.ones(8)), "x", freq="YS")
        assert _time_step(res)["matched_periods"] == 2


class TestTimeWeights:
    def test_n2_months_weighted_by_length(self) -> None:
        t = monthly_times("2001-01-01", 2)  # January (31 days), February (28 days)
        ref = _block(t, [0.0, 0.0])
        can = _block(t, [1.0, 0.0])
        res = climval.compare(ref, can, "x")
        assert res["mean_bias"] == pytest.approx(31.0 / 59.0)
        assert res.value("mean_bias", "climatology") == pytest.approx(31.0 / 59.0)
        # consistent with aggregation to years, which weights months by length
        yearly = climval.compare(ref, can, "x", freq="YS", min_coverage=0.15)
        assert yearly["mean_bias"] == pytest.approx(31.0 / 59.0)


class TestSeasons:
    def test_n3_contiguous_djf(self) -> None:
        t = monthly_times("2000-01-01", 24)
        values = np.where(pd.DatetimeIndex(t).year == 2000, 1.0, 0.0)
        ref = _block(t, np.zeros(24))
        can = _block(t, values)
        res = climval.compare(ref, can, "x", by="season", metrics=["mean_bias"])
        notes = res.provenance["metric_notes"]["seasons"]
        assert notes["incomplete_seasons_left_out"] == ["DJF 2000", "DJF 2002"]
        # DJF 2001 = December 2000 (bias 1) with January and February 2001 (bias 0)
        assert res.value("mean_bias", group="DJF") == pytest.approx(31.0 / 90.0)
        old = climval.compare(ref, can, "x", by="season", seasons="calendar")
        assert old.value("mean_bias", group="DJF") == pytest.approx(
            (31 + 29 + 31) / (31 + 29 + 31 + 31 + 28 + 31)
        )


# ---------------------------------------------------------------------------
# C4, C5, C6: metrics on large dask arrays and errors
# ---------------------------------------------------------------------------


def _era5_like() -> climval.AlignedPair:
    lat = np.arange(90, -90.01, -0.25)
    lon = np.arange(0, 360, 0.25)
    t = pd.date_range("2000-01-01", periods=8784, freq="h")
    data = dsa.zeros((t.size, lat.size, lon.size), chunks=(24, 721, 1440), dtype="float32")
    ref = xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={"time": t, "latitude": lat, "longitude": lon},
        name="t2m",
        attrs={"units": "K"},
    )
    return align(ref, ref.copy(), "t2m")


def _graph_peak_bytes(pair: climval.AlignedPair, metric: str, samples: tuple[str, ...]) -> int:
    tracemalloc.start()
    try:
        out = evaluate(
            pair.reference,
            pair.candidate,
            pair.weights,
            resolve_metrics([metric]),
            pair.spatial_dims,
            True,
            samples=samples,
            time_weights=pair.time_weights,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    for ds in out.values():
        assert max(max(c) for c in ds[metric].data.chunks) < 10**6
    return peak


class TestLargeSamples:
    def test_c4_percentiles_scale_on_era5_hourly_year(self) -> None:
        pair = _era5_like()
        out = evaluate(
            pair.reference,
            pair.candidate,
            pair.weights,
            resolve_metrics(["percentile_bias_p95"]),
            pair.spatial_dims,
            True,
            samples=("space_time", "climatology"),
        )
        st = out["space_time"]["percentile_bias_p95"]
        assert st.attrs["climval_method"] == "histogram"
        assert out["climatology"]["percentile_bias_p95"].attrs["climval_method"] == "exact"
        # 0.2.0 built single 73 GB chunks here; no task input is that large now
        graph = dict(st.data.__dask_graph__())
        assert not any("rechunk" in str(k) for k in graph)
        assert _graph_peak_bytes(pair, "percentile_bias_p95", ("space_time",)) < 300 * 2**20

    def test_c5_spearman_and_compute_only_metrics_do_not_allocate_index(self) -> None:
        pair = _era5_like()
        # 0.2.0 allocated a 67.9 GiB MultiIndex here before computing anything
        with pytest.warns(UserWarning, match="gathers each sample"):
            peak = _graph_peak_bytes(pair, "spearman_r", ("space_time",))
        assert peak < 300 * 2**20

        class MaxAbs(BaseMetric):
            name = "max_abs"

            def compute(self, reference: Any, candidate: Any, weights: Any = None) -> float:
                return float(np.max(np.abs(candidate - reference)))

        tracemalloc.start()
        try:
            with pytest.warns(UserWarning, match="gathers each sample"):
                MaxAbs().reduce(
                    pair.reference,
                    pair.candidate,
                    xr.ones_like(pair.reference),
                    ["time", "lat", "lon"],
                )
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 300 * 2**20

    @pytest.mark.parametrize("q", [0.05, 0.5, 0.95])
    def test_c4_histogram_accuracy(self, q: float) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(280.0, 5.0, (40, 30, 50))
        r = xr.DataArray(x, dims=("time", "lat", "lon")).chunk({"time": 7, "lat": 11})
        w = xr.ones_like(r)
        qr, _ = histogram_quantiles(r, r + 1.0, w, ["time", "lat", "lon"], q)
        s = np.sort(x.ravel())
        k = int(math.floor(q * (s.size - 1)))
        bound = 2.0**-20 * abs(s[k]) + (s[k + 1] - s[k - 1])
        assert abs(float(qr) - np.percentile(x, 100 * q)) <= bound

    def test_c4_histogram_weights_equal_replication(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.gamma(2.0, 3.0, (20, 30))
        counts = rng.integers(1, 4, x.shape)
        r = xr.DataArray(x, dims=("a", "b")).chunk({"a": 6})
        w = xr.DataArray(counts.astype(float), dims=("a", "b")).chunk({"a": 6})
        got, _ = histogram_quantiles(r, r, w, ["a", "b"], 0.9)
        rep = xr.DataArray(np.repeat(x.ravel(), counts.ravel()), dims="n").chunk({"n": 100})
        expected, _ = histogram_quantiles(rep, rep, xr.ones_like(rep), ["n"], 0.9)
        assert float(got) == pytest.approx(float(expected), rel=1e-12)

    def test_c4_histogram_keeps_extra_dims_and_one_pass(self) -> None:
        rng = np.random.default_rng(6)
        ref = make_field(
            np.arange(-87.0, 90.0, 6.0),
            np.arange(-177.0, 180.0, 6.0),
            monthly_times("2000-01-01", 12),
        )
        can = ref.expand_dims(member=["a", "b"]) + xr.DataArray(
            rng.normal(0, 1, (2, *ref.shape)), dims=("member", *ref.dims)
        )
        can = can.rename("tas").assign_attrs(units="K")
        exact = climval.compare(ref, can, metrics=[PercentileBias(90, method="exact")])
        counter = {"n": 0}

        def scheduler(dsk: Any, keys: Any, **kw: Any) -> Any:
            counter["n"] += 1
            return dask.get(dsk, keys, **kw)

        with dask.config.set(scheduler=scheduler):
            hist = climval.compare(
                ref.chunk({"time": 4}),
                can.chunk({"time": 4}),
                metrics=[PercentileBias(90, max_exact_bytes=1)],
            )
        assert counter["n"] == 1
        assert hist.provenance["metric_methods"]["space_time"] == {
            "percentile_bias_p90": "histogram"
        }
        a = exact.samples["space_time"]["percentile_bias_p90"].values
        b = hist.samples["space_time"]["percentile_bias_p90"].values
        np.testing.assert_allclose(b, a, atol=0.01)

    def test_c6_original_exception_is_raised(self) -> None:
        class NeedsTwoArgsError(Exception):
            def __init__(self, a: str, b: str) -> None:
                super().__init__(a, b)

        class Broken(BaseMetric):
            name = "broken"

            def reduce(self, reference: Any, candidate: Any, weights: Any, dims: Any) -> Any:
                raise NeedsTwoArgsError("one", "two")

        ref = make_field(
            np.array([0.0, 10.0]), np.array([0.0, 10.0]), monthly_times("2000-01-01", 2)
        )
        # 0.2.0 raised TypeError from type(exc)(message) and hid this error
        with pytest.raises(NeedsTwoArgsError) as info:
            climval.compare(ref, ref, metrics=[Broken()])
        if sys.version_info >= (3, 11):
            assert any("broken" in n for n in info.value.__notes__)


# ---------------------------------------------------------------------------
# M3: ratio-scale metrics
# ---------------------------------------------------------------------------


class TestRatioScale:
    def test_m3_kge_and_percentile_bias_do_not_depend_on_degc_or_k(self) -> None:
        times = monthly_times("2000-01-01", 12)
        lat, lon = np.arange(-80.0, 81.0, 20.0), np.arange(0.0, 360.0, 30.0)
        ref = make_field(lat, lon, times)
        can = ref + np.random.default_rng(1).normal(0.5, 1.0, ref.shape)
        can = can.assign_attrs(ref.attrs)
        in_k = climval.compare(ref, can)
        in_c = climval.compare((ref - 273.15).assign_attrs(units="degC"), can)
        for name in ("kge", "percentile_bias_p95", "percentile_bias_p5"):
            assert in_c[name] == pytest.approx(in_k[name], rel=1e-9), name
        assert "absolute_scale" in in_c.provenance["metric_notes"]
        assert "absolute_scale" not in in_k.provenance["metric_notes"]

    def test_m3_offsets(self) -> None:
        assert absolute_offset("degC") == 273.15
        assert absolute_offset("degF") == 459.67
        assert absolute_offset("K") == 0.0
        assert absolute_offset("mm day-1") == 0.0


# ---------------------------------------------------------------------------
# M4, N1, N13: weights
# ---------------------------------------------------------------------------


class TestWeights:
    def test_m4_non_uniform_latitudes_use_cell_area(self) -> None:
        lat = np.array([-60.0, -20.0, 0.0, 10.0, 50.0])
        lon = np.array([0.0, 10.0, 40.0])
        da = make_field(lat, lon, monthly_times("2000-01-01", 2))
        pair = align(da, da)
        assert pair.weights is not None
        edges = np.array([-80.0, -40.0, -10.0, 5.0, 30.0, 70.0])
        band = np.diff(np.sin(np.deg2rad(edges)))
        widths = np.array([10.0, 20.0, 30.0])  # midpoint bounds -5, 5, 25, 55
        expected = band[:, None] * widths[None, :]
        got = pair.weights.values
        np.testing.assert_allclose(got / got.sum(), expected / expected.sum(), rtol=1e-12)
        cos = align(da, da, weights="cos_lat").weights
        assert cos is not None and cos.dims == ("lat",)

    def test_m4_dataset_bounds_are_used(self) -> None:
        lat = np.array([-45.0, 0.0, 45.0])
        da = make_field(lat, np.array([0.0, 120.0, 240.0]), monthly_times("2000-01-01", 2))
        ds = da.to_dataset()
        ds["lat_bnds"] = (("lat", "bnds"), np.array([[-90.0, -30.0], [-30.0, 30.0], [30.0, 90.0]]))
        ds["lat"].attrs["bounds"] = "lat_bnds"
        pair = align(ds, da, "tas", target="reference")
        assert pair.weights is not None
        w = pair.weights.isel(lon=0).values
        band = np.diff(np.sin(np.deg2rad([-90.0, -30.0, 30.0, 90.0])))
        np.testing.assert_allclose(w / w.sum(), band / band.sum(), rtol=1e-12)
        step = next(s for s in pair.steps if s["step"] == "weights")
        assert "dataset bounds" in step["weights"]

    def test_n1_negative_weights(self) -> None:
        da = make_field(
            np.array([0.0, 10.0]), np.array([0.0, 10.0]), monthly_times("2000-01-01", 2)
        )
        w = xr.DataArray(
            [[1.0, -1.0], [1.0, 1.0]],
            dims=("lat", "lon"),
            coords={"lat": [0.0, 10.0], "lon": [0.0, 10.0]},
        )
        with pytest.raises(AlignmentError, match="negative"):
            climval.compare(da, da, weights=w)
        with pytest.raises(AlignmentError, match="negative"):
            climval.compare(da.chunk(), da.chunk(), weights=w.chunk())
        with pytest.raises(ValueError, match="negative"):
            get_metric("rmse").compute(np.arange(3.0), np.arange(3.0), np.array([1.0, -1.0, 1.0]))

    def test_n13_variable_named_area_is_not_cell_area(self) -> None:
        da = make_field(
            np.array([0.0, 10.0]), np.array([0.0, 10.0]), monthly_times("2000-01-01", 2)
        )
        area = xr.full_like(da.isel(time=0, drop=True), 5.0)
        pair = align(xr.Dataset({"tas": da, "area": area.rename("area")}), da, "tas")
        assert (
            "cell area from bounds"
            in next(s for s in pair.steps if s["step"] == "weights")["weights"]
        )
        named = area.rename("areacella")
        pair = align(xr.Dataset({"tas": da, "areacella": named}), da, "tas")
        assert next(s for s in pair.steps if s["step"] == "weights")["weights"] == (
            "cell area (areacella)"
        )
        tagged = area.rename("cellarea").assign_attrs(standard_name="cell_area")
        pair = align(xr.Dataset({"tas": da, "cellarea": tagged}), da, "tas")
        assert (
            "cell area (cellarea)"
            == next(s for s in pair.steps if s["step"] == "weights")["weights"]
        )


# ---------------------------------------------------------------------------
# M6: Zarr stores without a .zarr suffix
# ---------------------------------------------------------------------------


class TestZarrDetection:
    def test_m6_suffixes(self) -> None:
        assert _is_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
        assert _is_zarr("s3://bucket/store.zarr/")
        assert not _is_zarr("https://example.org/file.nc")

    def test_m6_local_directory_without_suffix(self, tmp_path: Path) -> None:
        pytest.importorskip("zarr")
        da = make_field(
            np.array([0.0, 10.0]), np.array([0.0, 10.0]), monthly_times("2000-01-01", 3)
        )
        store = tmp_path / "CMIP6" / "tas" / "gn" / "v20190308"
        da.to_dataset().to_zarr(store)
        res = climval.compare(str(store), da, "tas")
        assert res["rmse"] == 0.0
        assert res.provenance["inputs"]["reference"]["lazy"] is True

    def test_m6_fsspec_url_without_suffix(self) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("fsspec")
        da = make_field(
            np.array([0.0, 10.0]), np.array([0.0, 10.0]), monthly_times("2000-01-01", 3)
        )
        url = "memory://climval-test/CMIP6/v20190308"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da.to_dataset().to_zarr(url, mode="w")
        assert _is_zarr(url)
        res = climval.compare(url, da, "tas")
        assert res["rmse"] == 0.0 and res.provenance["inputs"]["reference"]["kind"] == "url"


# ---------------------------------------------------------------------------
# M7: dask is a core dependency; files open lazily
# ---------------------------------------------------------------------------


def test_m7_dask_is_core_and_netcdf_opens_lazily(tmp_path: Path) -> None:
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parents[1]
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(d.startswith("dask") for d in meta["project"]["dependencies"])
    pytest.importorskip("netCDF4")
    da = make_field(np.array([0.0, 10.0]), np.array([0.0, 10.0]), monthly_times("2000-01-01", 3))
    da.to_dataset().to_netcdf(tmp_path / "x.nc")
    res = climval.compare(str(tmp_path / "x.nc"), da, compute=False)
    assert isinstance(res.aligned.reference.data, dsa.Array)


# ---------------------------------------------------------------------------
# M8: CLI regions with negative edges
# ---------------------------------------------------------------------------


class TestCliRegion:
    def _args(self, *extra: str) -> Any:
        from climval.cli import _join_negative_values, build_parser

        argv = ["compare", "a.nc", "b.nc", *extra]
        return build_parser().parse_args(_join_negative_values(argv))

    @pytest.mark.parametrize(
        "extra",
        [
            ("--lat", "-60,-30", "--lon", "-10,40"),
            ("--lat", "-60", "-30", "--lon", "-10", "40"),
            ("--lat=-60,-30", "--lon=-10,40"),
            ("--region", "-60,-30,-10,40"),
            ("--region=-60,-30,-10,40",),
        ],
    )
    def test_m8_negative_south_edge(self, extra: tuple[str, ...]) -> None:
        from climval.cli import _region

        args = self._args(*extra)
        box = _region(args.region, args.lat, args.lon)
        assert box == climval.Box(lat=(-60.0, -30.0), lon=(-10.0, 40.0))

    def test_m8_end_to_end(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from climval.cli import main

        ref, can = climval.demo_data()
        ref.to_dataset().to_netcdf(tmp_path / "ref.nc")
        can.to_dataset().to_netcdf(tmp_path / "can.nc")
        main(["compare", str(tmp_path / "ref.nc"), str(tmp_path / "can.nc"), "--lat", "-60,-30"])
        assert "mean_bias" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# N4: station data
# ---------------------------------------------------------------------------


class TestStations:
    def _csv(self, tmp_path: Path) -> tuple[Path, xr.DataArray]:
        rng = np.random.default_rng(0)
        st_lat = rng.uniform(40, 60, 50).round(3)
        st_lon = rng.uniform(0, 20, 50).round(3)
        t = pd.date_range("2000-01-01", periods=12, freq="MS")
        rows = [
            (tt, la, lo, 280 + 0.1 * la) for tt in t for la, lo in zip(st_lat, st_lon, strict=True)
        ]
        path = tmp_path / "st.csv"
        pd.DataFrame(rows, columns=["time", "lat", "lon", "tas"]).to_csv(path, index=False)
        lat, lon = np.arange(30, 70.1, 1.0), np.arange(-10, 30.1, 1.0)
        grid = xr.DataArray(
            280 + 0.1 * lat[None, :, None] + np.zeros((12, 1, lon.size)) + 1,
            dims=("time", "lat", "lon"),
            coords={"time": t, "lat": lat, "lon": lon},
            name="tas",
            attrs={"units": "K"},
        )
        return path, grid

    def test_n4_csv_stations_on_a_site_dimension(self, tmp_path: Path) -> None:
        path, grid = self._csv(tmp_path)
        res = climval.compare(str(path), grid, "tas", units={"reference": "K"})
        # 0.2.0 built a 50 x 50 lat-lon grid that was almost all NaN
        assert res.aligned.reference.dims == ("time", "station")
        assert res.aligned.reference.shape == (12, 50)
        assert res.value("n_points") == 600
        assert res["mean_bias"] == pytest.approx(1.0, abs=1e-9)
        weights = next(s for s in res.provenance["steps"] if s["step"] == "weights")
        assert weights["weights"] == "uniform (station data)"
        assert "climatology" in res.samples

    def test_n4_region_selects_stations(self, tmp_path: Path) -> None:
        path, grid = self._csv(tmp_path)
        res = climval.compare(
            str(path), grid, "tas", units={"reference": "K"}, region=climval.Box(lat=(50, 60))
        )
        n = res.aligned.reference.sizes["station"]
        assert 0 < n < 50

    def test_n4_full_grid_csv_stays_gridded(self, tmp_path: Path) -> None:
        lat, lon = [0.0, 10.0], [0.0, 10.0, 20.0]
        rows = [(la, lo, 1.0) for la in lat for lo in lon]
        path = tmp_path / "grid.csv"
        pd.DataFrame(rows, columns=["lat", "lon", "x"]).to_csv(path, index=False)
        from climval.align.inputs import open_path

        assert open_path(str(path))["x"].dims == ("lat", "lon")


# ---------------------------------------------------------------------------
# N5: edges and poles
# ---------------------------------------------------------------------------


class TestEdges:
    @pytest.mark.parametrize("method", ["nearest", "bilinear"])
    def test_n5_half_cell_beyond_edge(self, method: str) -> None:
        times = monthly_times("2000-01-01", 2)
        src = make_field(np.arange(0.0, 10.1, 2.0), np.arange(0.0, 10.1, 2.0), times)
        tgt = make_field(np.array([5.0, 10.9, 11.5]), np.array([4.0]), times)
        pair = align(tgt, src, regrid=method, target="reference")
        col = pair.candidate.isel(time=0, lon=0).values
        assert np.isfinite(col[1]) and np.isnan(col[2])
        assert col[1] == pytest.approx(float(src.sel(lat=10.0, lon=4.0)[0]))

    def test_n5_polar_rows_kept(self) -> None:
        times = monthly_times("2000-01-01", 2)
        coarse = make_field(np.arange(-88.75, 89.0, 2.5), np.arange(1.25, 360.0, 2.5), times)
        fine = make_field(np.arange(-89.5, 90.0, 1.0), np.arange(0.5, 360.0, 1.0), times)
        pair = align(fine, coarse, target="reference")
        assert bool(pair.candidate.isel(lat=[0, -1]).notnull().all())


# ---------------------------------------------------------------------------
# N6: default target
# ---------------------------------------------------------------------------


def test_n6_default_target_is_the_coarser_grid() -> None:
    times = monthly_times("2000-01-01", 2)
    fine = make_field(np.arange(-40.0, 41.0, 2.0), np.arange(0.0, 41.0, 2.0), times)
    coarse = make_field(np.arange(-40.0, 41.0, 10.0), np.arange(0.0, 41.0, 10.0), times)
    pair = align(fine, coarse)
    assert pair.reference.sizes["lat"] == 9
    step = next(s for s in pair.steps if s["step"] == "target_grid")
    assert step == {
        "step": "target_grid",
        "target": "candidate",
        "requested": "coarser",
        "regridded": ["reference"],
    }


# ---------------------------------------------------------------------------
# N8: load_model domains
# ---------------------------------------------------------------------------


def test_n8_load_model_domains_applied() -> None:
    from datetime import datetime

    ref, can = climval.demo_data()
    a = climval.load_model(name="a", variables=["tas"], data=ref, lat_range=(0.0, 40.0))
    b = climval.load_model(
        name="b",
        variables=["tas"],
        data=can,
        time_start=datetime(2001, 1, 1),
        time_end=datetime(2001, 12, 31),
    )
    suite = climval.BenchmarkSuite().register(a, role="reference").register(b)
    with pytest.warns(FutureWarning):
        report = suite.run()
    comp = report.results[0].comparisons["tas"]
    assert comp.provenance["options"]["region"]["lat"] == [0.0, 40.0]
    assert comp.provenance["options"]["period"] == ["2001-01-01", "2001-12-31"]
    assert "not set" in climval.load_model("era5").summary()


# ---------------------------------------------------------------------------
# N9, N10: names
# ---------------------------------------------------------------------------


def test_n9_threshold_aliases() -> None:
    ref = make_field(np.array([0.0, 10.0]), np.array([0.0, 10.0]), monthly_times("2000-01-01", 2))
    res = climval.compare(ref, (ref + 0.5).assign_attrs(ref.attrs))
    verdict = res.check({"bias": {"abs_max": 1.0}, "tss": {"min": 0.9}, "pearson": {"min": 0.9}})
    assert verdict.passed
    assert [c.metric for c in verdict.checks] == ["mean_bias", "taylor_skill_score", "pearson_r"]


def test_n10_percentile_names() -> None:
    assert PercentileBias(99.9).name == "percentile_bias_p99_9"
    assert PercentileBias(99).name == "percentile_bias_p99"
    assert resolve_metrics([PercentileBias(99), PercentileBias(99.9)])[1].name.endswith("99_9")
    m = get_metric("percentile_bias_p99_9")
    assert isinstance(m, PercentileBias) and m.percentile == pytest.approx(99.9)
    assert get_metric("percentile_bias_p2.5").name == "percentile_bias_p2_5"
    with pytest.raises(ValueError, match="99_9"):
        get_metric("percentile_bias_p999")


# ---------------------------------------------------------------------------
# Nits: unit case
# ---------------------------------------------------------------------------


def test_unit_symbols_are_case_sensitive() -> None:
    assert canonical("mb") == "hPa"
    assert canonical("MB") is None
    assert canonical("KELVIN") == "K"
    assert canonical("Degrees Celsius") == "degC"
    assert canonical("percent") == "%"
