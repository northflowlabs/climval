# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of climval.compare and the Comparison result."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import make_field, monthly_times, smooth

import climval
from climval.errors import InputError
from climval.metrics.engine import SAMPLES


def _candidate_on_other_grid(rng_seed: int = 1) -> xr.DataArray:
    """5 degree grid, 0..360, descending latitude, 360_day calendar, degC."""
    lat = np.arange(87.5, -90.0, -5.0)
    lon = np.arange(2.5, 360.0, 5.0)
    times = monthly_times("1999-07-01", 30, calendar="360_day", day=16)
    can = make_field(lat, lon, times)
    noise = np.random.default_rng(rng_seed).normal(0.0, 0.3, can.shape)
    return (can + 0.7 + noise - 273.15).assign_attrs(units="degC")


class TestDifferentGridsAndCalendars:
    def test_same_metrics_as_manual_alignment(self, ref_grid: xr.DataArray) -> None:
        xs = pytest.importorskip("xskillscore")
        can = _candidate_on_other_grid()
        result = climval.compare(ref_grid, can, "tas")

        # Manual alignment written out independently of climval's code path.
        m = can + 273.15
        m = m.assign_coords(lon=((m["lon"] + 180) % 360) - 180).sortby("lon").sortby("lat")
        m = xr.concat(
            [
                m.isel(lon=[-1]).assign_coords(lon=m["lon"][-1:] - 360),
                m,
                m.isel(lon=[0]).assign_coords(lon=m["lon"][:1] + 360),
            ],
            dim="lon",
        )
        m = m.interp(lat=ref_grid["lat"], lon=ref_grid["lon"])
        keys_can = [(t.year, t.month) for t in m["time"].values]
        keys_ref = [(t.year, t.month) for t in pd.DatetimeIndex(ref_grid["time"].values)]
        common = [k for k in keys_ref if k in keys_can]
        r = ref_grid.isel(time=[keys_ref.index(k) for k in common])
        m = m.isel(time=[keys_can.index(k) for k in common]).assign_coords(time=r["time"])
        valid = r.notnull() & m.notnull()
        r, m = r.where(valid), m.where(valid)
        # cell area on a uniform grid is proportional to cos(lat); each month
        # counts by its length in days
        w = (np.cos(np.deg2rad(r["lat"])) * r["time"].dt.days_in_month).broadcast_like(r)

        dims = ["time", "lat", "lon"]
        expected = {
            "rmse": xs.rmse(m, r, dim=dims, weights=w, skipna=True),
            "mae": xs.mae(m, r, dim=dims, weights=w, skipna=True),
            "mean_bias": xs.me(m, r, dim=dims, weights=w, skipna=True),
            "pearson_r": xs.pearson_r(r, m, dim=dims, weights=w, skipna=True),
        }
        for name, value in expected.items():
            assert result[name] == pytest.approx(float(value), rel=1e-10), name
        assert len(common) == 24
        assert result.value("n_points") == int(valid.sum())
        # The candidate was built with a +0.7 K bias: the aligned bias finds it.
        assert result["mean_bias"] == pytest.approx(0.7, abs=0.05)

    def test_unaligned_0_1_path_would_be_wrong(self, ref_grid: xr.DataArray) -> None:
        """Ravel-and-truncate (0.1 behaviour) gives a meaningless bias here."""
        can = _candidate_on_other_grid()
        a = ref_grid.values.ravel()
        b = (can.values + 273.15).ravel()
        n = min(a.size, b.size)
        naive_rmse = float(np.sqrt(np.mean((b[:n] - a[:n]) ** 2)))
        result = climval.compare(ref_grid, can, "tas")
        assert naive_rmse > 5 * result["rmse"]


class TestComparison:
    def test_samples_and_groups(self, ref_grid: xr.DataArray) -> None:
        can = (ref_grid + 1.0).assign_attrs(ref_grid.attrs)
        res = climval.compare(ref_grid, can, by="season", samples=list(SAMPLES))
        assert res.groups == ["all", "DJF", "MAM", "JJA", "SON"]
        for g in res.groups:
            assert res.value("mean_bias", group=g) == pytest.approx(1.0)
            assert res.value("pearson_r", "climatology", g) == pytest.approx(1.0)
        assert res.field("space")["rmse"].dims == ("group", "time")
        assert res.field("time")["rmse"].dims == ("group", "lat", "lon")
        np.testing.assert_allclose(res.field("time")["mean_bias"].values, 1.0)

    def test_by_month_and_year(self, ref_grid: xr.DataArray) -> None:
        assert climval.compare(ref_grid, ref_grid, by="month").groups[1:3] == ["Jan", "Feb"]
        assert climval.compare(ref_grid, ref_grid, by="year").groups == ["all", "2000", "2001"]

    def test_climatology_differs_from_space_time(self, ref_grid: xr.DataArray) -> None:
        rng = np.random.default_rng(0)
        can = (ref_grid + rng.normal(0, 1, ref_grid.shape)).assign_attrs(ref_grid.attrs)
        res = climval.compare(ref_grid, can)
        # time-averaging removes most noise from the pattern
        assert res.value("rmse", "climatology") < 0.5 * res.value("rmse", "space_time")

    def test_outputs(self, ref_grid: xr.DataArray, tmp_path: object) -> None:
        other = (ref_grid + 0.5).assign_attrs(ref_grid.attrs)
        res = climval.compare(ref_grid, other, names=("ref", "cand"))
        frame = res.to_dataframe()
        assert {"sample", "group", "metric", "value", "units"} <= set(frame.columns)
        d = json.loads(res.to_json())
        assert d["reference"] == "ref" and d["synthetic_data"] is False
        assert any(v["metric"] == "rmse" and v["units"] == "K" for v in d["values"])
        for ext in ("json", "html", "md"):
            path = tmp_path / f"r.{ext}"  # type: ignore[operator]
            res.export(path)
            assert path.stat().st_size > 0
        html = (tmp_path / "r.html").read_text(encoding="utf-8")  # type: ignore[operator]
        assert "Alignment" in html and "Provenance" in html
        assert "rmse" in res._repr_html_()
        assert "mean_bias" in res.summary_text()

    def test_provenance(self, ref_grid: xr.DataArray) -> None:
        res = climval.compare(ref_grid, _candidate_on_other_grid())
        p = res.provenance
        json.dumps(p)
        assert p["environment"]["climval"] == climval.__version__
        assert "xarray" in p["environment"]["libraries"]
        inp = p["inputs"]["candidate"]
        assert inp["kind"] == "dataarray" and inp["shape"] == [30, 36, 72]
        assert len(inp["coordinates_sha256"]) == 64 and len(inp["values_sample_sha256"]) == 64
        steps = [s["step"] for s in p["steps"]]
        assert steps == [
            "inputs",
            "coordinates",
            "target_grid",
            "longitude",
            "region",
            "time",
            "regrid",
            "units",
            "missing",
            "weights",
        ]
        assert p["aligned_values_sha256"]["reference"]
        units = next(s for s in p["steps"] if s["step"] == "units")
        assert units["offset"] == pytest.approx(273.15)
        # stable: same inputs, same hashes
        again = climval.compare(ref_grid, _candidate_on_other_grid())
        assert again.provenance["aligned_values_sha256"] == p["aligned_values_sha256"]
        assert again.provenance["inputs"] == p["inputs"]

    def test_cmip_attributes_recorded(self, ref_grid: xr.DataArray) -> None:
        ds = xr.Dataset(
            {"tas": ref_grid},
            attrs={
                "source_id": "MPI-ESM1-2-HR",
                "experiment_id": "historical",
                "member_id": "r1i1p1f1",
                "tracking_id": "hdl:x",
            },
        )
        res = climval.compare(ref_grid, ds, "tas")
        attrs = res.provenance["inputs"]["candidate"]["global_attributes"]
        assert attrs["source_id"] == "MPI-ESM1-2-HR" and attrs["tracking_id"] == "hdl:x"
        assert res.candidate_name == "MPI-ESM1-2-HR"

    def test_ensemble_member_dimension(self, ref_grid: xr.DataArray) -> None:
        members = xr.concat([ref_grid + 1, ref_grid + 2], dim=pd.Index(["r1", "r2"], name="member"))
        members = members.assign_attrs(ref_grid.attrs)
        res = climval.compare(ref_grid, members, metrics=["mean_bias", "rmse"])
        frame = res.to_dataframe()
        bias = frame[(frame.metric == "mean_bias") & (frame["sample"] == "space_time")]
        values = dict(zip(bias.member, bias.value, strict=True))
        assert values == pytest.approx({"r1": 1.0, "r2": 2.0})

    def test_thresholds(self, ref_grid: xr.DataArray) -> None:
        res = climval.compare(ref_grid, (ref_grid + 0.5).assign_attrs(ref_grid.attrs), by="season")
        ok = res.check({"mean_bias": {"abs_max": 1.0}, "pearson_r": {"min": 0.99, "group": "*"}})
        assert ok.passed and len(ok.checks) == 6
        bad = climval.evaluate_thresholds(res, {"rmse": {"max": 0.1, "sample": "climatology"}})
        assert not bad.passed and "above max" in bad.failures[0].message
        with pytest.raises(ValueError, match="No value"):
            res.check({"nope": {"max": 1}})

    def test_lazy_option(self, ref_grid: xr.DataArray) -> None:
        res = climval.compare(ref_grid, ref_grid, compute=False)
        assert not res.computed and "lazy" in repr(res)
        assert res["rmse"] == 0.0 and res.computed


class TestNoSyntheticFallback:
    def test_compare_needs_data(self) -> None:
        with pytest.raises(InputError, match="does not substitute random numbers"):
            climval.compare(None, None)

    def test_demo_is_flagged(self) -> None:
        res = climval.demo()
        assert res.synthetic and res.provenance["synthetic_data"] is True
        assert "not a validation" in res.summary_text()
        assert res["mean_bias"] == pytest.approx(1.0, abs=0.1)


class TestFiles:
    def test_netcdf_and_zarr_paths(self, ref_grid: xr.DataArray, tmp_path: object) -> None:
        pytest.importorskip("zarr")
        nc = tmp_path / "ref.nc"  # type: ignore[operator]
        zr = tmp_path / "can.zarr"  # type: ignore[operator]
        ref_grid.to_dataset().to_netcdf(nc)
        _candidate_on_other_grid().to_dataset().to_zarr(zr)
        res = climval.compare(str(nc), str(zr), "tas")
        direct = climval.compare(ref_grid, _candidate_on_other_grid(), "tas")
        assert res["rmse"] == pytest.approx(direct["rmse"], rel=1e-12)
        assert res.provenance["inputs"]["reference"]["location"] == str(nc)
        assert res.provenance["inputs"]["reference"]["values_sample_sha256"] is None

    def test_csv_station_series(self, tmp_path: object) -> None:
        path = tmp_path / "station.csv"  # type: ignore[operator]
        times = pd.date_range("2000-01-01", periods=10, freq="D")
        pd.DataFrame({"time": times, "tas": np.arange(10.0)}).to_csv(path, index=False)
        series = xr.DataArray(np.arange(10.0) + 2, dims="time", coords={"time": times}, name="tas")
        res = climval.compare(str(path), series, "tas", units="ignore")
        assert res["mean_bias"] == pytest.approx(2.0)
        assert "climatology" not in res.samples

    def test_missing_file(self) -> None:
        with pytest.raises(InputError, match="not found"):
            climval.compare("does/not/exist.nc", "x.nc")


class TestAccessor:
    def test_dataarray_and_dataset(self, ref_grid: xr.DataArray) -> None:
        res = (ref_grid + 1).assign_attrs(ref_grid.attrs).climval.compare(ref_grid)
        assert res["mean_bias"] == pytest.approx(1.0)
        minus = (ref_grid - 1).assign_attrs(ref_grid.attrs)
        res2 = xr.Dataset({"tas": minus}).climval.compare(ref_grid, "tas")
        assert res2["mean_bias"] == pytest.approx(-1.0)


def test_smooth_helper_shape() -> None:
    assert smooth(np.zeros(2), np.zeros(3), np.ones(4)).shape == (4, 2, 3)
