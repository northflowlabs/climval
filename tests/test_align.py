# SPDX-License-Identifier: Apache-2.0
"""Unit tests for every alignment step."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import make_field, monthly_times

import climval
from climval.align import align
from climval.align.coords import (
    find_coord,
    is_global_lon,
    normalise_coords,
    pad_periodic,
    to_lon_frame,
)
from climval.align.region import Box, as_region, choose_lon_frame, subset_box
from climval.align.temporal import align_time, classify_step
from climval.align.units import affine, canonical, harmonise_units
from climval.align.weights import make_weights
from climval.errors import (
    AlignmentError,
    CurvilinearGridError,
    InputError,
    NoOverlapError,
    UnitsError,
)


def linear(lat: np.ndarray, lon: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Linear in lat and lon, so bilinear interpolation reproduces it exactly."""
    return 280.0 + 0.1 * lat[None, :, None] + 0.01 * lon[None, None, :] + month[:, None, None]


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


class TestCoordinates:
    def test_detects_cf_names_and_renames(self) -> None:
        da = xr.DataArray(
            np.zeros((2, 3, 4)),
            dims=("valid_time", "latitude", "longitude"),
            coords={
                "valid_time": pd.date_range("2000-01-01", periods=2),
                "latitude": [10.0, 0.0, -10.0],
                "longitude": [0.0, 90.0, 180.0, 270.0],
            },
        )
        out, info = normalise_coords(da, "reference")
        assert out.dims == ("time", "lat", "lon")
        assert info.kind == "rectilinear"
        assert info.original_names == {"lat": "latitude", "lon": "longitude", "time": "valid_time"}

    def test_detects_by_attributes(self) -> None:
        da = xr.DataArray(
            np.zeros((3, 2)),
            dims=("y", "x"),
            coords={
                "y": ("y", [1.0, 2.0, 3.0], {"units": "degrees_north"}),
                "x": ("x", [5.0, 6.0], {"standard_name": "longitude"}),
            },
        )
        assert find_coord(da, "lat") == "y"
        assert find_coord(da, "lon") == "x"

    def test_descending_latitude_is_reversed(self) -> None:
        da = xr.DataArray(np.arange(3.0), dims="lat", coords={"lat": [30.0, 0.0, -30.0]})
        out, info = normalise_coords(da, "x")
        assert list(out["lat"].values) == [-30.0, 0.0, 30.0]
        assert list(out.values) == [2.0, 1.0, 0.0]
        assert "latitude reversed to ascending order" in info.changes

    def test_curvilinear_detected(self) -> None:
        lat2d, lon2d = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=("j", "i"),
            coords={"lat": (("j", "i"), lat2d), "lon": (("j", "i"), lon2d)},
        )
        _, info = normalise_coords(da, "candidate")
        assert info.kind == "curvilinear"


class TestLongitude:
    def test_0_360_to_minus180_180(self) -> None:
        da = xr.DataArray(np.arange(4.0), dims="lon", coords={"lon": [0.0, 90.0, 180.0, 270.0]})
        out, changed = to_lon_frame(da, -180.0)
        assert changed
        assert list(out["lon"].values) == [-180.0, -90.0, 0.0, 90.0]
        # value at 270 E is the value at -90
        assert float(out.sel(lon=-90.0)) == 3.0

    def test_duplicate_seam_dropped(self) -> None:
        da = xr.DataArray(
            np.arange(5.0), dims="lon", coords={"lon": [0.0, 90.0, 180.0, 270.0, 360.0]}
        )
        out, _ = to_lon_frame(da, 0.0)
        assert list(out["lon"].values) == [0.0, 90.0, 180.0, 270.0]

    def test_periodic_padding(self) -> None:
        lon = np.arange(0.0, 360.0, 30.0)
        assert is_global_lon(lon)
        assert not is_global_lon(np.arange(0.0, 90.0, 30.0))
        da = xr.DataArray(np.arange(12.0), dims="lon", coords={"lon": lon})
        padded = pad_periodic(da)
        assert padded["lon"].values[0] == -30.0 and padded["lon"].values[-1] == 360.0
        assert float(padded.isel(lon=0)) == 11.0 and float(padded.isel(lon=-1)) == 0.0

    def test_interpolation_across_seam(self) -> None:
        # Source on 0..360 at 10 degree spacing, target at -175 (between 180 and 190 E).
        lon = np.arange(5.0, 360.0, 10.0)
        lat = np.array([-10.0, 10.0])
        times = monthly_times("2000-01-01", 2)
        src = make_field(
            lat,
            lon,
            times,
            func=lambda la, lo, m: np.broadcast_to(
                np.cos(np.deg2rad(lo))[None, None, :], (m.size, la.size, lo.size)
            ).copy(),
        )
        tgt = make_field(
            lat,
            np.array([-179.0, -1.0, 1.0, 179.0]),
            times,
            func=lambda la, lo, m: np.zeros((m.size, la.size, lo.size)),
        )
        pair = align(tgt, src, "tas")
        assert not bool(pair.candidate.isnull().any())


class TestRegion:
    def test_box_crossing_antimeridian(self) -> None:
        box = Box(lat=(-10.0, 10.0), lon=(170.0, -170.0))
        assert box.lon_width == pytest.approx(20.0)
        frame = choose_lon_frame(box, np.array([-180.0, 0.0, 179.0]))
        assert frame == 0.0
        lon = np.arange(-180.0, 180.0, 5.0)
        da = xr.DataArray(np.arange(lon.size, dtype=float), dims="lon", coords={"lon": lon})
        da, _ = to_lon_frame(da, frame)
        sub = subset_box(da, box, frame, "x")
        assert list(sub["lon"].values) == [170.0, 175.0, 180.0, 185.0, 190.0]

    def test_box_west_of_greenwich_in_0_360_grid(self) -> None:
        box = Box(lat=(0.0, 1.0), lon=(-10.0, 10.0))
        frame = choose_lon_frame(box, np.array([0.0, 359.0]))
        assert frame == -180.0
        lon = np.arange(0.0, 360.0, 5.0)
        da = xr.DataArray(np.zeros(lon.size), dims="lon", coords={"lon": lon})
        da, _ = to_lon_frame(da, frame)
        sub = subset_box(da, box, frame, "x")
        assert list(sub["lon"].values) == [-10.0, -5.0, 0.0, 5.0, 10.0]

    def test_region_applied_before_regrid_on_both(self, ref_grid: xr.DataArray) -> None:
        can = ref_grid.assign_coords(lon=np.mod(ref_grid["lon"], 360.0)).sortby("lon")
        pair = align(ref_grid, can, region=Box(lat=(-30, 30), lon=(150, -150)))
        assert (
            float(pair.reference["lat"].min()) >= -30 and float(pair.reference["lat"].max()) <= 30
        )
        lon = pair.reference["lon"].values
        assert lon.min() >= 150 and lon.max() <= 210
        xr.testing.assert_allclose(pair.reference, pair.candidate)

    def test_dict_and_legacy_domain(self) -> None:
        from climval.models.schema import SpatialDomain

        assert as_region({"lat": (0, 10), "lon": (5, 6)}) == Box(lat=(0.0, 10.0), lon=(5.0, 6.0))
        box = as_region(SpatialDomain(lat_min=1, lat_max=2, lon_min=-180, lon_max=180))
        assert isinstance(box, Box) and box.lon is None

    def test_empty_region_raises(self, ref_grid: xr.DataArray) -> None:
        with pytest.raises(NoOverlapError):
            align(ref_grid, ref_grid, region=Box(lat=(89.5, 89.9)))


class TestRegionmask:
    def test_ar6_region(self, ref_grid: xr.DataArray) -> None:
        pytest.importorskip("regionmask")
        pair = align(ref_grid, ref_grid, region="ar6.land:NEU")
        step = next(s for s in pair.steps if s["step"] == "region_mask")
        assert step["region"] == "N.Europe"
        assert step["cells_inside"] > 0
        assert bool(pair.reference.isnull().any())


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------


class TestRegrid:
    def test_bilinear_exact_for_linear_field(self) -> None:
        times = monthly_times("2000-01-01", 3)
        ref = make_field(np.arange(-40.0, 41.0, 10.0), np.arange(-40.0, 41.0, 10.0), times, linear)
        can = make_field(np.arange(-45.0, 46.0, 3.0), np.arange(-45.0, 46.0, 3.0), times, linear)
        pair = align(ref, can)
        np.testing.assert_allclose(pair.candidate.values, pair.reference.values, rtol=0, atol=1e-9)

    def test_no_extrapolation(self) -> None:
        times = monthly_times("2000-01-01", 2)
        ref = make_field(np.arange(-60.0, 61.0, 10.0), np.arange(0.0, 50.0, 10.0), times, linear)
        can = make_field(np.arange(-30.0, 31.0, 10.0), np.arange(0.0, 50.0, 10.0), times, linear)
        pair = align(ref, can)
        outside = (pair.reference["lat"] < -30) | (pair.reference["lat"] > 30)
        assert bool(pair.candidate.where(outside).isnull().all())
        assert bool(pair.reference.where(outside).isnull().all())  # paired mask

    def test_gap_in_source_is_not_bridged(self) -> None:
        # A regional candidate around the antimeridian, given in -180..180, has
        # a hole from -170 to 170; a global target must not be filled across it.
        times = monthly_times("2000-01-01", 2)
        lon_can = np.concatenate([np.arange(-180.0, -169.0, 2.0), np.arange(170.0, 180.0, 2.0)])
        can = make_field(np.array([-10.0, 10.0]), lon_can, times, linear)
        ref = make_field(np.array([0.0]), np.arange(-179.0, 180.0, 2.0), times, linear)
        pair = align(ref, can, target="reference")
        lon = pair.candidate["lon"]
        inside = (lon <= -170) | (lon >= 170)
        assert bool(pair.candidate.where(~inside).isnull().all())
        assert bool(pair.candidate.where(inside).notnull().sum() > 0)

    def test_coincident_grid_selects(self, ref_grid: xr.DataArray) -> None:
        pair = align(ref_grid, ref_grid.isel(lat=slice(None, None, -1)))
        step = next(s for s in pair.steps if s["step"] == "regrid")
        assert "nothing interpolated" in step["action"]

    def test_nearest(self) -> None:
        times = monthly_times("2000-01-01", 2)
        ref = make_field(np.array([0.0, 10.0]), np.array([0.0, 10.0]), times, linear)
        can = make_field(np.array([1.0, 9.0]), np.array([1.0, 9.0]), times, linear)
        pair = align(can, ref, regrid="nearest")
        # target 1.0 picks source 0.0, target 9.0 picks source 10.0
        expected = ref.sel(lat=[0.0, 10.0], lon=[0.0, 10.0]).values
        np.testing.assert_allclose(pair.candidate.values, expected)

    def test_target_candidate_and_coarser(self) -> None:
        times = monthly_times("2000-01-01", 2)
        fine = make_field(np.arange(-40.0, 41.0, 2.0), np.arange(0.0, 41.0, 2.0), times, linear)
        coarse = make_field(np.arange(-40.0, 41.0, 10.0), np.arange(0.0, 41.0, 10.0), times, linear)
        assert align(fine, coarse, target="candidate").reference.sizes["lat"] == 9
        assert align(fine, coarse, target="coarser").reference.sizes["lat"] == 9
        assert align(coarse, fine, target="coarser").reference.sizes["lat"] == 9
        grid = xr.Dataset(coords={"lat": [0.0, 5.0], "lon": [10.0, 20.0]})
        pair = align(fine, coarse, target=grid)
        assert pair.reference.sizes == {"time": 2, "lat": 2, "lon": 2}

    def test_curvilinear_needs_conservative(self, ref_grid: xr.DataArray) -> None:
        lat2d, lon2d = np.meshgrid(
            np.arange(-40.0, 41.0, 20.0), np.arange(0.0, 61.0, 20.0), indexing="ij"
        )
        curvi = xr.DataArray(
            np.zeros((1, 5, 4)),
            dims=("time", "j", "i"),
            coords={
                "time": ref_grid["time"].values[:1],
                "lat": (("j", "i"), lat2d),
                "lon": (("j", "i"), lon2d),
            },
            name="tas",
            attrs={"units": "K"},
        )
        with pytest.raises(CurvilinearGridError, match="regrid='conservative'"):
            align(ref_grid, curvi)
        with pytest.raises(CurvilinearGridError, match="cannot be the target grid"):
            align(curvi, ref_grid, regrid="conservative", target="reference")

    def test_conservative_with_real_xesmf(self, ref_grid: xr.DataArray) -> None:
        pytest.importorskip("xesmf")
        const = xr.full_like(ref_grid, 280.0)
        coarse = const.isel(lat=slice(0, None, 2), lon=slice(0, None, 2))
        pair = align(const, coarse, regrid="conservative")
        inner = pair.candidate.isel(lat=slice(2, -2)).values
        np.testing.assert_allclose(inner[np.isfinite(inner)], 280.0, rtol=1e-10)
        assert np.isfinite(inner).mean() > 0.9

    def test_conservative_without_xesmf(
        self, ref_grid: xr.DataArray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "xesmf", None)
        from climval.errors import MissingDependencyError

        with pytest.raises(MissingDependencyError, match="xESMF"):
            align(ref_grid, ref_grid.isel(lat=slice(0, -1, 2)), regrid="conservative")

    def test_conservative_calls_xesmf(
        self, ref_grid: xr.DataArray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}

        class FakeRegridder:
            def __init__(self, src: xr.Dataset, tgt: xr.Dataset, method: str, **kw: object) -> None:
                calls.update(method=method, kw=kw, lat_b=tgt["lat_b"].values)
                self.tgt = tgt

            def __call__(self, da: xr.DataArray, **kw: object) -> xr.DataArray:
                calls["call_kw"] = kw
                return da.interp(lat=self.tgt["lat"], lon=self.tgt["lon"])

        monkeypatch.setitem(sys.modules, "xesmf", types.SimpleNamespace(Regridder=FakeRegridder))
        pair = align(ref_grid, ref_grid.isel(lat=slice(0, -1, 2)), regrid="conservative")
        assert calls["method"] == "conservative_normed"
        # ref_grid is global: xESMF is told so, and gets no wrapped seam columns
        assert calls["kw"] == {"unmapped_to_nan": True, "periodic": True}
        lat_b = np.asarray(calls["lat_b"])
        assert lat_b.min() >= -90 and lat_b.max() <= 90
        assert pair.candidate.sizes == pair.reference.sizes


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def _series(times: object, values: np.ndarray | None = None, units: str = "K") -> xr.DataArray:
    t = xr.DataArray(times, dims="time")
    v = np.arange(t.size, dtype=float) if values is None else values
    return xr.DataArray(v, dims="time", coords={"time": t.values}, name="x", attrs={"units": units})


class TestTime:
    @pytest.mark.parametrize(
        "calendar", ["noleap", "360_day", "all_leap", "proleptic_gregorian", "julian"]
    )
    def test_monthly_calendars_match_by_label(self, calendar: str) -> None:
        ref = _series(monthly_times("2000-01-01", 24))
        can = _series(monthly_times("2000-07-01", 24, calendar=calendar, day=16))
        r, c, rec = align_time(ref, can)
        assert rec["matched_periods"] == 18
        assert rec["calendars"]["candidate"] in (calendar, "proleptic_gregorian")
        # candidate value for 2000-07 is index 0
        assert float(c.isel(time=0)) == 0.0 and float(r.isel(time=0)) == 6.0
        np.testing.assert_array_equal(c["time"].values, r["time"].values)

    def test_daily_360_day_vs_gregorian(self) -> None:
        ref = _series(pd.date_range("2000-01-01", "2000-03-31", freq="D"))
        can = _series(
            xr.date_range("2000-01-01", "2000-03-30", freq="D", calendar="360_day", use_cftime=True)
        )
        r, c, rec = align_time(ref, can)
        assert rec["frequency"] == "daily"
        # Jan 1..30, Feb 1..29 (360_day has Feb 29 and 30, gregorian 2000 has 29), Mar 1..30
        assert rec["matched_periods"] == 30 + 29 + 30
        assert rec["unmatched_dropped"] == {"reference": 2, "candidate": 1}

    def test_daily_to_monthly_means(self) -> None:
        ref = _series(pd.date_range("2000-01-01", "2000-03-31", freq="D"))
        can = _series(
            xr.date_range("2000-01-01", "2000-03-30", freq="D", calendar="360_day", use_cftime=True)
        )
        r, c, rec = align_time(ref, can, freq="MS")
        assert rec["matched_periods"] == 3
        assert float(r.isel(time=0)) == pytest.approx(15.0)  # mean of 0..30
        assert float(c.isel(time=1)) == pytest.approx(44.5)  # mean of 30..59 in 360_day

    def test_daily_vs_monthly_aggregates_daily(self) -> None:
        daily = _series(pd.date_range("2000-01-01", "2000-12-31", freq="D"))
        monthly = _series(monthly_times("2000-01-01", 12, calendar="noleap"))
        r, c, rec = align_time(monthly, daily)
        assert rec["frequency"] == "monthly"
        assert rec["candidate_resampled"]["aggregation"] == "mean"
        assert rec["matched_periods"] == 12

    def test_incomplete_period_dropped(self) -> None:
        ref = _series(monthly_times("2000-01-01", 3))
        can = _series(pd.date_range("2000-01-10", "2000-03-31", freq="D"))
        r, c, rec = align_time(ref, can)
        assert rec["matched_periods"] == 2
        assert len(rec["candidate_resampled"]["incomplete_periods_dropped"]) == 1

    def test_monthly_to_yearly_weighted_by_month_length(self) -> None:
        values = np.zeros(12)
        values[1] = 29.0  # February 2000 (29 days)
        ref = _series(monthly_times("2000-01-01", 12), values)
        r, _, rec = align_time(ref, ref.copy(), freq="YS")
        assert float(r.isel(time=0)) == pytest.approx(29.0 * 29.0 / 366.0)
        assert rec["reference_resampled"]["aggregation"] == "month-length weighted mean"

    def test_period_on_calendar_fields(self) -> None:
        ref = _series(monthly_times("2000-01-01", 36))
        can = _series(monthly_times("2000-01-01", 36, calendar="360_day"))
        r, c, rec = align_time(ref, can, period=("2001", "2001-06"))
        assert rec["matched_periods"] == 6
        assert int(r["time"].dt.year[0]) == 2001

    def test_no_overlap_raises(self) -> None:
        ref = _series(monthly_times("2000-01-01", 12))
        can = _series(monthly_times("2005-01-01", 12, calendar="noleap"))
        with pytest.raises(NoOverlapError, match="share no time period"):
            align_time(ref, can)

    def test_finer_freq_refused(self) -> None:
        ref = _series(monthly_times("2000-01-01", 12))
        with pytest.raises(AlignmentError, match="does not disaggregate"):
            align_time(ref, ref.copy(), freq="D")

    def test_only_one_has_time(self) -> None:
        ref = _series(monthly_times("2000-01-01", 3))
        with pytest.raises(AlignmentError, match="Only the reference"):
            align_time(ref, ref.isel(time=0, drop=True))

    def test_classify(self) -> None:
        assert classify_step(1.0 / 24.0) == "1h"
        assert classify_step(0.125) == "3h"
        assert classify_step(0.25) == "6h"
        assert classify_step(1.0) == "D"
        assert classify_step(30.0) == "MS"
        assert classify_step(365.0) == "YS"
        assert classify_step(5.0) is None


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class TestUnits:
    @pytest.mark.parametrize(
        "src,tgt,x,expected",
        [
            ("degC", "K", 0.0, 273.15),
            ("K", "degC", 300.0, 26.85),
            ("kg m-2 s-1", "mm/day", 1.0 / 86400.0, 1.0),
            ("mm day-1", "kg m-2 s-1", 86400.0, 1.0),
            ("Pa", "hPa", 101325.0, 1013.25),
            ("1", "%", 0.25, 25.0),
            ("%", "1", 50.0, 0.5),
            ("degF", "degC", 212.0, 100.0),
        ],
    )
    def test_builtin_table(self, src: str, tgt: str, x: float, expected: float) -> None:
        a, b, how = affine(src, tgt)
        assert how == "built-in table"
        assert a * x + b == pytest.approx(expected)

    def test_aliases(self) -> None:
        assert canonical("degrees_Celsius") == "degC"
        assert canonical("kg m**-2 s**-1") == "kg m-2 s-1"
        assert canonical("mbar") == "hPa"
        assert canonical("furlongs") is None

    def test_convert_candidate_lazily(self) -> None:
        ref = _series(monthly_times("2000-01-01", 2), np.array([273.15, 274.15]))
        can = _series(monthly_times("2000-01-01", 2), np.array([0.0, 1.0]), units="degC")
        r, c, rec = harmonise_units(ref, can)
        np.testing.assert_allclose(c.values, ref.values)
        assert c.attrs["units"] == "K"
        assert rec["offset"] == pytest.approx(273.15)

    def test_incompatible_raises(self) -> None:
        ref = _series(monthly_times("2000-01-01", 1))
        with pytest.raises(UnitsError, match="Cannot convert"):
            harmonise_units(ref, ref.assign_attrs(units="hPa"))

    def test_missing_units_raises_unless_declared(self) -> None:
        ref = _series(monthly_times("2000-01-01", 1))
        can = ref.copy()
        can.attrs = {}
        with pytest.raises(UnitsError, match="no units attribute"):
            harmonise_units(ref, can)
        _, c, _ = harmonise_units(ref, can - 273.15, {"candidate": "degC"})
        np.testing.assert_allclose(c.values, ref.values)
        _, _, rec = harmonise_units(ref, can, "ignore")
        assert "not checked" in rec["action"]

    def test_strict(self) -> None:
        ref = _series(monthly_times("2000-01-01", 1))
        with pytest.raises(UnitsError, match="strict"):
            harmonise_units(ref, ref.assign_attrs(units="degC"), "strict")

    def test_pint_fallback(self) -> None:
        pytest.importorskip("pint")
        a, b, how = affine("km", "m")
        assert how == "pint" and a == pytest.approx(1000.0) and b == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Missing values and weights
# ---------------------------------------------------------------------------


class TestMissingAndWeights:
    def test_paired_mask(self, ref_grid: xr.DataArray) -> None:
        can = ref_grid.copy()
        can[0, 0, 0] = np.nan
        ref = ref_grid.copy()
        ref[1, 2, 3] = np.nan
        pair = align(ref, can)
        assert bool(np.isnan(pair.reference[0, 0, 0])) and bool(np.isnan(pair.candidate[1, 2, 3]))
        assert int(pair.valid.sum()) == ref_grid.size - 2

    def test_cell_area_default_is_cos_lat_on_uniform_grid(self, ref_grid: xr.DataArray) -> None:
        pair = align(ref_grid, ref_grid)
        assert pair.weights is not None
        w = pair.weights.isel(lon=0).values
        cos = np.cos(np.deg2rad(ref_grid["lat"].values))
        np.testing.assert_allclose(w / w.sum(), cos / cos.sum(), rtol=1e-12)
        explicit = align(ref_grid, ref_grid, weights="cos_lat")
        assert explicit.weights is not None
        np.testing.assert_allclose(explicit.weights.values, cos)

    def test_area_from_cell_measures(self, ref_grid: xr.DataArray) -> None:
        area = xr.DataArray(
            np.full((ref_grid.sizes["lat"], ref_grid.sizes["lon"]), 2.0),
            dims=("lat", "lon"),
            coords={"lat": ref_grid["lat"], "lon": ref_grid["lon"]},
            name="areacella",
        )
        ds = xr.Dataset(
            {"tas": ref_grid.assign_attrs(cell_measures="area: areacella"), "areacella": area}
        )
        pair = align(ds, ref_grid, "tas")
        assert pair.weights is not None and float(pair.weights.mean()) == 2.0
        step = next(s for s in pair.steps if s["step"] == "weights")
        assert step["weights"] == "cell area (areacella)"

    def test_area_follows_region_and_lon_frame(self, ref_grid: xr.DataArray) -> None:
        lon360 = np.mod(ref_grid["lon"].values, 360.0)
        area = xr.DataArray(
            np.broadcast_to(lon360, (ref_grid.sizes["lat"], lon360.size)).copy(),
            dims=("lat", "lon"),
            coords={"lat": ref_grid["lat"].values, "lon": lon360},
        ).sortby("lon")
        pair = align(ref_grid, ref_grid, area=area, region=Box(lat=(-10, 10), lon=(-30, 30)))
        assert pair.weights is not None
        np.testing.assert_allclose(
            pair.weights.isel(lat=0).values, np.mod(pair.reference["lon"].values, 360.0)
        )

    def test_uniform_and_custom(self, ref_grid: xr.DataArray) -> None:
        w, rec = make_weights(None, ref_grid, None)
        assert w is None and rec["weights"] == "uniform"
        with pytest.raises(AlignmentError):
            make_weights(xr.DataArray([1.0], dims="lat", coords={"lat": [999.0]}), ref_grid, None)


class TestInputs:
    def test_none_input_raises_clear_error(self, ref_grid: xr.DataArray) -> None:
        with pytest.raises(InputError, match="climval.demo"):
            climval.compare(ref_grid, None)

    def test_dataset_needs_variable(self, ref_grid: xr.DataArray) -> None:
        ds = xr.Dataset({"tas": ref_grid, "pr": ref_grid})
        with pytest.raises(InputError, match="pass variable="):
            align(ds, ds)
        pair = align(ds, ds, ("tas", "pr"))
        assert pair.variable == "tas"

    def test_extra_dims_must_match(self, ref_grid: xr.DataArray) -> None:
        a = ref_grid.expand_dims(plev=[850.0])
        b = ref_grid.expand_dims(plev=[500.0])
        with pytest.raises(AlignmentError, match="plev"):
            align(a, b)
