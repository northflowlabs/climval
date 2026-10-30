"""Dask: inputs stay lazy until compute, one pass computes everything, results match NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr
from conftest import make_field, monthly_times

import climval

dask = pytest.importorskip("dask")
import dask.array as dsa  # noqa: E402


class CountingScheduler:
    """Synchronous scheduler that counts graph executions and fails past a limit."""

    def __init__(self, max_computes: int) -> None:
        self.max_computes = max_computes
        self.computes = 0

    def __call__(self, dsk: Any, keys: Any, **kwargs: Any) -> Any:
        self.computes += 1
        if self.computes > self.max_computes:
            raise RuntimeError(f"dask computed {self.computes} times, limit {self.max_computes}")
        return dask.get(dsk, keys, **kwargs)


def _inputs() -> tuple[xr.DataArray, xr.DataArray]:
    ref = make_field(
        np.arange(-87.0, 90.0, 6.0), np.arange(-177.0, 180.0, 6.0), monthly_times("2000-01-01", 24)
    )
    can_lat = np.arange(87.5, -90.0, -5.0)
    can = make_field(
        can_lat,
        np.arange(2.5, 360.0, 5.0),
        monthly_times("1999-07-01", 30, calendar="noleap", day=16),
    )
    can = can + np.random.default_rng(2).normal(0.5, 0.5, can.shape) - 273.15
    can = can.assign_attrs(units="degC")
    can[3, 4, 5] = np.nan
    return ref, can


def _chunked(da: xr.DataArray) -> xr.DataArray:
    return da.chunk({"time": 6, "lat": 12, "lon": 24})


OPTIONS: list[dict[str, Any]] = [
    {},
    {"by": "season", "samples": ["space_time", "climatology", "space", "time"]},
    {"region": climval.Box(lat=(-40, 40), lon=(150, -120)), "freq": "YS"},
    {"metrics": ["spearman_r", "percentile_bias_p90", "kge"]},
]


@pytest.mark.parametrize("options", OPTIONS)
def test_no_compute_before_compute(options: dict[str, Any]) -> None:
    ref, can = _inputs()
    with dask.config.set(scheduler=CountingScheduler(0)):
        res = climval.compare(_chunked(ref), _chunked(can), compute=False, **options)
    assert not res.computed
    assert isinstance(res.aligned.reference.data, dsa.Array)
    assert isinstance(res.aligned.candidate.data, dsa.Array)
    for ds in res.samples.values():
        for var in ds.data_vars.values():
            assert isinstance(var.data, dsa.Array)


@pytest.mark.parametrize("options", OPTIONS)
def test_one_pass_and_matches_numpy(options: dict[str, Any]) -> None:
    ref, can = _inputs()
    eager = climval.compare(ref, can, **options)
    scheduler = CountingScheduler(1)
    with dask.config.set(scheduler=scheduler):
        lazy = climval.compare(_chunked(ref), _chunked(can), **options)
    assert scheduler.computes == 1
    # Ranks of values that tie exactly in theory can flip on 1e-16 differences
    # between NumPy and chunked sums, so rank metrics get a looser tolerance.
    rtol = 1e-6 if "spearman_r" in options.get("metrics", []) else 1e-10
    for sample, ds in eager.samples.items():
        xr.testing.assert_allclose(ds, lazy.samples[sample], rtol=rtol)
    assert eager.provenance["aligned_values_sha256"] == lazy.provenance["aligned_values_sha256"]


def test_zarr_store_opened_lazily(tmp_path: Any) -> None:
    pytest.importorskip("zarr")
    ref, can = _inputs()
    store = tmp_path / "can.zarr"
    _chunked(can).to_dataset().to_zarr(store)
    with dask.config.set(scheduler=CountingScheduler(0)):
        res = climval.compare(ref, str(store), "tas", compute=False)
    assert isinstance(res.aligned.candidate.data, dsa.Array)
    assert res.provenance["inputs"]["candidate"]["lazy"] is True
    assert res.provenance["inputs"]["candidate"]["chunks"]["time"] == [6] * 5
    eager = climval.compare(ref, can)
    assert res.compute()["rmse"] == pytest.approx(eager["rmse"], rel=1e-10)


def test_area_weights_with_dask() -> None:
    ref, can = _inputs()
    area = np.cos(np.deg2rad(ref["lat"])).broadcast_like(ref.isel(time=0, drop=True)) * 3.0
    eager = climval.compare(ref, can, area=area)
    with dask.config.set(scheduler=CountingScheduler(1)):
        lazy = climval.compare(_chunked(ref), _chunked(can), area=area.chunk({"lat": 10}))
    assert lazy["rmse"] == pytest.approx(eager["rmse"], rel=1e-10)
