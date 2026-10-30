"""Metric correctness: hand-computed cases and agreement with xskillscore."""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from climval.metrics import (
    MAE,
    RMSE,
    CenteredRMSE,
    KlingGuptaEfficiency,
    MeanBias,
    NormalizedRMSE,
    PearsonCorrelation,
    PercentileBias,
    SpearmanCorrelation,
    StdRatio,
    TaylorSkillScore,
    get_metric,
)
from climval.metrics.stats import BaseMetric

DIMS3 = ["time", "lat", "lon"]


def _fields(with_nan: bool) -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(3)
    shape = (6, 5, 7)
    coords = {"time": np.arange(6), "lat": np.linspace(-60, 60, 5), "lon": np.arange(7.0)}
    ref = xr.DataArray(rng.normal(280, 5, shape), dims=DIMS3, coords=coords)
    can = ref + rng.normal(1.0, 2.0, shape)
    if with_nan:  # paired masking is part of the contract
        ref[0, 0, 0] = np.nan
        can[2, 3, 4] = np.nan
        valid = ref.notnull() & can.notnull()
        ref, can = ref.where(valid), can.where(valid)
    return ref, can


def _weights(ref: xr.DataArray, dims: list[str], weighted: bool) -> xr.DataArray:
    """Weights over the reduction dims only, the shape xskillscore expects."""
    template = ref.isel({d: 0 for d in ref.dims if d not in dims}, drop=True)
    if weighted and "lat" in dims:
        return np.cos(np.deg2rad(template["lat"])).broadcast_like(template)
    if weighted:
        return xr.DataArray(
            np.linspace(1.0, 2.0, template.size).reshape(template.shape),
            dims=template.dims,
            coords=template.coords,
        )
    return xr.ones_like(template)


class TestAgainstXskillscore:
    @pytest.mark.parametrize("dims", [DIMS3, ["lat", "lon"], ["time"]])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("with_nan", [False, True])
    def test_agreement(self, dims: list[str], weighted: bool, with_nan: bool) -> None:
        xs = pytest.importorskip("xskillscore")
        ref, can = _fields(with_nan)
        xw = _weights(ref, dims, weighted)
        # climval's engine contract: weights broadcast to the data, zero where invalid
        w = (xw.broadcast_like(ref) * (ref.notnull() & can.notnull())).astype(float)
        pairs = [
            (RMSE(), xs.rmse(can, ref, dim=dims, weights=xw, skipna=True)),
            (MAE(), xs.mae(can, ref, dim=dims, weights=xw, skipna=True)),
            (MeanBias(), xs.me(can, ref, dim=dims, weights=xw, skipna=True)),
            (PearsonCorrelation(), xs.pearson_r(ref, can, dim=dims, weights=xw, skipna=True)),
        ]
        if not with_nan:
            # xskillscore 0.0.29 returns NaN for spearman_r whenever a NaN is
            # present even with skipna=True; checked against scipy below.
            pairs.append(
                (SpearmanCorrelation(), xs.spearman_r(ref, can, dim=dims, weights=xw, skipna=True))
            )
        for metric, expected in pairs:
            got = metric.reduce(ref, can, w, dims)
            np.testing.assert_allclose(got.values, expected.values, rtol=1e-10, err_msg=metric.name)

    def test_spearman_with_nan_matches_scipy(self) -> None:
        from scipy.stats import spearmanr

        ref, can = _fields(True)
        w = (ref.notnull() & can.notnull()).astype(float)
        got = SpearmanCorrelation().reduce(ref, can, w, DIMS3)
        ok = np.isfinite(ref.values) & np.isfinite(can.values)
        assert float(got) == pytest.approx(spearmanr(ref.values[ok], can.values[ok])[0], rel=1e-12)


class TestHandComputed:
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    can = np.array([2.0, 2.0, 5.0, 3.0])

    def test_error_metrics(self) -> None:
        d = self.can - self.ref  # [1, 0, 2, -1]
        assert MeanBias().compute(self.ref, self.can) == pytest.approx(0.5)
        assert MAE().compute(self.ref, self.can) == pytest.approx(1.0)
        assert RMSE().compute(self.ref, self.can) == pytest.approx(math.sqrt(6.0 / 4.0))
        crmse = math.sqrt(np.mean((d - d.mean()) ** 2))
        assert CenteredRMSE().compute(self.ref, self.can) == pytest.approx(crmse)

    def test_rmse_decomposition(self) -> None:
        rmse = RMSE().compute(self.ref, self.can)
        bias = MeanBias().compute(self.ref, self.can)
        crmse = CenteredRMSE().compute(self.ref, self.can)
        assert rmse**2 == pytest.approx(bias**2 + crmse**2)

    def test_std_ratio_nrmse(self) -> None:
        sr, sc = np.std(self.ref), np.std(self.can)
        assert StdRatio().compute(self.ref, self.can) == pytest.approx(sc / sr)
        assert NormalizedRMSE().compute(self.ref, self.can) == pytest.approx(math.sqrt(1.5) / sr)

    def test_weighted_by_hand(self) -> None:
        w = np.array([1.0, 1.0, 2.0, 0.0])
        # weights normalised: [0.25, 0.25, 0.5, 0]
        assert MeanBias().compute(self.ref, self.can, w) == pytest.approx(0.25 * 1 + 0.5 * 2)
        assert RMSE().compute(self.ref, self.can, w) == pytest.approx(math.sqrt(0.25 + 0.5 * 4))

    def test_taylor_eq5(self) -> None:
        r = np.corrcoef(self.ref, self.can)[0, 1]
        s = np.std(self.can) / np.std(self.ref)
        expected = 4 * (1 + r) ** 4 / ((s + 1 / s) ** 2 * 2**4)
        assert TaylorSkillScore().compute(self.ref, self.can) == pytest.approx(expected)
        assert TaylorSkillScore().compute(self.ref, self.ref) == pytest.approx(1.0)

    def test_kge(self) -> None:
        r = np.corrcoef(self.ref, self.can)[0, 1]
        a = np.std(self.can) / np.std(self.ref)
        b = np.mean(self.can) / np.mean(self.ref)
        expected = 1 - math.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
        assert KlingGuptaEfficiency().compute(self.ref, self.can) == pytest.approx(expected)
        assert KlingGuptaEfficiency().compute(self.ref, self.ref) == pytest.approx(1.0)

    def test_percentile_matches_numpy(self) -> None:
        x = np.random.default_rng(1).gamma(2.0, 2.0, 501)
        y = x * 1.1
        got = PercentileBias(90).compute(x, y)
        q = np.percentile(x, 90)
        assert got == pytest.approx((np.percentile(y, 90) - q) / q * 100)

    def test_nan_pairs_dropped(self) -> None:
        ref = np.array([1.0, np.nan, 3.0])
        can = np.array([2.0, 5.0, np.nan])
        assert MeanBias().compute(ref, can) == pytest.approx(1.0)

    def test_extra_dims_kept(self) -> None:
        ref = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("member", "lat"))
        can = ref + xr.DataArray([1.0, 2.0], dims="member")
        w = xr.ones_like(ref)
        out = MeanBias().reduce(ref, can, w, ["lat"])
        assert out.dims == ("member",)
        np.testing.assert_allclose(out.values, [1.0, 2.0])


class TestExtension:
    def test_numpy_only_metric_works_in_engine(self) -> None:
        class MaxAbs(BaseMetric):
            name = "max_abs"

            def compute(
                self,
                reference: np.ndarray,
                candidate: np.ndarray,
                weights: np.ndarray | None = None,
            ) -> float:
                return float(np.max(np.abs(candidate - reference)))

        ref = xr.DataArray([[1.0, 2.0], [3.0, np.nan]], dims=("time", "lat"))
        can = xr.DataArray([[1.5, 2.0], [0.0, 9.0]], dims=("time", "lat"))
        valid = ref.notnull() & can.notnull()
        out = MaxAbs().reduce(
            ref.where(valid), can.where(valid), valid.astype(float), ["time", "lat"]
        )
        assert float(out) == 3.0

    def test_registry_aliases(self) -> None:
        assert get_metric("bias").name == "mean_bias"
        assert get_metric("percentile_bias_p99").name == "percentile_bias_p99"
        assert get_metric("spearman_r").name == "spearman_r"

    def test_base_metric_without_implementation(self) -> None:
        with pytest.raises(NotImplementedError):
            BaseMetric().compute(np.zeros(2), np.zeros(2))
