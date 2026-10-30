"""
climval.metrics
===============
Weighted statistics comparing a candidate with a reference.

Each metric is a :class:`BaseMetric` with two entry points:

``reduce(reference, candidate, weights, dims) -> xr.DataArray``
    The engine entry point. Inputs are aligned DataArrays (NaN where either
    input is missing) and weights that are zero at those points. The
    reduction runs over ``dims`` with xarray operations, so dask arrays stay
    lazy and other dimensions (an ensemble member, say) are kept.

``compute(reference, candidate, weights=None) -> float``
    The 0.1 NumPy interface, kept for existing code. It flattens both arrays,
    drops points where either is not finite and calls ``reduce``.

Conventions used by every built-in metric:

* differences are candidate minus reference;
* means, variances and covariances are weighted, with weights normalised
  to sum to one over the valid points;
* standard deviations are population values (ddof=0), which is what Taylor
  (2001) uses.

To add a metric, subclass :class:`BaseMetric` and implement ``reduce`` (works
on dask) or only ``compute`` (NumPy; climval then gathers each sample into
one chunk and applies it point set by point set).

References
----------
- Gleckler, Taylor and Doutriaux (2008), J. Geophys. Res., 113, D06104.
- Taylor (2001), J. Geophys. Res., 106(D7), 7183-7192.
- Gupta, Kling, Yilmaz and Martinez (2009), J. Hydrol., 377, 80-91.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Weighted building blocks
# ---------------------------------------------------------------------------


def _total(w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    tot = w.sum(dims)
    out: xr.DataArray = tot.where(tot > 0)
    return out


def wmean(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """Weighted mean over ``dims``; points with zero weight do not count."""
    out: xr.DataArray = (x.fillna(0.0) * w).sum(dims) / _total(w, dims)
    return out


def wvar(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    m = wmean(x, w, dims)
    return wmean((x - m) ** 2, w, dims)


def wstd(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    return np.sqrt(wvar(x, w, dims))  # type: ignore[return-value]


def wcov(a: xr.DataArray, b: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    return wmean((a - wmean(a, w, dims)) * (b - wmean(b, w, dims)), w, dims)


def wcorr(a: xr.DataArray, b: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    sa, sb = wstd(a, w, dims), wstd(b, w, dims)
    out: xr.DataArray = (wcov(a, b, w, dims) / (sa * sb)).where((sa > 0) & (sb > 0))
    return out


def _gather(x: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """Put ``dims`` in a single dask chunk (no-op for NumPy arrays)."""
    if x.chunks is None:
        return x
    out: xr.DataArray = x.chunk({d: -1 for d in dims})
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseMetric:
    """Base for all metrics.

    Attributes
    ----------
    name : str
        Key used in results and in the registry.
    higher_is_better : bool
    units_kind : str or None
        "data" (units of the variable), "dimensionless", "percent", or None.
    needs_full_sample : bool
        True for rank and quantile metrics, which gather each sample into one
        dask chunk when they are computed.
    """

    name: str = "base"
    units_suffix: str = ""
    higher_is_better: bool = False
    units_kind: str | None = None
    needs_full_sample: bool = False

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        """Reduce over ``dims``. Default: apply the NumPy ``compute`` per sample."""
        if type(self).compute is BaseMetric.compute:
            raise NotImplementedError(f"{type(self).__name__} must implement reduce or compute")
        dims = list(dims)
        stack = "_climval_sample"
        r = _gather(reference.stack({stack: dims}), [stack])
        c = _gather(candidate.stack({stack: dims}), [stack])
        w = _gather(weights.broadcast_like(reference).stack({stack: dims}), [stack])

        def _apply(rv: np.ndarray, cv: np.ndarray, wv: np.ndarray) -> float:
            ok = np.isfinite(rv) & np.isfinite(cv) & (wv > 0)
            if not ok.any():
                return float("nan")
            return float(self.compute(rv[ok], cv[ok], wv[ok]))

        out: xr.DataArray = xr.apply_ufunc(
            _apply,
            r,
            c,
            w,
            input_core_dims=[[stack], [stack], [stack]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return out

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Value for two NumPy arrays (flattened; non-finite pairs dropped)."""
        if type(self).reduce is BaseMetric.reduce:
            raise NotImplementedError(f"{type(self).__name__} must implement reduce or compute")
        r = np.asarray(reference, dtype=float).ravel()
        c = np.asarray(candidate, dtype=float).ravel()
        if r.shape != c.shape:
            raise ValueError(f"reference and candidate differ in size: {r.size} and {c.size}")
        w = (
            np.ones_like(r)
            if weights is None
            else np.broadcast_to(np.asarray(weights, dtype=float), np.shape(reference)).ravel()
        )
        ok = np.isfinite(r) & np.isfinite(c) & np.isfinite(w)
        w = np.where(ok, w, 0.0)
        dims = ("sample",)
        value = self.reduce(
            xr.DataArray(np.where(ok, r, np.nan), dims=dims),
            xr.DataArray(np.where(ok, c, np.nan), dims=dims),
            xr.DataArray(w, dims=dims),
            ["sample"],
        )
        return float(value)

    def description(self) -> str:
        return (self.__doc__ or "").strip()

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "higher_is_better": self.higher_is_better,
            "units_kind": self.units_kind,
            "description": self.description().splitlines()[0] if self.description() else "",
        }


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


class MeanBias(BaseMetric):
    """Mean bias (mean error): weighted mean of candidate minus reference."""

    name = "mean_bias"
    higher_is_better = False
    units_kind = "data"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        return wmean(candidate - reference, weights, dims)


class MAE(BaseMetric):
    """Mean absolute error: weighted mean of |candidate minus reference|."""

    name = "mae"
    higher_is_better = False
    units_kind = "data"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        return wmean(abs(candidate - reference), weights, dims)


class RMSE(BaseMetric):
    """Root mean square error: square root of the weighted mean squared difference."""

    name = "rmse"
    higher_is_better = False
    units_kind = "data"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        return np.sqrt(wmean((candidate - reference) ** 2, weights, dims))  # type: ignore[return-value]


class CenteredRMSE(BaseMetric):
    """Centred RMSE: RMSE after removing each field's weighted mean (Taylor 2001, eq. 2).

    It measures pattern and variability error without the mean bias;
    rmse**2 = mean_bias**2 + crmse**2.
    """

    name = "crmse"
    higher_is_better = False
    units_kind = "data"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        ra = reference - wmean(reference, weights, dims)
        ca = candidate - wmean(candidate, weights, dims)
        return np.sqrt(wmean((ca - ra) ** 2, weights, dims))  # type: ignore[return-value]


class NormalizedRMSE(BaseMetric):
    """Normalised RMSE: RMSE divided by the weighted standard deviation of the reference.

    Values below 1 mean the error is smaller than the reference's variability
    over the same sample. NaN when the reference is constant.
    """

    name = "nrmse"
    higher_is_better = False
    units_kind = "dimensionless"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        sr = wstd(reference, weights, dims)
        rmse = RMSE().reduce(reference, candidate, weights, dims)
        out: xr.DataArray = (rmse / sr).where(sr > 0)
        return out


# ---------------------------------------------------------------------------
# Correlation and variability
# ---------------------------------------------------------------------------


class PearsonCorrelation(BaseMetric):
    """Pearson correlation: weighted covariance over the product of weighted standard deviations."""

    name = "pearson_r"
    higher_is_better = True
    units_kind = "dimensionless"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        return wcorr(reference, candidate, weights, dims)


def _rank(x: xr.DataArray, dim: str) -> xr.DataArray:
    from scipy.stats import rankdata

    def _r(v: np.ndarray) -> np.ndarray:
        out = np.full(v.shape, np.nan)
        ok = np.isfinite(v)
        if ok.any():
            out[ok] = rankdata(v[ok])
        return out

    ranked: xr.DataArray = xr.apply_ufunc(
        _r,
        x,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return ranked


class SpearmanCorrelation(BaseMetric):
    """Spearman rank correlation: weighted Pearson correlation of the ranks.

    Ranks are unweighted (ties get the average rank) and computed over the
    valid points of the sample; the correlation of the ranks uses the
    weights. This matches xskillscore.spearman_r.
    """

    name = "spearman_r"
    higher_is_better = True
    units_kind = "dimensionless"
    needs_full_sample = True

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        stack = "_climval_sample"
        r = _gather(reference.stack({stack: list(dims)}), [stack])
        c = _gather(candidate.stack({stack: list(dims)}), [stack])
        w = weights.broadcast_like(reference).stack({stack: list(dims)})
        return wcorr(_rank(r, stack), _rank(c, stack), w, [stack])


class StdRatio(BaseMetric):
    """Standard deviation ratio: weighted std of the candidate over that of the reference."""

    name = "std_ratio"
    higher_is_better = False  # best value is 1
    units_kind = "dimensionless"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        sr = wstd(reference, weights, dims)
        out: xr.DataArray = (wstd(candidate, weights, dims) / sr).where(sr > 0)
        return out


# ---------------------------------------------------------------------------
# Skill scores
# ---------------------------------------------------------------------------


class TaylorSkillScore(BaseMetric):
    """Taylor skill score, Taylor (2001) eq. 5.

    S = 4 (1 + R)**4 / ((s + 1/s)**2 (1 + R0)**4), with R the weighted Pearson
    correlation, s the standard deviation ratio (candidate over reference)
    and R0 the highest attainable correlation (default 1). S is 1 for a
    perfect pattern and tends to 0 as the pattern degrades.
    """

    name = "taylor_skill_score"
    higher_is_better = True
    units_kind = "dimensionless"

    def __init__(self, r0: float = 1.0) -> None:
        self.r0 = r0

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        r = wcorr(reference, candidate, weights, dims)
        sr, sc = wstd(reference, weights, dims), wstd(candidate, weights, dims)
        s = sc / sr
        score = 4.0 * (1.0 + r) ** 4 / ((s + 1.0 / s) ** 2 * (1.0 + self.r0) ** 4)
        out: xr.DataArray = score.where((sr > 0) & (sc > 0))
        return out


class KlingGuptaEfficiency(BaseMetric):
    """Kling-Gupta efficiency, Gupta et al. (2009) eq. 9.

    KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2), with r the
    weighted Pearson correlation, alpha the standard deviation ratio and beta
    the ratio of weighted means (candidate over reference). beta depends on
    the zero of the unit scale, so KGE is meaningful for ratio-scale
    variables such as precipitation or runoff, not for temperature in degC.
    """

    name = "kge"
    higher_is_better = True
    units_kind = "dimensionless"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        r = wcorr(reference, candidate, weights, dims)
        sr = wstd(reference, weights, dims)
        mr = wmean(reference, weights, dims)
        alpha = wstd(candidate, weights, dims) / sr
        beta = wmean(candidate, weights, dims) / mr
        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        return kge.where((sr > 0) & (mr != 0))  # type: ignore[no-any-return]


class PercentileBias(BaseMetric):
    """Percentile bias: relative difference of a weighted percentile, in percent.

    100 * (Q_c - Q_r) / |Q_r|, with Q the weighted percentile of each sample
    (xarray weighted quantile, which equals NumPy's linear method for equal
    weights). NaN when Q_r is 0. Being relative to the reference value, it
    depends on the unit's zero: use Kelvin, not degC, for temperature.
    """

    name = "percentile_bias"
    higher_is_better = False
    units_kind = "percent"
    needs_full_sample = True

    def __init__(self, percentile: float = 95.0) -> None:
        self.percentile = percentile
        self.name = f"percentile_bias_p{int(percentile)}"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        q = self.percentile / 100.0
        w = _gather(weights.broadcast_like(reference), dims)
        qr = _gather(reference, dims).weighted(w).quantile(q, dim=list(dims))
        qc = _gather(candidate, dims).weighted(w).quantile(q, dim=list(dims))
        qr = qr.drop_vars("quantile", errors="ignore")
        qc = qc.drop_vars("quantile", errors="ignore")
        out: xr.DataArray = ((qc - qr) / abs(qr) * 100.0).where(qr != 0)
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEFAULT_METRICS: list[BaseMetric] = [
    RMSE(),
    MAE(),
    MeanBias(),
    NormalizedRMSE(),
    PearsonCorrelation(),
    TaylorSkillScore(),
    PercentileBias(percentile=95.0),
    PercentileBias(percentile=5.0),
    CenteredRMSE(),
    StdRatio(),
    KlingGuptaEfficiency(),
]
"""Metrics computed when none are named. Spearman r is available by name."""

METRIC_REGISTRY: dict[str, BaseMetric] = {m.name: m for m in DEFAULT_METRICS}
METRIC_REGISTRY["spearman_r"] = SpearmanCorrelation()

_ALIASES = {
    "bias": "mean_bias",
    "me": "mean_bias",
    "tss": "taylor_skill_score",
    "pearson": "pearson_r",
    "spearman": "spearman_r",
    "centred_rmse": "crmse",
    "centered_rmse": "crmse",
}


def get_metric(name: str) -> BaseMetric:
    """Metric by name. 'percentile_bias_p<N>' builds a percentile bias for any N."""
    key = _ALIASES.get(name, name)
    if key in METRIC_REGISTRY:
        return METRIC_REGISTRY[key]
    if key.startswith("percentile_bias_p") and key[len("percentile_bias_p") :].isdigit():
        return PercentileBias(float(key[len("percentile_bias_p") :]))
    raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}")


def resolve_metrics(metrics: Sequence[str | BaseMetric] | None) -> list[BaseMetric]:
    if metrics is None:
        return list(DEFAULT_METRICS)
    out = [get_metric(m) if isinstance(m, str) else m for m in metrics]
    for m in out:
        if not isinstance(m, BaseMetric):
            raise TypeError(f"Metrics must be names or BaseMetric instances, got {m!r}")
    names = [m.name for m in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Metric names must be unique, got {names}")
    return out
