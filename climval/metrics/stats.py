# SPDX-License-Identifier: Apache-2.0
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
    drops points where either input or the weight is not finite, rejects
    negative weights and calls ``reduce``. Since 0.2.0 weights are
    normalised to sum to one over the valid points (see the CHANGELOG).

Conventions used by every built-in metric:

* differences are candidate minus reference;
* means, variances and covariances are weighted, with weights normalised
  to sum to one over the valid points;
* standard deviations are population values (ddof=0), which is what Taylor
  (2001) uses.

Memory on dask arrays: every metric except Spearman r, percentile bias and
metrics that implement only ``compute`` is a chunk-wise reduction. Spearman r
and ``compute``-only metrics gather each sample into one chunk (the whole
space-time sample for ``samples=("space_time",)``), so their memory grows with
the sample. Percentile bias gathers a sample only when it fits in dask's
``array.chunk-size`` (128 MiB by default) and otherwise uses a one-pass
histogram estimate; see :class:`PercentileBias`.

Kling-Gupta efficiency and percentile bias divide by the reference's mean or
percentile, so they depend on where the unit scale has its zero.
:func:`climval.compare` computes them on an absolute scale: temperatures in
degC or degF are shifted to K (or Rankine, which gives the same ratios)
for these two metrics only. ``BaseMetric.compute`` has no units and applies
no shift.

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

import math
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import xarray as xr

from climval._chunking import chunk_limit_bytes

# ---------------------------------------------------------------------------
# Weighted building blocks
# ---------------------------------------------------------------------------


CONSTANT_RTOL = 1e-10
"""A sample counts as constant when its weighted standard deviation is at most
this fraction of its weighted mean absolute value. The two-pass standard
deviation of a constant field is not exactly zero (about 1e-16 of the value),
so a test for ``std > 0`` would divide by rounding noise."""

_MEMO: dict[tuple[Any, ...], tuple[Any, tuple[Any, ...]]] | None = None


@contextmanager
def shared_reductions() -> Iterator[None]:
    """Reuse weighted means, standard deviations and covariances inside the block.

    The metrics of one sample share their inputs, so the weighted mean of the
    reference, say, is built once and not once per metric. This keeps the dask
    graph, and the time to build it, small when many metrics and groups are
    evaluated.
    """
    global _MEMO
    previous = _MEMO
    _MEMO = {}
    try:
        yield
    finally:
        _MEMO = previous


def _memo(
    kind: str, fn: Callable[[], xr.DataArray], *objs: Any, dims: Sequence[str]
) -> xr.DataArray:
    if _MEMO is None:
        return fn()
    key = (kind, *(id(o) for o in objs), tuple(dims))
    hit = _MEMO.get(key)
    if hit is not None:
        out: xr.DataArray = hit[0]
        return out
    value = fn()
    # Keep the inputs alive with the value, so their ids cannot be reused.
    _MEMO[key] = (value, objs)
    return value


def _total(w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    def build() -> xr.DataArray:
        tot = w.sum(dims)
        out: xr.DataArray = tot.where(tot > 0)
        return out

    return _memo("total", build, w, dims=dims)


def wmean(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """Weighted mean over ``dims``; points with zero weight do not count."""

    def build() -> xr.DataArray:
        out: xr.DataArray = (x.fillna(0.0) * w).sum(dims) / _total(w, dims)
        return out

    return _memo("mean", build, x, w, dims=dims)


def wvar(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    return _memo("var", lambda: wmean((x - wmean(x, w, dims)) ** 2, w, dims), x, w, dims=dims)


def wstd(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    return _memo("std", lambda: np.sqrt(wvar(x, w, dims)), x, w, dims=dims)  # type: ignore[arg-type,return-value]


def wscale(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """Weighted mean absolute value: the scale against which a std or a quantile is small."""
    return _memo("scale", lambda: wmean(abs(x), w, dims), x, w, dims=dims)


def varies(x: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """True where ``x`` is not constant over ``dims`` (see :data:`CONSTANT_RTOL`)."""

    def build() -> xr.DataArray:
        out: xr.DataArray = wstd(x, w, dims) > CONSTANT_RTOL * wscale(x, w, dims)
        return out

    return _memo("varies", build, x, w, dims=dims)


def wcov(a: xr.DataArray, b: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    return _memo(
        "cov",
        lambda: wmean((a - wmean(a, w, dims)) * (b - wmean(b, w, dims)), w, dims),
        a,
        b,
        w,
        dims=dims,
    )


def wcorr(a: xr.DataArray, b: xr.DataArray, w: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """Weighted Pearson correlation; NaN when either input is constant."""

    def build() -> xr.DataArray:
        sa, sb = wstd(a, w, dims), wstd(b, w, dims)
        out: xr.DataArray = (wcov(a, b, w, dims) / (sa * sb)).where(
            varies(a, w, dims) & varies(b, w, dims)
        )
        return out

    return _memo("corr", build, a, b, w, dims=dims)


def _gather(x: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """Put ``dims`` in a single dask chunk (no-op for NumPy arrays).

    The other dimensions get dask's automatic chunk size, so a chunk holds
    one full sample for as many kept points as fit in ``array.chunk-size``.
    """
    if x.chunks is None:
        return x
    spec: dict[str, Any] = {str(d): "auto" for d in x.dims if d not in dims}
    spec.update({str(d): -1 for d in dims})
    out: xr.DataArray = x.chunk(spec)
    return out


def _stack(x: xr.DataArray, dims: Sequence[str], name: str) -> xr.DataArray:
    """Stack ``dims`` into one dimension without building a pandas MultiIndex.

    A MultiIndex would hold every sample point in memory as soon as the graph
    is built (68 GiB for a year of hourly 0.25 degree data).
    """
    bare = x.reset_coords(drop=True).drop_vars([d for d in dims if d in x.coords])
    out: xr.DataArray = bare.stack({name: list(dims)}, create_index=False)
    return out


def sample_bytes(x: xr.DataArray, dims: Sequence[str]) -> int:
    """Bytes of one full sample over ``dims`` in float64."""
    return int(math.prod(int(x.sizes[d]) for d in dims)) * 8


def _warn_full_sample(name: str, x: xr.DataArray, dims: Sequence[str]) -> None:
    if x.chunks is None:
        return
    nbytes = sample_bytes(x, dims)
    if nbytes > chunk_limit_bytes():
        warnings.warn(
            f"Metric {name!r} gathers each sample into one dask chunk; here one sample "
            f"over {list(dims)} is {nbytes / 2**30:.3g} GiB. Use the climatology or time "
            "sample, a region or a period to keep it in memory.",
            UserWarning,
            stacklevel=4,
        )


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
        True for rank and quantile metrics, which need the whole sample at
        once (see the module notes on memory).
    ratio_scale : bool
        True for metrics that divide by a reference value (KGE, percentile
        bias). :func:`climval.compare` gives them temperatures on an
        absolute scale.
    """

    name: str = "base"
    units_suffix: str = ""
    higher_is_better: bool = False
    units_kind: str | None = None
    needs_full_sample: bool = False
    ratio_scale: bool = False

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
        _warn_full_sample(self.name, reference, dims)
        r = _gather(_stack(reference, dims, stack), [stack])
        c = _gather(_stack(candidate, dims, stack), [stack])
        w = _gather(_stack(weights.broadcast_like(reference), dims, stack), [stack])

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
        """Value for two NumPy arrays.

        Both are flattened; points where either value or the weight is not
        finite are dropped; weights must not be negative and are normalised
        to sum to one over the remaining points.
        """
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
        if np.any(w < 0):
            raise ValueError("weights must not be negative")
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
    over the same sample. NaN when the reference is constant (its standard
    deviation at most :data:`CONSTANT_RTOL` of its mean absolute value).
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
        out: xr.DataArray = (rmse / sr).where(varies(reference, weights, dims))
        return out


# ---------------------------------------------------------------------------
# Correlation and variability
# ---------------------------------------------------------------------------


class PearsonCorrelation(BaseMetric):
    """Pearson correlation: weighted covariance over the product of weighted standard deviations.

    NaN when either input is constant over the sample.
    """

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
    valid points of the sample with a weight above zero; the correlation of
    the ranks uses the weights. For points with positive weights this
    matches xskillscore.spearman_r.
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
        _warn_full_sample(self.name, reference, dims)
        # Points with zero weight (outside a mask, or a missing area) are not ranked.
        counted = weights.broadcast_like(reference) > 0
        r = _gather(_stack(reference.where(counted), dims, stack), [stack])
        c = _gather(_stack(candidate.where(counted), dims, stack), [stack])
        w = _gather(_stack(weights.broadcast_like(reference), dims, stack), [stack])
        return wcorr(_rank(r, stack), _rank(c, stack), w, [stack])


class StdRatio(BaseMetric):
    """Standard deviation ratio: weighted std of the candidate over that of the reference.

    NaN when the reference is constant over the sample.
    """

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
        out: xr.DataArray = (wstd(candidate, weights, dims) / sr).where(
            varies(reference, weights, dims)
        )
        return out


# ---------------------------------------------------------------------------
# Skill scores
# ---------------------------------------------------------------------------


class TaylorSkillScore(BaseMetric):
    """Taylor skill score, Taylor (2001) eq. 5.

    S = 4 (1 + R)**4 / ((s + 1/s)**2 (1 + R0)**4), with R the weighted Pearson
    correlation, s the standard deviation ratio (candidate over reference)
    and R0 the highest attainable correlation (default 1). S is 1 for a
    perfect pattern and tends to 0 as the pattern degrades. NaN when either
    input is constant over the sample.
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
        out: xr.DataArray = score.where(
            varies(reference, weights, dims) & varies(candidate, weights, dims)
        )
        return out


class KlingGuptaEfficiency(BaseMetric):
    """Kling-Gupta efficiency, Gupta et al. (2009) eq. 9.

    KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2), with r the
    weighted Pearson correlation, alpha the standard deviation ratio and beta
    the ratio of weighted means (candidate over reference). beta depends on
    the zero of the unit scale; :func:`climval.compare` computes KGE for
    temperatures in degC or degF on the Kelvin scale. KGE was designed for
    ratio-scale variables such as precipitation or runoff. NaN when either
    input is constant over the sample, or when the reference's mean is at
    most :data:`CONSTANT_RTOL` of its mean absolute value.
    """

    name = "kge"
    higher_is_better = True
    units_kind = "dimensionless"
    ratio_scale = True

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
        ok = varies(reference, weights, dims) & (
            abs(mr) > CONSTANT_RTOL * wscale(reference, weights, dims)
        )
        return kge.where(ok)  # type: ignore[no-any-return]


HIST_MANTISSA_BITS = 20
"""Mantissa bits kept by the histogram quantile: bins are 2**-20 (about 1e-6) of the value wide."""
_SHIFT = np.uint64(52 - HIST_MANTISSA_BITS)
_SIGN = np.uint64(1 << 63)


def _sortable(x: np.ndarray) -> np.ndarray:
    """Map float64 to uint64 with the same order."""
    u = np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)
    return np.where(u & _SIGN, ~u, u | _SIGN)


def _unsortable(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.uint64)
    u = np.where(s & _SIGN, s & ~_SIGN, ~s)
    return np.ascontiguousarray(u).view(np.float64)


Partial = list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
"""Per side: bin keys, summed weights, and per kept point its row, minimum and maximum."""


def _extremes(
    rows: np.ndarray, mins: np.ndarray, maxs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum and maximum per row (rows need not be sorted)."""
    if rows.size == 0:
        return rows, mins, maxs
    order = np.argsort(rows, kind="stable")
    rows, mins, maxs = rows[order], mins[order], maxs[order]
    starts = np.flatnonzero(np.concatenate([[True], rows[1:] != rows[:-1]]))
    return rows[starts], np.minimum.reduceat(mins, starts), np.maximum.reduceat(maxs, starts)


def _hist_block(
    r: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    kept_shape: tuple[int, ...],
    kept_offset: tuple[int, ...],
) -> Partial:
    """Weighted sparse histograms of one block, keyed by (kept point, value bin)."""
    nk = len(kept_shape)
    block_kept = r.shape[:nk]
    k = int(math.prod(block_kept))
    r2, c2 = r.reshape(k, -1), c.reshape(k, -1)
    w2 = np.broadcast_to(w, r.shape).reshape(k, -1)
    if nk:
        grids = np.meshgrid(
            *[np.arange(n) + o for n, o in zip(block_kept, kept_offset, strict=True)],
            indexing="ij",
        )
        flat = np.ravel_multi_index([g.ravel() for g in grids], kept_shape)
    else:
        flat = np.zeros(1, dtype=np.int64)
    rows = np.broadcast_to(flat.astype(np.uint64)[:, None], r2.shape)
    ok = np.isfinite(r2) & np.isfinite(c2) & np.isfinite(w2) & (w2 > 0)
    out: Partial = []
    for x in (r2, c2):
        keys = (rows[ok] << np.uint64(32)) | (_sortable(x[ok]) >> _SHIFT)
        uniq, inv = np.unique(keys, return_inverse=True)
        sums = np.bincount(inv.ravel(), weights=w2[ok], minlength=uniq.size)
        ext = _extremes(rows[ok], x[ok], x[ok])
        out.append((uniq, sums, *ext))
    return out


def _hist_merge(*parts: Partial) -> Partial:
    out: Partial = []
    for side in range(2):
        keys = np.concatenate([p[side][0] for p in parts])
        sums = np.concatenate([p[side][1] for p in parts])
        uniq, inv = np.unique(keys, return_inverse=True)
        ext = _extremes(*(np.concatenate([p[side][k] for p in parts]) for k in (2, 3, 4)))
        out.append((uniq, np.bincount(inv.ravel(), weights=sums, minlength=uniq.size), *ext))
    return out


def _hist_final(part: Partial, q: float, kept_shape: tuple[int, ...]) -> np.ndarray:
    n = int(math.prod(kept_shape))
    result = np.full((2, n), np.nan)
    for side in range(2):
        keys, sums, ext_rows, ext_min, ext_max = part[side]
        point = (keys >> np.uint64(32)).astype(np.int64)
        bins = keys & np.uint64(0xFFFFFFFF)
        starts = np.searchsorted(point, np.arange(n), side="left")
        stops = np.searchsorted(point, np.arange(n), side="right")
        for i in range(n):
            a, b = int(starts[i]), int(stops[i])
            if a == b:
                continue
            cw = np.cumsum(sums[a:b])
            total = float(cw[-1])
            if total <= 0:
                continue
            target = q * total
            j = min(int(np.searchsorted(cw, target, side="left")), b - a - 1)
            before = float(cw[j - 1]) if j > 0 else 0.0
            frac = 0.0 if sums[a + j] <= 0 else (target - before) / float(sums[a + j])
            frac = min(max(frac, 0.0), 1.0)
            lo = float(_unsortable(np.asarray([bins[a + j] << _SHIFT]))[0])
            hi = float(_unsortable(np.asarray([(bins[a + j] + np.uint64(1)) << _SHIFT]))[0])
            if not np.isfinite(hi) or hi < lo:
                hi = lo
            # The bin can reach beyond the sample; the quantile cannot.
            e = int(np.searchsorted(ext_rows, np.uint64(i)))
            value = lo + frac * (hi - lo)
            result[side, i] = min(max(value, float(ext_min[e])), float(ext_max[e]))
    return result.reshape((2, *kept_shape))


def histogram_quantiles(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    weights: xr.DataArray,
    dims: Sequence[str],
    q: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Weighted quantiles of both inputs over ``dims`` in one pass over dask chunks.

    Each chunk contributes a sparse weighted histogram whose bins keep the top
    20 mantissa bits of each value; the histograms are summed in a tree, and
    the quantile is the inverse of the weighted empirical distribution
    function, interpolated linearly inside the bin where it falls and clipped
    to the sample's minimum and maximum, which are tracked exactly. The result
    lies in the same bin as the inverse of the weighted empirical
    distribution function at q (the smallest value whose cumulative weight
    reaches q of the total), so it differs from that value by at most the
    bin width, 2**-20 (9.5e-7) of the quantile's magnitude. Memory is bounded
    by the number of distinct bins, not by the sample size.
    """
    import dask.array as dsa
    from dask.delayed import delayed

    dims = list(dims)
    kept = [str(d) for d in reference.dims if d not in dims]
    w = weights.broadcast_like(reference)
    arrays = [x.transpose(*kept, *dims) for x in (reference, candidate, w)]
    template = next(x for x in arrays if x.chunks is not None)
    arrays = [x if x.chunks is not None else x.chunk(template.chunksizes) for x in arrays]
    r, c, wb = xr.unify_chunks(*arrays)
    kept_shape = tuple(int(r.sizes[d]) for d in kept)
    assert r.chunks is not None
    offsets = [np.concatenate([[0], np.cumsum(ch[:-1])]).astype(int) for ch in r.chunks]
    blocks = [x.data.to_delayed(optimize_graph=False) for x in (r, c, wb)]
    block_fn = delayed(_hist_block, pure=True)
    parts = [
        block_fn(
            blocks[0][idx],
            blocks[1][idx],
            blocks[2][idx],
            kept_shape,
            tuple(int(offsets[a][idx[a]]) for a in range(len(kept))),
        )
        for idx in np.ndindex(blocks[0].shape)
    ]
    merge_fn = delayed(_hist_merge, pure=True)
    while len(parts) > 1:
        parts = [merge_fn(*parts[i : i + 8]) for i in range(0, len(parts), 8)]
    final = delayed(_hist_final, pure=True)(parts[0], q, kept_shape)
    data = dsa.from_delayed(final, shape=(2, *kept_shape), dtype=float)  # type: ignore[no-untyped-call]
    coords = {d: r[d].values for d in kept if d in r.coords}
    both = xr.DataArray(data, dims=("_side", *kept), coords=coords)
    return both.isel(_side=0, drop=True), both.isel(_side=1, drop=True)


def _percentile_label(percentile: float) -> str:
    p = float(percentile)
    return f"p{int(p)}" if p.is_integer() else "p" + f"{p:g}".replace(".", "_")


class PercentileBias(BaseMetric):
    """Percentile bias: relative difference of a weighted percentile, in percent.

    100 * (Q_c - Q_r) / |Q_r|, with Q the weighted percentile of each sample.
    NaN when |Q_r| is below ``min_relative`` (default 0.01) of the
    reference's weighted mean absolute value: a quantile near zero, such as
    the 5th percentile of daily precipitation, would turn a small absolute
    difference into an arbitrarily large percentage. Being relative to the
    reference value, it depends on the unit's zero; :func:`climval.compare`
    computes it for temperatures in degC or degF on the Kelvin scale.

    Two ways to get Q, chosen per sample with ``method="auto"``:

    ``"exact"``
        xarray's weighted quantile (equal to NumPy's linear method for equal
        weights) on the whole sample, gathered into one dask chunk.
    ``"histogram"``
        :func:`histogram_quantiles`, one pass over the chunks with memory
        bounded by the number of distinct value bins. It estimates the
        inverse of the weighted empirical distribution function to within
        2**-20 of the quantile's magnitude, clipped to the sample's range.
        That is a different estimator from the exact method. For equal
        weights at most one sorted sample value lies between the two (they
        differ by at most the two gaps around the quantile, plus the bin
        width); for unequal weights xarray interpolates between weighted
        order statistics in another way, and the two can be a few sorted
        values apart (up to three in the package's tests).

    ``"auto"`` (default) uses ``"exact"`` for NumPy arrays and for dask
    samples of at most ``max_exact_bytes`` (default: dask's
    ``array.chunk-size``, 128 MiB, which is 16.8 million float64 values),
    and ``"histogram"`` above that. ``method="exact"`` on a larger dask
    sample warns, because it gathers the whole sample into one chunk. The
    method used is recorded in the provenance under ``metric_methods``.
    Points with zero weight are left out by both methods.
    """

    name = "percentile_bias"
    higher_is_better = False
    units_kind = "percent"
    needs_full_sample = True
    ratio_scale = True

    def __init__(
        self,
        percentile: float = 95.0,
        method: str = "auto",
        max_exact_bytes: int | None = None,
        min_relative: float = 0.01,
    ) -> None:
        if not 0.0 <= float(percentile) <= 100.0:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
        if method not in ("auto", "exact", "histogram"):
            raise ValueError(f"method must be 'auto', 'exact' or 'histogram', got {method!r}")
        self.percentile = float(percentile)
        self.method = method
        self.max_exact_bytes = max_exact_bytes
        if not 0.0 <= float(min_relative) < 1.0:
            raise ValueError(f"min_relative must be in [0, 1), got {min_relative}")
        self.min_relative = float(min_relative)
        self.name = f"percentile_bias_{_percentile_label(percentile)}"

    def method_for(self, reference: xr.DataArray, dims: Sequence[str]) -> str:
        """The method ``reduce`` uses for this sample."""
        if self.method != "auto":
            return self.method
        if reference.chunks is None:
            return "exact"
        limit = self.max_exact_bytes if self.max_exact_bytes is not None else chunk_limit_bytes()
        return "exact" if sample_bytes(reference, dims) <= limit else "histogram"

    def reduce(
        self,
        reference: xr.DataArray,
        candidate: xr.DataArray,
        weights: xr.DataArray,
        dims: Sequence[str],
    ) -> xr.DataArray:
        q = self.percentile / 100.0
        dims = list(dims)
        method = self.method_for(reference, dims)
        if method == "histogram" and (reference.chunks is not None or candidate.chunks is not None):
            qr, qc = histogram_quantiles(reference, candidate, weights, dims, q)
        elif method == "histogram":
            ref_d = reference.chunk({d: -1 for d in reference.dims})
            qr, qc = histogram_quantiles(ref_d, candidate, weights, dims, q)
        else:
            _warn_full_sample(self.name, reference, dims)
            wb = weights.broadcast_like(reference)
            counted = wb > 0
            w = _gather(wb, dims)
            qr = _gather(reference.where(counted), dims).weighted(w).quantile(q, dim=dims)
            qc = _gather(candidate.where(counted), dims).weighted(w).quantile(q, dim=dims)
            qr = qr.drop_vars("quantile", errors="ignore")
            qc = qc.drop_vars("quantile", errors="ignore")
        scale = wscale(reference, weights, dims)
        far_from_zero = (abs(qr) > self.min_relative * scale) & (qr != 0)
        out: xr.DataArray = ((qc - qr) / abs(qr) * 100.0).where(far_from_zero)
        out.attrs["climval_method"] = method
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


_PERCENTILE_NAME = re.compile(r"^percentile_bias_p(\d+(?:[._]\d+)?)$")


def get_metric(name: str) -> BaseMetric:
    """Metric by name or alias.

    'percentile_bias_p<N>' builds a percentile bias for any N from 0 to 100;
    write a decimal point as '_' or '.': 'percentile_bias_p99_9' is the
    99.9th percentile (its name is 'percentile_bias_p99_9').
    """
    key = _ALIASES.get(name, name)
    if key in METRIC_REGISTRY:
        return METRIC_REGISTRY[key]
    match = _PERCENTILE_NAME.match(key)
    if match:
        value = float(match.group(1).replace("_", "."))
        if value > 100.0:
            raise ValueError(
                f"Unknown metric '{name}': percentiles run from 0 to 100; write 99.9 as "
                "'percentile_bias_p99_9'"
            )
        return PercentileBias(value)
    raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}")


def canonical_name(name: str) -> str:
    """The result name of a metric given by name or alias (unknown names unchanged)."""
    try:
        return get_metric(name).name
    except ValueError:
        return name


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
