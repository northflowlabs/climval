"""
climval.metrics
==========================
Statistical metrics for comparing climate model outputs.

All metrics follow a consistent interface:
    compute(reference: np.ndarray, candidate: np.ndarray, **kwargs) -> float

References:
- Gleckler et al. (2008): Performance metrics for climate models.
  J. Geophys. Res., 113, D06104.
- Taylor (2001): Summarizing multiple aspects of model performance
  in a single diagram. J. Geophys. Res., 106, 7183–7192.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseMetric:
    """Abstract base for all climate metrics."""

    name: str = "base"
    units_suffix: str = ""
    higher_is_better: bool = False

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        raise NotImplementedError

    def description(self) -> str:
        return self.__doc__ or ""


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

class RMSE(BaseMetric):
    """Root Mean Square Error — measures average magnitude of error."""

    name = "rmse"
    higher_is_better = False

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        diff = candidate - reference
        if weights is not None:
            weights = weights / weights.sum()
            return float(np.sqrt(np.sum(weights * diff**2)))
        return float(np.sqrt(np.mean(diff**2)))


class MAE(BaseMetric):
    """Mean Absolute Error — robust to outliers."""

    name = "mae"
    higher_is_better = False

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        diff = np.abs(candidate - reference)
        if weights is not None:
            weights = weights / weights.sum()
            return float(np.sum(weights * diff))
        return float(np.mean(diff))


class MeanBias(BaseMetric):
    """Mean Bias — systematic over/under-estimation."""

    name = "mean_bias"
    higher_is_better = False

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        diff = candidate - reference
        if weights is not None:
            weights = weights / weights.sum()
            return float(np.sum(weights * diff))
        return float(np.mean(diff))


class NormalizedRMSE(BaseMetric):
    """Normalized RMSE (NRMSE) — RMSE normalized by observed std dev.

    Values < 1 indicate the model error is smaller than natural variability.
    """

    name = "nrmse"
    higher_is_better = False

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        rmse_val = RMSE().compute(reference, candidate, weights)
        std_ref = float(np.std(reference))
        if std_ref == 0:
            return float("nan")
        return rmse_val / std_ref


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------

class PearsonCorrelation(BaseMetric):
    """Pearson correlation coefficient — linear agreement."""

    name = "pearson_r"
    higher_is_better = True

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        if weights is not None:
            weights = weights / weights.sum()
            ref_mean = np.sum(weights * reference)
            can_mean = np.sum(weights * candidate)
            cov = np.sum(weights * (reference - ref_mean) * (candidate - can_mean))
            std_r = np.sqrt(np.sum(weights * (reference - ref_mean) ** 2))
            std_c = np.sqrt(np.sum(weights * (candidate - can_mean) ** 2))
        else:
            ref_mean = np.mean(reference)
            can_mean = np.mean(candidate)
            cov = np.mean((reference - ref_mean) * (candidate - can_mean))
            std_r = np.std(reference)
            std_c = np.std(candidate)

        if std_r == 0 or std_c == 0:
            return float("nan")
        return float(cov / (std_r * std_c))


class SpearmanCorrelation(BaseMetric):
    """Spearman rank correlation — monotonic agreement, robust to outliers."""

    name = "spearman_r"
    higher_is_better = True

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        from scipy.stats import spearmanr
        r, _ = spearmanr(reference.ravel(), candidate.ravel())
        return float(r)


# ---------------------------------------------------------------------------
# Skill scores
# ---------------------------------------------------------------------------

class TaylorSkillScore(BaseMetric):
    """Taylor Skill Score (TSS) — combines correlation and variance ratio.

    TSS = 4(1+R)^4 / [(σ_f/σ_o + σ_o/σ_f)^2 * (1+R_0)^4]
    where R_0 is the maximum attainable correlation.

    Reference: Taylor (2001), J. Geophys. Res., 106, 7183–7192.
    """

    name = "taylor_skill_score"
    higher_is_better = True

    def __init__(self, r0: float = 1.0) -> None:
        self.r0 = r0  # maximum attainable correlation

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        r = PearsonCorrelation().compute(reference, candidate, weights)
        std_f = float(np.std(candidate))
        std_o = float(np.std(reference))

        if std_o == 0 or std_f == 0:
            return float("nan")

        ratio = std_f / std_o
        numerator = 4 * (1 + r) ** 4
        denominator = (ratio + 1 / ratio) ** 2 * (1 + self.r0) ** 4
        return float(numerator / denominator)


class PercentileBias(BaseMetric):
    """Percentile bias — bias at a specific quantile (e.g. extremes).

    Useful for assessing model skill in capturing tail events.
    """

    name = "percentile_bias"
    higher_is_better = False

    def __init__(self, percentile: float = 95.0) -> None:
        self.percentile = percentile
        self.name = f"percentile_bias_p{int(percentile)}"

    def compute(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        q_ref = float(np.percentile(reference, self.percentile))
        q_can = float(np.percentile(candidate, self.percentile))
        if q_ref == 0:
            return float("nan")
        return float((q_can - q_ref) / abs(q_ref) * 100.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEFAULT_METRICS = [
    RMSE(),
    MAE(),
    MeanBias(),
    NormalizedRMSE(),
    PearsonCorrelation(),
    TaylorSkillScore(),
    PercentileBias(percentile=95.0),
    PercentileBias(percentile=5.0),
]

METRIC_REGISTRY: dict[str, BaseMetric] = {m.name: m for m in DEFAULT_METRICS}


def get_metric(name: str) -> BaseMetric:
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]
