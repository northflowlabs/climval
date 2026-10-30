"""Evaluate pass or fail thresholds against a comparison.

This is the hook for CI gates: a dict of limits in, a :class:`ThresholdResult`
with one :class:`ThresholdCheck` per value out.

Example
-------
>>> result = climval.compare(ref, cand, variable="tas")
>>> verdict = climval.evaluate_thresholds(result, {
...     "rmse": {"max": 2.0},
...     "mean_bias": {"abs_max": 0.5},
...     "pearson_r": {"min": 0.9, "sample": "climatology"},
... })
>>> verdict.passed
True

Each limit may name ``sample`` (default "space_time") and ``group``
(default "all"; "*" checks every group). A NaN value fails, because it
cannot show that the limit holds. Metrics may be named by the aliases
:func:`climval.metrics.get_metric` accepts ("bias", "tss", "pearson" and so
on); the checks report the metric's own name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from climval.result import Comparison

_LIMITS = ("min", "max", "abs_max")


@dataclass
class ThresholdCheck:
    metric: str
    sample: str
    group: str
    value: float
    limits: dict[str, float]
    passed: bool
    message: str
    point: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "sample": self.sample,
            "group": self.group,
            "point": self.point,
            "value": self.value if math.isfinite(self.value) else None,
            "limits": self.limits,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass
class ThresholdResult:
    checks: list[ThresholdCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[ThresholdCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks]}

    def summary(self) -> str:
        lines = [f"{'PASS' if c.passed else 'FAIL'}  {c.message}" for c in self.checks]
        lines.append(f"{len(self.checks) - len(self.failures)} of {len(self.checks)} checks passed")
        return "\n".join(lines)


def _check(value: float, limits: dict[str, float]) -> tuple[bool, list[str]]:
    if not math.isfinite(value):
        return False, ["value is NaN"]
    problems = []
    if "min" in limits and value < limits["min"]:
        problems.append(f"below min {limits['min']:g}")
    if "max" in limits and value > limits["max"]:
        problems.append(f"above max {limits['max']:g}")
    if "abs_max" in limits and abs(value) > limits["abs_max"]:
        problems.append(f"|value| above {limits['abs_max']:g}")
    return not problems, problems


def evaluate_thresholds(
    comparison: Comparison, thresholds: Mapping[str, Mapping[str, Any]]
) -> ThresholdResult:
    """Check metric values against limits; see the module docstring."""
    from climval.metrics.stats import canonical_name

    rows = comparison.rows()
    checks: list[ThresholdCheck] = []
    for given, spec in thresholds.items():
        metric = canonical_name(given)
        limits = {k: float(spec[k]) for k in _LIMITS if k in spec}
        if not limits:
            raise ValueError(f"Threshold for {metric!r} needs one of {_LIMITS}")
        unknown = set(spec) - set(_LIMITS) - {"sample", "group"}
        if unknown:
            raise ValueError(f"Unknown threshold keys for {metric!r}: {sorted(unknown)}")
        sample = str(spec.get("sample", "space_time"))
        group = str(spec.get("group", "all"))
        matched = [
            r
            for r in rows
            if r["metric"] == metric
            and r["sample"] == sample
            and (group == "*" or r["group"] == group)
        ]
        if not matched:
            raise ValueError(
                f"No value for metric {given!r}, sample {sample!r}, group {group!r} "
                "in this comparison"
            )
        for r in matched:
            point = {
                k: v
                for k, v in r.items()
                if k not in ("sample", "group", "metric", "value", "units")
            }
            ok, problems = _check(float(r["value"]), limits)
            where = f"{metric} [{sample}, {r['group']}" + (
                "".join(f", {k}={v}" for k, v in point.items()) + "]"
            )
            msg = f"{where} = {r['value']:.6g}" + ("" if ok else ": " + "; ".join(problems))
            checks.append(
                ThresholdCheck(
                    metric=metric,
                    sample=sample,
                    group=str(r["group"]),
                    value=float(r["value"]),
                    limits=limits,
                    passed=ok,
                    message=msg,
                    point=point,
                )
            )
    return ThresholdResult(checks=checks)
