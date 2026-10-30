"""
climval.core.suite
==================
BenchmarkSuite: the 0.1 multi-model interface, now a layer over
:func:`climval.compare`.

Usage
-----
    suite = BenchmarkSuite()
    suite.register(reference_model, role="reference")
    suite.register(candidate_model_1)
    suite.register(candidate_model_2)
    report = suite.run(variables=["tas", "pr"])
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from climval.metrics.engine import metric_units
from climval.metrics.stats import DEFAULT_METRICS, BaseMetric
from climval.models.schema import BenchmarkResult, ClimateModel, MetricResult

if TYPE_CHECKING:
    from climval.core.report import BenchmarkReport

logger = logging.getLogger(__name__)

SYNTHETIC_DATA_NOTE = (
    "Scores computed on synthetic random data, not on model output. "
    "They are not a validation. Load data (path= or data= on both "
    "models) to validate real models."
)


class BenchmarkSuite:
    """
    Orchestrates multi-model climate validation.

    Parameters
    ----------
    metrics : list of BaseMetric, optional
        Metrics to compute. Defaults to DEFAULT_METRICS.
    name : str, optional
        Label for this benchmark run.

    Examples
    --------
    >>> suite = BenchmarkSuite(name="Europe-2000-2020")
    >>> suite.register(era5, role="reference")
    >>> suite.register(cmip6_mpi)
    >>> report = suite.run(variables=["tas", "pr"], region=climval.Box(lat=(35, 72)))
    >>> report.export("results/report.html")
    """

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        name: str = "benchmark",
    ) -> None:
        self.name = name
        self.metrics: list[BaseMetric] = metrics or list(DEFAULT_METRICS)
        self._reference: ClimateModel | None = None
        self._candidates: list[ClimateModel] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model: ClimateModel,
        role: str = "candidate",
    ) -> BenchmarkSuite:
        """
        Register a model.

        Parameters
        ----------
        model : ClimateModel
        role : str
            "reference" designates the ground truth. Only one reference allowed.
            Any other value registers the model as a candidate.

        Returns
        -------
        BenchmarkSuite (for chaining)
        """
        if role == "reference":
            if self._reference is not None:
                raise ValueError(
                    f"A reference model ('{self._reference.name}') is already "
                    "registered. Only one reference is allowed per suite."
                )
            self._reference = model
            logger.info("Registered reference model: %s", model.name)
        else:
            self._candidates.append(model)
            logger.info("Registered candidate model: %s", model.name)

        return self

    def add_metric(self, metric: BaseMetric) -> BenchmarkSuite:
        """Add a custom metric to the suite."""
        self.metrics.append(metric)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        variables: Sequence[str] | None = None,
        n_samples: int = 1000,
        seed: int = 42,
        *,
        synthetic: bool = False,
        **compare_kwargs: Any,
    ) -> BenchmarkReport:
        """
        Run the benchmark.

        With data on every model (``source_path`` or ``data``), each
        candidate is compared with the reference through
        :func:`climval.compare`, which aligns grids, calendars, units and
        missing values first. ``compare_kwargs`` (``region``, ``period``,
        ``regrid``, ``freq`` and so on) are passed on. Domains given to
        :func:`load_model` (``lat_range``, ``lon_range``, ``time_start``,
        ``time_end``) are applied as ``region`` and ``period`` (the overlap
        of both models' domains) unless ``region`` or ``period`` is passed
        here. The values in the report are the ``space_time`` sample; the full
        :class:`~climval.Comparison` of each variable, with provenance, is in
        ``BenchmarkResult.comparisons``.

        Without data the run raises ``ValueError``, unless ``synthetic=True``
        asks for scores on random arrays (structure and CI checks only).
        Those runs emit a ``UserWarning`` and are flagged in the report.

        Parameters
        ----------
        variables : list of str, optional
            Variables to benchmark. Defaults to the variables declared on
            both models that exist in both datasets.
        n_samples : int
            Number of synthetic points (synthetic=True only).
        seed : int
            RNG seed (synthetic=True only).
        synthetic : bool
            Score random arrays when the models carry no data.

        Returns
        -------
        BenchmarkReport
        """
        if self._reference is None:
            raise RuntimeError(
                "No reference model registered. Call suite.register(model, role='reference') first."
            )
        if not self._candidates:
            raise RuntimeError(
                "No candidate models registered. "
                "Call suite.register(model) to add at least one candidate."
            )

        models = [self._reference, *self._candidates]
        if synthetic:
            if compare_kwargs:
                raise ValueError("compare options apply to real data, not to synthetic=True")
        else:
            missing = [m.name for m in models if not m.has_data]
            if missing:
                raise ValueError(
                    f"No data for {', '.join(missing)}. Since climval 0.2.0, "
                    "BenchmarkSuite.run() does not score random numbers in place "
                    "of data. Load data with load_model(path=...) or "
                    "load_model(data=...), or call run(synthetic=True) to check "
                    "the pipeline on random arrays (those scores are not a "
                    "validation)."
                )
            warnings.warn(
                "BenchmarkSuite.run() now aligns grids, calendars, units and "
                "missing values through climval.compare() before scoring. Scores "
                "differ from 0.1, which compared unaligned points. New code can "
                "call climval.compare() directly.",
                FutureWarning,
                stacklevel=2,
            )

        rng = np.random.default_rng(seed)
        results: list[BenchmarkResult] = []
        for candidate in self._candidates:
            if synthetic:
                result = self._compare_synthetic(
                    self._reference, candidate, variables, rng, n_samples
                )
            else:
                result = self._compare_data(self._reference, candidate, variables, compare_kwargs)
            results.append(result)

        flagged = [r.candidate for r in results if r.synthetic]
        if flagged:
            warnings.warn(
                f"{SYNTHETIC_DATA_NOTE} Affected candidates: {', '.join(flagged)}.",
                UserWarning,
                stacklevel=2,
            )

        from climval.core.report import BenchmarkReport as _BenchmarkReport

        return _BenchmarkReport(
            suite_name=self.name,
            reference=self._reference,
            candidates=self._candidates,
            results=results,
            metrics_used=self.metrics,
        )

    @staticmethod
    def _target_variables(
        reference: ClimateModel,
        candidate: ClimateModel,
        variables: Sequence[str] | None,
    ) -> list[str]:
        shared = {v.name for v in reference.variables} & {v.name for v in candidate.variables}
        if variables:
            missing = set(variables) - shared
            if missing:
                logger.warning("Variables %s not present in both models: skipping.", missing)
            return sorted(set(variables) & shared)
        return sorted(shared)

    def _compare_data(
        self,
        reference: ClimateModel,
        candidate: ClimateModel,
        variables: Sequence[str] | None,
        compare_kwargs: dict[str, Any],
    ) -> BenchmarkResult:
        from climval.align.inputs import open_path
        from climval.api import compare

        def source(model: ClimateModel) -> Any:
            # A path is handed to compare() as is, so provenance records it.
            return model.data if model.data is not None else str(model.source_path)

        ref_src, can_src = source(reference), source(candidate)
        declared = self._target_variables(reference, candidate, variables)
        compare_kwargs = {**_domain_options(reference, candidate, compare_kwargs), **compare_kwargs}

        def present(obj: Any) -> set[str]:
            if isinstance(obj, str):
                obj = open_path(obj)  # lazy: reads metadata only
            if isinstance(obj, xr.Dataset):
                return {str(v) for v in obj.data_vars}
            return {str(obj.name)} if obj.name is not None else set(declared)

        available = present(ref_src) & present(can_src)
        if variables:
            absent = set(declared) - available
            if absent:
                raise KeyError(
                    f"Variables {sorted(absent)} not found in both datasets "
                    f"({reference.name}, {candidate.name})"
                )
        target_vars = [v for v in declared if v in available]
        if not target_vars:
            raise ValueError(
                f"No shared variable to compare between {reference.name} and "
                f"{candidate.name}: declared {declared}, present in both datasets "
                f"{sorted(available)}"
            )

        metric_results: list[MetricResult] = []
        comparisons: dict[str, Any] = {}
        for var_name in target_vars:
            comp = compare(
                ref_src,
                can_src,
                var_name,
                metrics=self.metrics,
                samples=("space_time",),
                names=(reference.name, candidate.name),
                **compare_kwargs,
            )
            comparisons[var_name] = comp
            for metric in self.metrics:
                value = comp.samples["space_time"][metric.name].sel(group="all")
                metric_results.append(
                    MetricResult(
                        metric_name=metric.name,
                        variable=var_name,
                        value=float(value) if value.ndim == 0 else float("nan"),
                        units=metric_units(metric, comp.units),
                        description=metric.description(),
                        reference_model=reference.name,
                        candidate_model=candidate.name,
                        metadata={"sample": "space_time"},
                    )
                )
        return BenchmarkResult(
            reference=reference.name,
            candidate=candidate.name,
            metrics=metric_results,
            comparisons=comparisons,
        )

    def _compare_synthetic(
        self,
        reference: ClimateModel,
        candidate: ClimateModel,
        variables: Sequence[str] | None,
        rng: np.random.Generator,
        n_samples: int,
    ) -> BenchmarkResult:
        target_vars = self._target_variables(reference, candidate, variables)
        metric_results: list[MetricResult] = []
        for var_name in target_vars:
            ref_data = rng.normal(loc=288.0, scale=15.0, size=n_samples)
            bias = rng.uniform(-3.0, 3.0)
            noise_scale = rng.uniform(0.5, 2.5)
            can_data = ref_data + bias + rng.normal(0, noise_scale, n_samples)
            for metric in self.metrics:
                try:
                    value = metric.compute(ref_data, can_data)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Metric '%s' failed for var '%s': %s", metric.name, var_name, exc)
                    continue
                metric_results.append(
                    MetricResult(
                        metric_name=metric.name,
                        variable=var_name,
                        value=value,
                        units=self._metric_units(metric, var_name, reference),
                        description=metric.description(),
                        reference_model=reference.name,
                        candidate_model=candidate.name,
                    )
                )
        return BenchmarkResult(
            reference=reference.name,
            candidate=candidate.name,
            metrics=metric_results,
            notes=SYNTHETIC_DATA_NOTE if target_vars else "",
            synthetic=bool(target_vars),
        )

    @staticmethod
    def _metric_units(metric: BaseMetric, var_name: str, reference: ClimateModel) -> str:
        from climval.models.schema import STANDARD_VARIABLES

        var_units = STANDARD_VARIABLES.get(var_name)
        return metric_units(metric, var_units.units if var_units else "?")


def _domain_options(
    reference: ClimateModel, candidate: ClimateModel, given: dict[str, Any]
) -> dict[str, Any]:
    """region and period from the models' load_model domains (overlap of both)."""
    from climval.align.region import Box

    out: dict[str, Any] = {}
    models = [m for m in (reference, candidate)]
    if "region" not in given:
        boxes = [m.spatial_domain for m in models if m.region_given]
        if boxes:
            lat = (max(b.lat_min for b in boxes), min(b.lat_max for b in boxes))
            lons = {(b.lon_min, b.lon_max) for b in boxes}
            if len(lons) > 1:
                raise ValueError(
                    f"{reference.name} and {candidate.name} have different longitude "
                    f"ranges {sorted(lons)}; pass region= to run() to choose one"
                )
            lon_t = next(iter(lons))
            lon = None if lon_t[1] - lon_t[0] >= 360.0 else lon_t
            if lat[0] > lat[1]:
                raise ValueError(
                    f"The latitude ranges of {reference.name} and {candidate.name} do not overlap"
                )
            out["region"] = Box(lat=lat, lon=lon)
    if "period" not in given:
        dated = [m.temporal_domain for m in models if m.period_given]
        if dated:
            start = max(d.start for d in dated)
            end = min(d.end for d in dated)
            out["period"] = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return out
