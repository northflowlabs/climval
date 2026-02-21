"""
climval.core.suite
=============================
BenchmarkSuite — the central orchestrator for comparing climate models.

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
from collections.abc import Sequence

import numpy as np

from climval.metrics.stats import DEFAULT_METRICS, BaseMetric
from climval.models.schema import BenchmarkResult, ClimateModel, MetricResult

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Orchestrates multi-model climvaling.

    Parameters
    ----------
    metrics : list of BaseMetric, optional
        Metrics to compute. Defaults to DEFAULT_METRICS (RMSE, MAE, bias,
        Pearson r, Taylor Skill Score, percentile bias at P5 and P95).
    name : str, optional
        Label for this benchmark run.

    Examples
    --------
    >>> suite = BenchmarkSuite(name="Europe-2000-2020")
    >>> suite.register(era5, role="reference")
    >>> suite.register(cmip6_mpi)
    >>> suite.register(cordex_eur)
    >>> report = suite.run(variables=["tas", "pr"])
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
    ) -> BenchmarkReport:
        """
        Run the benchmark.

        When real data files are loaded (NetCDF/CSV), actual arrays are used.
        In the absence of data files, synthetic arrays are generated for
        structural validation and CI use.

        Parameters
        ----------
        variables : list of str, optional
            Subset of variable names to benchmark. Defaults to all shared
            variables between reference and candidates.
        n_samples : int
            Number of synthetic data points (used only when no file is loaded).
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        BenchmarkReport
        """
        if self._reference is None:
            raise RuntimeError(
                "No reference model registered. "
                "Call suite.register(model, role='reference') first."
            )
        if not self._candidates:
            raise RuntimeError(
                "No candidate models registered. "
                "Call suite.register(model) to add at least one candidate."
            )

        rng = np.random.default_rng(seed)
        results: list[BenchmarkResult] = []

        for candidate in self._candidates:
            result = self._compare(
                reference=self._reference,
                candidate=candidate,
                variables=variables,
                rng=rng,
                n_samples=n_samples,
            )
            results.append(result)

        from climval.core.report import BenchmarkReport
        return BenchmarkReport(
            suite_name=self.name,
            reference=self._reference,
            candidates=self._candidates,
            results=results,
            metrics_used=self.metrics,
        )

    def _compare(
        self,
        reference: ClimateModel,
        candidate: ClimateModel,
        variables: Sequence[str] | None,
        rng: np.random.Generator,
        n_samples: int,
    ) -> BenchmarkResult:
        ref_var_names = {v.name for v in reference.variables}
        can_var_names = {v.name for v in candidate.variables}
        shared = ref_var_names & can_var_names

        if variables:
            target_vars = set(variables) & shared
            missing = set(variables) - shared
            if missing:
                logger.warning(
                    "Variables %s not present in both models — skipping.", missing
                )
        else:
            target_vars = shared

        metric_results: list[MetricResult] = []

        for var_name in sorted(target_vars):
            ref_data, can_data = self._get_arrays(
                reference, candidate, var_name, rng, n_samples
            )

            for metric in self.metrics:
                try:
                    value = metric.compute(ref_data, can_data)
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
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Metric '%s' failed for var '%s': %s",
                        metric.name,
                        var_name,
                        exc,
                    )

        return BenchmarkResult(
            reference=reference.name,
            candidate=candidate.name,
            metrics=metric_results,
        )

    def _get_arrays(
        self,
        reference: ClimateModel,
        candidate: ClimateModel,
        var_name: str,
        rng: np.random.Generator,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (reference_array, candidate_array) for a given variable.

        If model files are loaded, real data is extracted. Otherwise,
        synthetic arrays are produced using realistic noise models.
        """
        if reference.source_path and candidate.source_path:
            return self._load_arrays_from_files(
                reference.source_path, candidate.source_path, var_name
            )

        # Synthetic fallback: reference is "truth", candidate has bias + noise
        ref_data = rng.normal(loc=288.0, scale=15.0, size=n_samples)  # ~15°C in K
        bias = rng.uniform(-3.0, 3.0)
        noise_scale = rng.uniform(0.5, 2.5)
        can_data = ref_data + bias + rng.normal(0, noise_scale, n_samples)

        logger.debug(
            "Using synthetic data for %s / %s (var=%s, bias=%.2f, noise=%.2f)",
            reference.name,
            candidate.name,
            var_name,
            bias,
            noise_scale,
        )
        return ref_data, can_data

    def _load_arrays_from_files(
        self, ref_path: str, can_path: str, var_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load variable arrays from NetCDF files using xarray."""
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError(
                "xarray is required for loading NetCDF files. "
                "Install it with: pip install xarray netCDF4"
            ) from e

        ref_ds = xr.open_dataset(ref_path)
        can_ds = xr.open_dataset(can_path)

        if var_name not in ref_ds:
            raise KeyError(f"Variable '{var_name}' not found in {ref_path}")
        if var_name not in can_ds:
            raise KeyError(f"Variable '{var_name}' not found in {can_path}")

        ref_arr = ref_ds[var_name].values.ravel()
        can_arr = can_ds[var_name].values.ravel()

        ref_arr = ref_arr[~np.isnan(ref_arr)]
        can_arr = can_arr[~np.isnan(can_arr)]

        min_len = min(len(ref_arr), len(can_arr))
        return ref_arr[:min_len], can_arr[:min_len]

    @staticmethod
    def _metric_units(
        metric: BaseMetric, var_name: str, reference: ClimateModel
    ) -> str:
        from climval.models.schema import STANDARD_VARIABLES

        if metric.name in {"pearson_r", "spearman_r", "taylor_skill_score", "nrmse"}:
            return "dimensionless"
        if metric.name.startswith("percentile_bias"):
            return "%"
        var_units = STANDARD_VARIABLES.get(var_name)
        return var_units.units if var_units else "?"
