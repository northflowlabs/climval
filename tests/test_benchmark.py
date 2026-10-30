# SPDX-License-Identifier: Apache-2.0
"""
Tests for climval core functionality.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from climval import BenchmarkSuite, load_model
from climval.metrics import (
    MAE,
    METRIC_REGISTRY,
    RMSE,
    MeanBias,
    NormalizedRMSE,
    PearsonCorrelation,
    PercentileBias,
    TaylorSkillScore,
    get_metric,
)
from climval.models.schema import (
    ClimateModel,
    ModelType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def reference_model() -> ClimateModel:
    return load_model(
        preset="era5",
        name="ERA5-test",
        time_start=datetime(2000, 1, 1),
        time_end=datetime(2010, 12, 31),
    )


@pytest.fixture
def candidate_model() -> ClimateModel:
    return load_model(
        preset="cmip6-mpi",
        name="MPI-test",
        time_start=datetime(2000, 1, 1),
        time_end=datetime(2010, 12, 31),
    )


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_identical_arrays(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        assert RMSE().compute(x, x) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        ref = np.array([0.0, 0.0, 0.0, 0.0])
        can = np.array([1.0, 1.0, 1.0, 1.0])
        assert RMSE().compute(ref, can) == pytest.approx(1.0)

    def test_weighted(self) -> None:
        ref = np.array([0.0, 0.0])
        can = np.array([1.0, 3.0])
        w = np.array([3.0, 1.0])
        result = RMSE().compute(ref, can, weights=w)
        # Weighted mean of squared errors: (3/4)*1 + (1/4)*9 = 3 → sqrt(3)
        assert result == pytest.approx(math.sqrt(3.0))

    def test_higher_is_better_flag(self) -> None:
        assert RMSE().higher_is_better is False


class TestMAE:
    def test_known_value(self) -> None:
        ref = np.array([0.0, 0.0, 0.0])
        can = np.array([1.0, 2.0, 3.0])
        assert MAE().compute(ref, can) == pytest.approx(2.0)


class TestMeanBias:
    def test_positive_bias(self) -> None:
        ref = np.zeros(100)
        can = np.ones(100) * 5.0
        assert MeanBias().compute(ref, can) == pytest.approx(5.0)

    def test_negative_bias(self) -> None:
        ref = np.ones(10) * 10.0
        can = np.ones(10) * 7.0
        assert MeanBias().compute(ref, can) == pytest.approx(-3.0)


class TestNormalizedRMSE:
    def test_zero_std_reference(self) -> None:
        ref = np.ones(10)  # zero std
        can = np.ones(10) * 2.0
        result = NormalizedRMSE().compute(ref, can)
        assert math.isnan(result)

    def test_unit_normalized(self, rng: np.random.Generator) -> None:
        ref = rng.normal(0, 1, 1000)
        can = ref.copy()
        result = NormalizedRMSE().compute(ref, can)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestPearsonCorrelation:
    def test_perfect_correlation(self) -> None:
        x = np.linspace(0, 1, 100)
        assert PearsonCorrelation().compute(x, x * 2 + 1) == pytest.approx(1.0)

    def test_anti_correlation(self) -> None:
        x = np.linspace(0, 1, 100)
        assert PearsonCorrelation().compute(x, -x) == pytest.approx(-1.0)

    def test_zero_std_returns_nan(self) -> None:
        ref = np.ones(10)
        can = np.ones(10)
        result = PearsonCorrelation().compute(ref, can)
        assert math.isnan(result)

    def test_higher_is_better(self) -> None:
        assert PearsonCorrelation().higher_is_better is True


class TestTaylorSkillScore:
    def test_perfect_model(self) -> None:
        x = np.random.default_rng(1).normal(0, 1, 500)
        score = TaylorSkillScore().compute(x, x)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_score_range(self, rng: np.random.Generator) -> None:
        ref = rng.normal(0, 1, 500)
        noisy = ref + rng.normal(0, 2, 500)
        score = TaylorSkillScore().compute(ref, noisy)
        assert 0.0 <= score <= 1.0


class TestPercentileBias:
    def test_no_bias(self) -> None:
        x = np.random.default_rng(7).normal(0, 1, 1000)
        result = PercentileBias(95).compute(x, x)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_positive_bias(self) -> None:
        ref = np.ones(1000) * 10.0
        can = np.ones(1000) * 12.0
        result = PercentileBias(50).compute(ref, can)
        assert result == pytest.approx(20.0, abs=1e-6)

    def test_zero_reference_quantile(self) -> None:
        ref = np.zeros(100)
        can = np.ones(100)
        result = PercentileBias(50).compute(ref, can)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestMetricRegistry:
    def test_all_defaults_registered(self) -> None:
        expected = {
            "rmse",
            "mae",
            "mean_bias",
            "nrmse",
            "pearson_r",
            "taylor_skill_score",
            "percentile_bias_p95",
            "percentile_bias_p5",
        }
        assert expected.issubset(set(METRIC_REGISTRY.keys()))

    def test_get_metric_valid(self) -> None:
        metric = get_metric("rmse")
        assert metric.name == "rmse"

    def test_get_metric_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("nonexistent_metric_xyz")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_preset_era5(self, reference_model: ClimateModel) -> None:
        assert reference_model.name == "ERA5-test"
        assert reference_model.model_type == ModelType.REANALYSIS

    def test_preset_cmip6(self, candidate_model: ClimateModel) -> None:
        assert candidate_model.model_type == ModelType.GCM

    def test_variables_populated(self, reference_model: ClimateModel) -> None:
        var_names = {v.name for v in reference_model.variables}
        assert "tas" in var_names
        assert "pr" in var_names

    def test_invalid_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            load_model(preset="not_a_real_preset")

    def test_temporal_domain(self, reference_model: ClimateModel) -> None:
        assert reference_model.temporal_domain.duration_years() == pytest.approx(11.0, abs=0.1)

    def test_custom_variable_subset(self) -> None:
        model = load_model(name="minimal", variables=["tas"])
        assert len(model.variables) == 1
        assert model.variables[0].name == "tas"


# ---------------------------------------------------------------------------
# Suite + Report
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def test_basic_run(self, reference_model: ClimateModel, candidate_model: ClimateModel) -> None:
        suite = BenchmarkSuite(name="test-suite")
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        report = suite.run(synthetic=True, n_samples=200)

        assert report.suite_name == "test-suite"
        assert len(report.results) == 1
        assert len(report.results[0].metrics) > 0

    def test_no_reference_raises(self, candidate_model: ClimateModel) -> None:
        suite = BenchmarkSuite()
        suite.register(candidate_model)
        with pytest.raises(RuntimeError, match="No reference model"):
            suite.run()

    def test_no_candidates_raises(self, reference_model: ClimateModel) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        with pytest.raises(RuntimeError, match="No candidate models"):
            suite.run()

    def test_double_reference_raises(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        with pytest.raises(ValueError, match="already registered"):
            suite.register(candidate_model, role="reference")

    def test_multi_candidate(self, reference_model: ClimateModel) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(load_model(preset="cmip6-mpi", name="MPI"))
        suite.register(load_model(preset="cordex-eur", name="CORDEX"))
        report = suite.run(synthetic=True, n_samples=100)
        assert len(report.results) == 2

    def test_variable_filter(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        report = suite.run(variables=["tas"], synthetic=True, n_samples=100)

        for m in report.results[0].metrics:
            assert m.variable == "tas"

    def test_to_dict_structure(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        report = suite.run(synthetic=True, n_samples=50)
        d = report.to_dict()

        assert "suite_name" in d
        assert "reference_model" in d
        assert "results" in d
        assert isinstance(d["results"], list)

    def test_chained_registration(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        suite = (
            BenchmarkSuite().register(reference_model, role="reference").register(candidate_model)
        )
        report = suite.run(synthetic=True, n_samples=50)
        assert len(report.results) == 1

    def test_export_json(
        self,
        reference_model: ClimateModel,
        candidate_model: ClimateModel,
        tmp_path,
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        report = suite.run(synthetic=True, n_samples=50)
        out = tmp_path / "report.json"
        report.export(str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_html(
        self,
        reference_model: ClimateModel,
        candidate_model: ClimateModel,
        tmp_path,
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        report = suite.run(synthetic=True, n_samples=50)
        out = tmp_path / "report.html"
        report.export(str(out))
        assert out.exists()
        content = out.read_text()
        assert "ClimVal" in content

    def test_export_markdown(
        self,
        reference_model: ClimateModel,
        candidate_model: ClimateModel,
        tmp_path,
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        report = suite.run(synthetic=True, n_samples=50)
        out = tmp_path / "report.md"
        report.export(str(out))
        assert out.exists()
        content = out.read_text()
        assert "# ClimVal" in content

    def test_export_invalid_format(
        self,
        reference_model: ClimateModel,
        candidate_model: ClimateModel,
        tmp_path,
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        report = suite.run(synthetic=True, n_samples=50)
        with pytest.raises(ValueError, match="Unsupported export format"):
            report.export(str(tmp_path / "report.xyz"))

    def test_reproducibility(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        def run_once():
            suite = BenchmarkSuite()
            suite.register(reference_model, role="reference")
            suite.register(candidate_model)
            return suite.run(synthetic=True, n_samples=100, seed=99)

        r1 = run_once()
        r2 = run_once()
        v1 = r1.results[0].score_summary()
        v2 = r2.results[0].score_summary()
        for key in v1:
            assert v1[key] == pytest.approx(v2[key])


class TestSyntheticFallback:
    def test_run_without_files_warns(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        with pytest.warns(UserWarning, match="synthetic random data"):
            report = suite.run(synthetic=True, n_samples=50)
        assert report.synthetic is True
        assert all(r.synthetic for r in report.results)
        assert "not a validation" in report.results[0].notes

    def test_synthetic_flag_in_exports(
        self,
        reference_model: ClimateModel,
        candidate_model: ClimateModel,
        tmp_path,
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        with pytest.warns(UserWarning):
            report = suite.run(synthetic=True, n_samples=50)
        d = report.to_dict()
        assert d["synthetic_data"] is True
        assert d["results"][0]["synthetic"] is True
        out = tmp_path / "report.md"
        report.export(str(out))
        assert "Synthetic data" in out.read_text(encoding="utf-8")

    def test_result_default_not_synthetic(self) -> None:
        from climval.models.schema import BenchmarkResult

        assert BenchmarkResult(reference="a", candidate="b").synthetic is False


# ---------------------------------------------------------------------------
# 0.2 behaviour of the 0.1 API
# ---------------------------------------------------------------------------


class TestSuiteOnRealData:
    """Since 0.2.0 run() aligns real data through climval.compare and refuses to
    score random numbers unless synthetic=True is passed."""

    def test_run_without_data_raises(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        suite = BenchmarkSuite()
        suite.register(reference_model, role="reference")
        suite.register(candidate_model)
        with pytest.raises(ValueError, match="synthetic=True"):
            suite.run()

    def test_run_on_files_aligns(self, tmp_path) -> None:
        import xarray as xr

        import climval

        ref_da, can_da = climval.demo_data()
        ref_path, can_path = tmp_path / "ref.nc", tmp_path / "can.nc"
        ref_da.to_dataset().to_netcdf(ref_path)
        can_da.to_dataset().to_netcdf(can_path)
        ref = load_model("era5", name="ERA5", path=ref_path)
        can = load_model("cmip6-mpi", name="MPI", path=can_path)
        suite = BenchmarkSuite().register(ref, role="reference").register(can)
        with pytest.warns(FutureWarning, match="aligns grids"):
            report = suite.run()
        result = report.results[0]
        assert not result.synthetic
        assert {m.variable for m in result.metrics} == {"tas"}  # only tas is in the files
        expected = climval.compare(ref_da, can_da)
        assert result.score_summary()["rmse"] == pytest.approx(expected["rmse"])
        inputs = result.comparisons["tas"].provenance["inputs"]
        assert inputs["reference"]["location"] == str(ref_path)
        units = {m.metric_name: m.units for m in result.metrics}
        assert units["rmse"] == "K" and units["pearson_r"] == "1"
        d = report.to_dict()
        assert d["results"][0]["provenance"]["tas"]["steps"]
        xr.testing.assert_allclose(
            result.comparisons["tas"].candidate.isel(time=0), expected.candidate.isel(time=0)
        )

    def test_run_on_in_memory_data_with_options(self) -> None:
        import climval

        ref_da, can_da = climval.demo_data()
        ref = load_model(name="ref", variables=["tas"], data=ref_da)
        can = load_model(name="can", variables=["tas"], data=can_da)
        suite = BenchmarkSuite().register(ref, role="reference").register(can)
        with pytest.warns(FutureWarning):
            report = suite.run(region=climval.Box(lat=(30, 70)), freq="YS")
        comp = report.results[0].comparisons["tas"]
        assert comp.provenance["options"]["region"]["lat"] == [30.0, 70.0]
        assert comp.aligned.frequency == "yearly"

    def test_synthetic_rejects_compare_options(
        self, reference_model: ClimateModel, candidate_model: ClimateModel
    ) -> None:
        suite = BenchmarkSuite().register(reference_model, role="reference")
        suite.register(candidate_model)
        with pytest.raises(ValueError, match="real data"):
            suite.run(synthetic=True, regrid="nearest")
