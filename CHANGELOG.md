# Changelog

All notable changes to climval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-02-21

### Added
- `BenchmarkSuite` — composable orchestrator for multi-model validation
- `load_model()` — unified loader with presets for ERA5, ERA5-Land, CMIP6, CORDEX
- Metric library: RMSE, MAE, Mean Bias, NRMSE, Pearson r, Spearman r,
  Taylor Skill Score (Taylor 2001), Percentile Bias P5/P95 (Gleckler et al. 2008)
- Typed data models: `ClimateModel`, `ClimateVariable`, `SpatialDomain`,
  `TemporalDomain`, `BenchmarkResult`, `MetricResult`
- CF Conventions v1.10 standard variable definitions: `tas`, `pr`, `psl`, `ua`, `va`, `hurs`
- Export: HTML, JSON, Markdown
- CLI: `climval run`, `climval list-metrics`, `climval list-presets`, `climval info`
- Synthetic data fallback for CI and testing without data files
- Reproducible runs via seeded RNG
- Full test suite with analytical ground-truth assertions
- GitHub Actions CI: Python 3.10, 3.11, 3.12 with coverage reporting
