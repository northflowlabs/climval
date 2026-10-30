# Changelog

All notable changes to climval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 0.2.0

### Fixed
- `BenchmarkSuite.run()` on data files compared unrelated points: it flattened
  both datasets, dropped missing values separately and cut both to the shorter
  length, so scores were only meaningful when grids, calendars and missing
  values were identical. It now aligns the data through `climval.compare`
  first
- Weighted Taylor skill score used unweighted standard deviations
- HTML reports escape names and values

### Added
- `climval.compare(reference, candidate, variable=...)`: aligns the two inputs
  and returns a `Comparison` with metric values, `to_dataframe()`,
  `to_dict()`, `to_json()`, `export()` (HTML, JSON, Markdown), the aligned pair
  (`.reference`, `.candidate`, `.aligned`) and a provenance record
- Inputs: `xarray.DataArray`, `xarray.Dataset`, or a path or URL to NetCDF,
  Zarr or CSV, local or remote through fsspec, opened lazily with dask chunks
- Alignment steps, each recorded: CF-aware coordinate detection (cf_xarray
  when installed), longitude convention, latitude order, regions
  (`climval.Box`, across the antimeridian; regionmask regions such as
  `"ar6.land:NEU"`), regridding (`bilinear`, `nearest`, `conservative` through
  xESMF; no extrapolation; holes in a source grid are not bridged), target
  grid choice (`reference`, `candidate`, `coarser` or a given grid), time
  alignment across calendars (`noleap`, `360_day`, `all_leap`, `julian`,
  standard) by aggregation to a common frequency and period labels, unit
  conversion (built-in table, pint for the rest), paired missing-value
  masking, cell-area or cos(latitude) weights
- Metrics: centred RMSE (`crmse`), standard deviation ratio (`std_ratio`),
  Kling-Gupta efficiency (`kge`); every metric is weighted and computed with
  xarray reductions, so dask arrays stay lazy and one pass computes all values
- Samples: `space_time`, `climatology`, `space` (per time step) and `time`
  (per grid point); groups with `by="season"`, `"month"` or `"year"`
- Extra dimensions such as an ensemble `member` are kept in the results
- Provenance: climval, Python and library versions, input identity
  (location, dimensions, coordinate ranges, coordinate and value hashes,
  CF/CMIP global attributes), every alignment step, timestamp
- `climval.evaluate_thresholds` and `Comparison.check` for pass or fail limits
- `climval.demo()` and `climval.demo_data()` for synthetic examples, flagged
  as synthetic in every output
- `.climval.compare(...)` accessor on DataArray and Dataset
- `BaseMetric.reduce(reference, candidate, weights, dims)` for metrics that
  stay lazy on dask; metrics that implement only the NumPy `compute` still work
- `load_model(data=...)` for xarray objects already in memory; remote URLs
  and Zarr directories as `path`
- CLI: `climval compare`, `climval demo`, `climval run --candidate-paths`
- Extras: `cloud`, `regrid`, `units`, `regions`, `cftime`; Python 3.13 in CI

### Changed
- Without data, `BenchmarkSuite.run()` raises `ValueError` instead of scoring
  random arrays. `run(synthetic=True)` keeps the 0.1 behaviour for pipeline
  checks, with the same warning and flags
- `BenchmarkSuite.run()` on real data emits a `FutureWarning` that its scores
  now come from aligned data, accepts `climval.compare` options (`region`,
  `period`, `regrid`, `freq` and others) and stores each `Comparison` in
  `BenchmarkResult.comparisons`. Variables declared on a model but absent
  from its data are skipped
- Dimensionless metrics report units `"1"` instead of `"dimensionless"`
- `DEFAULT_METRICS` adds `crmse`, `std_ratio` and `kge`; `spearman_r` is in
  `METRIC_REGISTRY` and available by name
- `xarray`, `pandas` and `cftime` are core dependencies; timestamps in results
  are timezone aware (UTC)

### Migration from 0.1
- Code that registered models with data files keeps working and now gets
  aligned scores. To call the engine directly:
  `climval.compare("era5.nc", "mpi.nc", variable="tas")`
- Code that ran a suite without data must pass `synthetic=True`, or better,
  load data. Scores from 0.1 runs on files with different grids, calendars or
  missing values are not comparable with 0.2 scores

## [0.1.1] - 2026-10-29

### Changed
- `BenchmarkSuite.run()` emits a `UserWarning` when it falls back to synthetic
  random arrays because no data files are loaded. The scores in that case are
  not a validation. Each affected `BenchmarkResult` has `synthetic=True` and a
  note in `notes`; `BenchmarkReport.synthetic` is True, and the JSON, HTML,
  Markdown and console output say so
- Project metadata: author Northflow Technologies AS, contact hello@northflow.no,
  homepage https://research.northflow.no/systems/climval, repository under github.com/northflowlabs
- The `all` extra lists its dependencies explicitly
- `__version__` matches the package version

## [0.1.0] - 2026-02-21

### Added
- `BenchmarkSuite`: composable orchestrator for multi-model validation
- `load_model()`: unified loader with presets for ERA5, ERA5-Land, CMIP6, CORDEX
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
