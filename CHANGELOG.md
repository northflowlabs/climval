# Changelog

All notable changes to climval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - Unreleased

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
  Zarr or CSV, local or remote through fsspec. NetCDF and Zarr open with dask
  chunks; CSV is read into memory. Zarr stores are recognised by a `.zarr`
  suffix (also `.zarr-v3` and similar) or, for paths and URLs without one
  (Pangeo CMIP6 stores end in a version directory), by their `.zmetadata`,
  `.zgroup` or `zarr.json`. `open_kwargs={"engine": "zarr"}` forces Zarr
- Station data: a CSV whose latitude and longitude columns do not form a full
  grid, or any input with `lat` and `lon` on one shared dimension, is station
  data on a `station` dimension. The gridded input is sampled at the stations
  (bilinear or nearest), a `Box` selects stations, and weights are uniform
- Alignment steps, each recorded: CF-aware coordinate detection (cf_xarray
  when installed), longitude convention, latitude order, regions
  (`climval.Box`, across the antimeridian; regionmask regions such as
  `"ar6.land:NEU"`), regridding (`bilinear`, `nearest`, `conservative` through
  xESMF; no extrapolation beyond the source cells; holes in a source grid are
  not bridged), target grid choice (`coarser` by default, `reference`,
  `candidate` or a given grid), time alignment across calendars (`noleap`,
  `360_day`, `all_leap`, `julian`, standard) by aggregation to a common
  frequency and period labels, unit conversion (built-in table, pint for the
  rest), paired missing-value masking, cell-area weights
- Native time frequency from time bounds (`time_bnds`, CF `bounds`), then a
  `frequency` attribute (CMIP vocabulary such as `mon`, `day`, `3hr`, or a
  pandas alias), then the time steps. With bounds, steps are labelled by the
  midpoints of their intervals, so values stamped at the end of their
  interval (ERA5 accumulations at 00 UTC) count for the right day. Sub-daily
  data keep their step (`freq="3h"` and other multiples of an hour that
  divide 24); hourly against 3-hourly data is compared in 3-hour bins
- `min_coverage` (default 0.8) on `compare` and `align`, and
  `--min-coverage` in the CLI: an aggregated value counts only where at least
  that fraction of the period's native steps hold a valid value, per grid
  cell. The rule is recorded
- Cell-area weights computed from the target grid's latitude and longitude
  bounds (`lat_bnds` and `lon_bnds` when the Dataset has them, else midpoints
  between centres): `R**2 * dlon * (sin(lat_north) - sin(lat_south))`. This
  equals cos(latitude) on a uniform grid and is correct on Gaussian and other
  non-uniform grids. `weights="cos_lat"` gives cos(latitude)
- Time weights: for monthly and yearly comparisons each time step counts in
  proportion to its length in days (reference calendar) in the `space_time`,
  `climatology` and `time` samples, as when months are aggregated into years
- `seasons=` on `compare`: `"contiguous"` (default) puts December into the
  following year's DJF and leaves out seasons missing one of their months,
  listed in the provenance; `"calendar"` pools every December, January and
  February
- Metrics: centred RMSE (`crmse`), standard deviation ratio (`std_ratio`),
  Kling-Gupta efficiency (`kge`); every metric is weighted and works on dask
  arrays. KGE and percentile bias are computed on an absolute temperature
  scale (degC and degF shifted to K) because they divide by a reference value
- `PercentileBias(percentile, method="auto")`: exact on samples that fit in
  dask's `array.chunk-size`, and above that a one-pass histogram estimate with
  bins 2**-20 of the value wide, so the space-time sample of large datasets
  no longer has to fit in one chunk. The method used per sample is in the
  provenance (`metric_methods`). Names of non-integer percentiles use `_` for
  the decimal point: `PercentileBias(99.9).name == "percentile_bias_p99_9"`,
  and `get_metric("percentile_bias_p99_9")` reads it back
- Samples: `space_time`, `climatology`, `space` (per time step) and `time`
  (per grid point); groups with `by="season"`, `"month"` or `"year"`
- Extra dimensions such as an ensemble `member` are kept in the results
- Provenance: climval, Python and library versions, input identity
  (location, dimensions, coordinate ranges, coordinate and value hashes,
  CF/CMIP global attributes), every alignment step, the metric methods and
  notes, timestamp
- `climval.evaluate_thresholds` and `Comparison.check` for pass or fail
  limits; metrics may be named by the aliases `get_metric` accepts
- `climval.demo()` and `climval.demo_data()` for synthetic examples, flagged
  as synthetic in every output
- `.climval.compare(...)` accessor on DataArray and Dataset
- `BaseMetric.reduce(reference, candidate, weights, dims)` for metrics that
  stay lazy on dask; metrics that implement only the NumPy `compute` still work
  (each sample is gathered into one chunk for them)
- `load_model(data=...)` for xarray objects already in memory; remote URLs
  and Zarr directories as `path`
- CLI: `climval compare` (with `--lat S,N` and `--lon W,E`, which accept
  negative edges such as `--lat -60,-30`), `climval demo`,
  `climval run --candidate-paths`
- Extras: `cloud`, `regrid`, `units`, `regions`; Python 3.13 in CI. The
  `cftime` extra is kept as a no-op alias, because cftime is a core
  dependency
- CI job that installs every core dependency at its lower bound on
  Python 3.10 and runs the tests

### Changed
- Without data, `BenchmarkSuite.run()` raises `ValueError` instead of scoring
  random arrays. `run(synthetic=True)` keeps the 0.1 behaviour for pipeline
  checks, with the same warning and flags
- `BenchmarkSuite.run()` on real data emits a `FutureWarning` that its scores
  now come from aligned data, accepts `climval.compare` options (`region`,
  `period`, `regrid`, `freq` and others) and stores each `Comparison` in
  `BenchmarkResult.comparisons`. Variables declared on a model but absent
  from its data are skipped
- `BenchmarkSuite.run()` applies the domains given to `load_model`
  (`lat_range`, `lon_range`, `time_start`, `time_end`) as region and period,
  the overlap of both models' domains, unless `region` or `period` is passed.
  `load_model` now defaults these to `None`; a model without them uses all
  its data, and `summary()` (and `climval info`) says "not set"
- `BaseMetric.compute(reference, candidate, weights=None)`, the 0.1 NumPy
  interface, now normalises weights to sum to one over the valid points,
  drops points where either value or the weight is not finite, and raises
  `ValueError` for negative weights. In 0.1 the weighted MAE omitted the
  absolute value; it is now the weighted mean of absolute differences
- Dimensionless metrics report units `"1"` instead of `"dimensionless"`
- `DEFAULT_METRICS` adds `crmse`, `std_ratio` and `kge`; `spearman_r` is in
  `METRIC_REGISTRY` and available by name
- `xarray`, `pandas`, `cftime` and `dask[array]` are core dependencies;
  timestamps in results are timezone aware (UTC)

### Migration from 0.1
- Code that registered models with data files keeps working and now gets
  aligned scores. To call the engine directly:
  `climval.compare("era5.nc", "mpi.nc", variable="tas")`
- Code that ran a suite without data must pass `synthetic=True`, or better,
  load data. Scores from 0.1 runs on files with different grids, calendars or
  missing values are not comparable with 0.2 scores
- Defaults to know about when comparing numbers across versions:
  `target="coarser"` (pass `target="reference"` to regrid onto the
  reference), cell-area weights from bounds (`weights="cos_lat"` for
  cos(latitude)), month-length time weights, `min_coverage=0.8`,
  `seasons="contiguous"`. When the time axis alone cannot tell an input's
  frequency (a single time step, one month per year, irregular steps), climval
  raises and asks for `freq=`, time bounds or a `frequency` attribute
- Custom metrics that raise inside `reduce` now propagate their own exception
  unchanged (on Python 3.11 and later with a note naming the metric)

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
  Taylor Skill Score (Taylor 2001), Percentile Bias P5/P95
- Typed data models: `ClimateModel`, `ClimateVariable`, `SpatialDomain`,
  `TemporalDomain`, `BenchmarkResult`, `MetricResult`
- CF Conventions v1.10 standard variable definitions: `tas`, `pr`, `psl`, `ua`, `va`, `hurs`
- Export: HTML, JSON, Markdown
- CLI: `climval run`, `climval list-metrics`, `climval list-presets`, `climval info`
- Synthetic data fallback for CI and testing without data files
- Reproducible runs via seeded RNG
- Full test suite with analytical ground-truth assertions
- GitHub Actions CI: Python 3.10, 3.11, 3.12 with coverage reporting
