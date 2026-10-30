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
  negative edges such as `--lat -60,-30`, `--min-coverage` and `--seasons`),
  `climval demo`, `climval run --candidate-paths`
- Extras: `cloud`, `regrid`, `units`, `regions`; Python 3.13 in CI. The
  `cftime` extra is kept as a no-op alias, because cftime is a core
  dependency
- CI job that installs every core dependency at its lower bound on
  Python 3.10 and runs the tests
- `climval check CONFIG`: a validation gate for CI. A YAML, TOML or JSON
  file declares checks (reference and candidate as paths, URLs or intake-esm
  catalogue queries, variables, region, period, frequency, `min_coverage`,
  groups, `seasons`, regridding, target grid, units, weights, metrics,
  samples) and thresholds per metric, sample and group; metrics may be named
  by their aliases. The file is validated before any data is opened,
  including the limits (finite numbers, `min` not above `max`, `abs_max` not
  negative; numeric strings such as `"1e-3"` are read as numbers) and the
  groups, which must be ones the check's `by` produces. A check without
  thresholds is a configuration error. Inputs shared by checks are opened
  once per run. Output:
  console summary, JSON
  result, Markdown summary for `$GITHUB_STEP_SUMMARY` (`--github-summary`)
  and HTML report (`--output-dir`, `--json`, `--markdown`, `--html`). Exit
  code 0 when every threshold holds, 1 on a threshold failure, 2 on a
  configuration or data error. Also `--check NAME`, `--validate`, `--verbose`
  and `--quiet`, and the Python API `climval.gate`
- `ConfigError` for invalid configuration files
- GitHub Action (`action.yml` in the repository root): installs climval
  (`version`, `extras`), runs `climval check`, writes the job summary,
  uploads the report as an artifact and gates the job (`fail-on`); outputs
  `exit-code`, `status`, `report-dir` and `json`. Used at a release tag
  (`@v0.2.0`), it installs the same climval version unless `version` says
  otherwise. The workflow `.github/workflows/action.yml` exercises it on
  committed test data
- `catalog` extra (intake-esm) and `docs` extra (MkDocs Material,
  mkdocstrings, mkdocs-jupyter)
- Documentation site at https://northflowlabs.github.io/climval/: getting
  started, concepts with formulas, how-to guides, API reference and examples,
  built strictly on every push and deployed from main
- Examples on real open data, read anonymously from public cloud storage:
  ERA5 (ARCO-ERA5) against five CMIP6 historical models from the Pangeo
  catalogue over Europe, each on the coarser of its grid and the ERA5 grid,
  with Taylor diagrams and seasonal bias maps; and the CI gate on ERA5 and two
  CMIP6 models over the AR6 Northern Europe land region
- `CITATION.cff`, `.zenodo.json`, a code of conduct (Contributor Covenant
  2.1), issue and pull request templates

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
- `pyyaml` is a core dependency (and `tomli` on Python 3.10), for
  `climval check` configuration files
- `climval.cli.main` returns the exit code
- The `viz` extra (matplotlib, cartopy) is removed: no part of climval
  imports either. The package metadata give the licence as the SPDX
  expression `Apache-2.0` (PEP 639), which needs setuptools 77 or later to
  build
- The notebook `examples/cmip6_taylor_diagram.ipynb`, which scored synthetic
  arrays through the 0.1 interface, is replaced by
  `examples/era5_cmip6_europe.ipynb` on real ERA5 and CMIP6 data

### Migration from 0.1
- Code that registered models with data files keeps working and now gets
  aligned scores. To call the engine directly:
  `climval.compare("era5.nc", "mpi.nc", variable="tas")`
- Code that ran a suite without data must pass `synthetic=True`, or better,
  load data. Scores from 0.1 runs on files with different grids, calendars or
  missing values are not comparable with 0.2 scores
- Defaults that change numbers, each recorded in the provenance and each
  with an option for the other behaviour:
  - `target="coarser"`: the comparison grid is the one with larger cells
    (the reference's when both are equal). `target="reference"` regrids onto
    the reference grid
  - `weights="area"`: cell areas computed from the target grid's latitude
    and longitude bounds, or a given cell area field. This equals
    cos(latitude) only on a uniform grid; `weights="cos_lat"` gives
    cos(latitude)
  - Time weights: monthly and yearly steps count in proportion to their
    length in days in every sample that pools time steps
  - `min_coverage=0.8`: an aggregated value counts only where at least 80 %
    of its native steps are valid, per grid cell
  - `seasons="contiguous"`: December counts in the following year's DJF and
    incomplete seasons are left out; `seasons="calendar"` pools every
    December, January and February
  - Sub-daily data keep their step (`"3h"`, `"6h"`) instead of being averaged
    to days; pass `freq="D"` for daily means
  - `PercentileBias(method="auto")`: a histogram estimate above dask's
    `array.chunk-size`; `method="exact"` forces the exact percentile
  - KGE and percentile bias are computed on the Kelvin scale for
    temperatures in degC or degF
- When the time axis alone cannot tell an input's frequency (a single time
  step, one month per year, irregular steps), climval raises and asks for
  `freq=`, time bounds or a `frequency` attribute
- Names of non-integer percentiles use `_` for the decimal point
  (`percentile_bias_p99_9`); `percentile_bias_p99.9` is read as the same
  metric
- Custom metrics that raise inside `reduce` now propagate their own exception
  unchanged (on Python 3.11 and later with a note naming the metric)
- The [migration guide](https://northflowlabs.github.io/climval/how-to/migrating/)
  has a table of these defaults

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
