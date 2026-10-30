# Migrating from 0.1

climval 0.2 adds an alignment engine and `climval.compare`. The 0.1 interface
(`load_model`, `BenchmarkSuite`, `BenchmarkReport`) still works, with two
changes in behaviour.

## Scores from data files are now aligned

In 0.1, `BenchmarkSuite.run()` flattened both datasets, dropped missing
values separately and cut both to the shorter length. Scores were only
meaningful when grids, calendars and missing values were identical. In 0.2
`run()` aligns each pair through `climval.compare` first and emits a
`FutureWarning` that says so. Scores from 0.1 runs on files with different
grids, calendars or missing values are not comparable with 0.2 scores.

`run()` accepts the `compare` options and keeps each `Comparison`:

```python
report = suite.run(region=climval.Box(lat=(35, 72), lon=(-10, 40)), period=("1995", "2014"))
comparison = report.results[0].comparisons["tas"]   # a Comparison per variable
```

## No data means no scores

In 0.1 a suite without data scored random arrays. In 0.2 it raises
`ValueError`. `run(synthetic=True)` keeps the old behaviour for pipeline
checks, with a warning and a `synthetic` flag in every output.

## From BenchmarkSuite to compare

| 0.1 | 0.2 |
|---|---|
| `load_model("era5", path="era5.nc")` and `suite.register(...)` | pass the path, Dataset or DataArray to `compare` |
| `suite.run()` | `climval.compare(reference, candidate, variable=...)` |
| `report.export("r.html")` | `result.export("r.html")` |
| one value per metric | values per sample (`space_time`, `climatology`, ...) and group |
| no record of what was compared | `result.provenance` |

```python
# 0.1
suite = BenchmarkSuite(name="Europe")
suite.register(load_model("era5", path="era5.nc"), role="reference")
suite.register(load_model(name="MPI-ESM1-2-LR", path="mpi.nc"))
report = suite.run()

# 0.2
result = climval.compare("era5.nc", "mpi.nc", variable=("t2m", "tas"), by="season")
```

## Defaults that change numbers

These are the choices `compare` (and `BenchmarkSuite.run()` on data files)
makes unless told otherwise. Each one changes the values compared with a
simpler script, and each is recorded in `result.provenance`.

| Default | What it does | To get the other behaviour |
|---|---|---|
| `target="coarser"` | compares on the grid with larger cells, the reference's when both are equal | `target="reference"`, `"candidate"` or a grid |
| `weights="area"` | cell areas computed from the grid's latitude and longitude bounds (or a given `areacella`); not cos(latitude) on non-uniform grids | `weights="cos_lat"` |
| time weights | monthly and yearly steps weighted by their length in days in every sample that pools time | none; compare daily data, or weight in your own metric |
| `min_coverage=0.8` | an aggregated value counts only where at least 80 % of the period's native steps are valid, per grid cell | `min_coverage=1.0` for complete periods, a lower value to keep more |
| `seasons="contiguous"` | December counts in the following year's DJF; seasons missing a month are left out and listed | `seasons="calendar"` |
| `freq=None` | the coarser native frequency; sub-daily data keep their step (`"3h"`) instead of being averaged to days | `freq="D"`, `"MS"` and so on |
| `PercentileBias(method="auto")` | exact on samples up to dask's `array.chunk-size`, a one-pass histogram estimate above | `method="exact"` |
| KGE and percentile bias | computed on the Kelvin scale when temperatures are in degC or degF | none |

When the time axis alone cannot tell an input's frequency (a single time
step, one month per year, irregular steps), climval raises instead of
guessing. Pass `freq=`, add time bounds, or set a `frequency` attribute such
as `da.assign_attrs(frequency="mon")`.

## Smaller changes

- Dimensionless metrics report units `"1"` instead of `"dimensionless"`.
- `DEFAULT_METRICS` adds `crmse`, `std_ratio` and `kge`; `spearman_r` is
  available by name.
- Names of non-integer percentiles use `_` for the decimal point:
  `percentile_bias_p99_9`. Metrics can be named by aliases such as `bias`
  and `pearson` in `metrics=` and in thresholds.
- `BaseMetric.compute`, the 0.1 NumPy interface, normalises weights over the
  valid points, drops non-finite points and raises `ValueError` for negative
  weights. The weighted MAE of 0.1 omitted the absolute value; it is now the
  weighted mean of absolute differences.
- An exception raised inside a metric reaches the caller unchanged, as its
  own type.
- `load_model` defaults `lat_range`, `lon_range`, `time_start` and
  `time_end` to `None`; when they are set, `BenchmarkSuite.run()` applies
  them as region and period.
- xarray, pandas, dask, flox, cftime and PyYAML are core dependencies (and tomli
  on Python 3.10).
- Timestamps in results are timezone aware (UTC).

The [changelog](../changelog.md) lists every change.
