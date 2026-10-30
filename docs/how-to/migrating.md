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

## Smaller changes

- Dimensionless metrics report units `"1"` instead of `"dimensionless"`.
- `DEFAULT_METRICS` adds `crmse`, `std_ratio` and `kge`; `spearman_r` is
  available by name.
- xarray, pandas, cftime and PyYAML are core dependencies.
- Timestamps in results are timezone aware (UTC).

The [changelog](../changelog.md) lists every change.
