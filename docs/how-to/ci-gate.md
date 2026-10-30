# Thresholds and the CI gate

A threshold turns a metric value into pass or fail. In Python it is
`Comparison.check` (or `climval.evaluate_thresholds`); from the command line
and in CI it is `climval check` with a configuration file.

## Thresholds in Python

```python
verdict = result.check({
    "rmse": {"max": 2.0},                                     # space_time, all
    "mean_bias": {"abs_max": 1.0, "group": "*"},              # every group
    "pearson_r": {"min": 0.95, "sample": "climatology"},
    "std_ratio": {"min": 0.8, "max": 1.25, "sample": "climatology", "group": "JJA"},
})
verdict.passed            # True or False
verdict.failures          # the checks that failed
print(verdict.summary())
```

Each limit takes `min`, `max` and `abs_max` (any combination), and optionally
`sample` (default `space_time`; `climatology` is the other scalar sample) and
`group` (default `all`; `"*"` checks every group, for example every season).
With an extra dimension such as `member` or `model`, every member is checked
and must pass. A NaN value fails, because it cannot show that the limit holds.
A limit must be a finite number, `min` may not be above `max` and `abs_max`
may not be negative: anything else raises `ValueError` before any value is
checked, because a NaN limit would pass every value and an inverted range
would fail every value.
A threshold that matches no value (a metric that was not computed, a group
that does not exist) raises an error rather than passing silently. Metrics
may be named by their aliases (`bias` for `mean_bias`, `pearson` for
`pearson_r`, `tss`, `centred_rmse`; see [Metrics](../concepts/metrics.md)),
and a percentile bias by its percentile, `percentile_bias_p99_9` for the
99.9th.

## A configuration file

`climval check` reads YAML (`.yml`, `.yaml`), TOML (`.toml`) or JSON. Every
check names a reference, a candidate and a variable, any
[`compare`](../reference/compare.md) options, and thresholds:

```yaml
version: 1

defaults:                      # merged into every check; a check overrides them
  period: ["1995", "2014"]
  by: season                   # contiguous seasons: December counts in the next DJF
  min_coverage: 0.8            # the default, written out

checks:
  - name: europe-tas
    description: Near-surface temperature over Europe against ERA5
    reference: data/era5_t2m_monthly.nc          # relative to this file
    candidate:
      catalog: https://storage.googleapis.com/cmip6/pangeo-cmip6.json
      query:
        source_id: NorESM2-LM
        experiment_id: historical
        table_id: Amon
        variable_id: tas
        member_id: r1i1p1f1
      storage_options: {token: anon}
    variable: {reference: t2m, candidate: tas}
    region: {lat: [35, 72], lon: [-10, 40]}
    thresholds:
      mean_bias: {abs_max: 2.0, group: "*"}
      pearson_r: {min: 0.95, sample: climatology}
      crmse:
        - {max: 2.5}
        - {max: 1.5, sample: climatology}

  - name: nordic-land
    reference: data/era5_t2m_monthly.nc
    candidate: {path: "gs://my-bucket/run-42/tas.zarr", storage_options: {token: anon}}
    variable: {reference: t2m, candidate: tas}
    region: "ar6.land:NEU"
    thresholds:
      - {metric: mean_bias, abs_max: 1.5}
      - {metric: pearson_r, min: 0.9, sample: climatology, group: "*"}
```

### Keys of a check

| Key | Value |
|---|---|
| `name` | unique name, used with `--check` (default `check-1`, `check-2`, ...) |
| `description` | free text shown in the reports |
| `reference`, `candidate` | a path or URL, or a mapping (below) |
| `variable` | a name, or `{reference: NAME, candidate: NAME}` |
| `variables` | a list of those: one comparison per variable, same thresholds |
| `region` | `{lat: [S, N], lon: [W, E], name: ...}`, `[S, N]`, `[S, N, W, E]` or a regionmask name such as `"ar6.land:NEU"` |
| `period` | `[start, end]`, for example `["1995", "2014"]` |
| `freq` | `h` or `<N>h` for a whole number of hours that divides 24 (`3h`, `6h`), `D`, `MS` or `YS`; default the coarser native frequency |
| `min_coverage` | fraction in (0, 1] of a period's native steps that must be valid, per grid cell, when aggregating (default `0.8`) |
| `by` | `season`, `month` or `year` |
| `seasons` | `contiguous` (default: December counts in the following DJF, incomplete seasons are left out) or `calendar` (every December, January and February form DJF) |
| `regrid` | `bilinear`, `nearest` or `conservative` |
| `target` | `coarser` (default), `reference` or `candidate` |
| `time` | `intersect` or `exact` |
| `units` | `convert`, `strict`, `ignore` or `{reference: UNITS, candidate: UNITS}` |
| `weights` | `area` (default: cell area from the grid's bounds), `cos_lat` or `none` |
| `metrics` | list of metric names or aliases (default: the default metrics; a threshold on any other metric, such as `spearman_r` or `percentile_bias_p99_9`, needs it listed here) |
| `samples` | list of samples (default `[space_time, climatology]`) |
| `names` | `[reference name, candidate name]` for the reports |
| `open_kwargs` | passed to the xarray opener for both inputs |
| `thresholds` | a mapping `metric: limits` (limits may be a list), or a list of `{metric: ..., min/max/abs_max, sample, group}` |

### Keys of an input

| Key | Value |
|---|---|
| `path` | path (relative to the configuration file), glob or URL to NetCDF, Zarr or CSV |
| `catalog` | intake-esm catalogue (path or URL); needs `query` |
| `query` | mapping passed to `catalog.search()`; must match exactly one dataset |
| `sel` | mapping passed to `Dataset.sel`, for example `{plev: 85000}` or `{member_id: r2i1p1f1}` |
| `storage_options` | fsspec options, for example `{token: anon}` for public Google Cloud buckets |
| `open_kwargs` | further xarray open options, for example `{engine: zarr}` |
| `name` | display name |

Everything is validated before any data is opened: unknown keys, invalid
choices, unknown metrics, thresholds on metrics or samples that are not
computed, limits that are not finite numbers, `min` above `max`, a negative
`abs_max`, and groups that the check's `by` does not produce are configuration errors
(exit code 2). The groups are `DJF`, `MAM`, `JJA` and `SON` for
`by: season`, `Jan` to `Dec` for `by: month`, a year such as `2001` for
`by: year`, and `all` or `"*"` with any `by`. A year that is well formed but
outside the data is only found when the data are read, and is then an error
of that check.

Every check needs at least one threshold. A check without thresholds would
pass whatever the values are, so it is a configuration error rather than a
silent pass; to look at the values without gating, use `climval compare`.

PyYAML reads YAML 1.1, in which `1e-3` (no decimal point) is a string, not a
number. Limits therefore also accept numeric strings such as `"1e-3"`;
writing `1.0e-3` gives a number in every YAML reader.

## Running it

```bash
climval check climval.yml                               # console summary
climval check climval.yml --output-dir climval-report   # .json, .md and .html
climval check climval.yml --json result.json --markdown summary.md --html report.html
climval check climval.yml --check europe-tas            # one check (repeatable)
climval check climval.yml --github-summary              # append to $GITHUB_STEP_SUMMARY
climval check climval.yml --validate                    # only validate the file
climval check climval.yml --verbose                     # tracebacks for errors
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | every check ran and every threshold held |
| 1 | at least one threshold failed, and nothing else went wrong |
| 2 | the configuration is invalid, or a check could not run (missing file, failed query, alignment or units error) |

An error takes precedence over a threshold failure. Every check runs even
when an earlier one fails, so one report shows everything.

### Outputs

- **Console**: one block per check with each limit, the value and pass or
  fail, and a final line with the counts and the exit code.
- **JSON**: the status, the exit code, the configuration path and its
  SHA-256, the environment, and per check the limits, the full comparison
  (every value and the provenance) or the error.
- **Markdown**: a table of checks and a table of limits per check, failures
  first, with the alignment steps folded. It is written for a GitHub job
  summary.
- **HTML**: a self-contained report with the same tables, every value and the
  alignment steps.

## In Python

```python
from climval.gate import load_config, run, run_file

report = run_file("climval.yml")
report.exit_code, report.counts()
report.to_json("result.json")
print(report.to_markdown())
```

## Choosing limits

Limits encode a judgement about what is good enough for a purpose, and climval
does not supply defaults. A common approach is to compute the values for a
known good version (a previous release of the model, or several models of an
ensemble), then set limits a margin away from them so that a regression
fails. Keep the reference data fixed, and record why each limit has its
value next to it in the file.

KGE and the percentile bias are computed on the Kelvin scale for
temperatures, so a temperature bias looks small in them (about 1/280 of the
value, since near-surface temperatures are about 280 K: a 1 K error in the
95th percentile is a percentile bias of about 0.36 percent), and their limits
must be set accordingly.

## Data for CI

CI runners have little memory and time. Compare small, fixed subsets: monthly
means of a region, a reference file committed to the repository or cached
between runs, and the candidate written by the job itself. The
[GitHub Action](github-action.md) runs the gate and uploads the report.
