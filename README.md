<a href="https://northflow.no"><img src="https://raw.githubusercontent.com/northflowlabs/climval/main/.github/assets/northflow-mark.svg" alt="Northflow" width="40" height="40"></a>

# climval

climval is a Python library that validates climate model output against reference data. It puts both datasets on the same grid, period and units, then computes weighted metrics with xarray, lazily on dask arrays, and records how every number was made.

[Documentation](https://northflowlabs.github.io/climval/) · [Examples](https://github.com/northflowlabs/climval/tree/main/examples) · [GitHub Action](https://northflowlabs.github.io/climval/how-to/github-action/) · [PyPI](https://pypi.org/project/climval/) · [Research page](https://research.northflow.no/systems/climval) · [Licence](#licence)

[![PyPI](https://img.shields.io/pypi/v/climval?color=0e6b63)](https://pypi.org/project/climval/)
[![CI](https://github.com/northflowlabs/climval/actions/workflows/ci.yml/badge.svg)](https://github.com/northflowlabs/climval/actions/workflows/ci.yml)
[![Docs](https://github.com/northflowlabs/climval/actions/workflows/docs.yml/badge.svg)](https://northflowlabs.github.io/climval/)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-0e6b63)](https://github.com/northflowlabs/climval/blob/main/LICENSE)

## Why climval

Every lab compares climate models with its own scripts for RMSE, bias and Taylor diagrams, and the results rarely reproduce across groups. Most of the work, and most of the mistakes, sit before the metric: grids that differ, longitudes in 0..360 against -180..180, a `360_day` model calendar against a standard-calendar reanalysis, K against degC, missing values in one dataset and not the other. climval does those steps explicitly, in one call, and writes each one into the result. It takes the data you already have open in Python (xarray, dask, Zarr in cloud storage, intake-esm catalogues). NetCDF and Zarr inputs are read chunk by chunk with dask, and all metrics come out of one dask computation.

## Features

- **Inputs:** `xarray.DataArray`, `xarray.Dataset`, or a path or URL to NetCDF, Zarr or CSV, local or remote through fsspec (`gs://`, `s3://`, `https://`). NetCDF and Zarr open with dask chunks; Zarr stores are recognised by a `.zarr` suffix (also `.zarr-v3`) or by their metadata, so Pangeo CMIP6 store URLs work as given. CSV tables are read into memory; station tables become a `station` dimension
- **Alignment:** CF-aware coordinate detection, longitude convention, latitude order, regions (boxes across the antimeridian, regionmask regions such as IPCC AR6), regridding (bilinear, nearest, first-order conservative through xESMF), station sampling, calendars and frequencies (with time bounds when present), unit conversion, paired missing-value masking and cell-area weights. Each step is recorded
- **Metrics:** mean bias, MAE, RMSE, centred RMSE, normalised RMSE, Pearson r, Spearman r, standard deviation ratio, Taylor skill score, Kling-Gupta efficiency and percentile bias, all weighted
- **Samples and groups:** the full space-time sample, the spatial pattern of the climatology, per-time-step patterns and per-grid-point maps; per season (contiguous DJF), month or year
- **Dask:** inputs stay lazy until compute: alignment reads no data values (only coordinates and, when they are stored as dask arrays, the time and cell bounds, in one small compute), and one `dask.compute` call computes every metric and sample, so each input chunk is read once. Graphs stay small at ERA5 scale: a year of hourly 0.25 degree data against a daily model builds in seconds, in tens of thousands of tasks. Memory per task stays near the chunk size for every metric except Spearman r and NumPy-only custom metrics (see [Memory](#memory))
- **Provenance:** software versions, input identity (path or URL, coordinate and value hashes, CMIP attributes such as `source_id`, `experiment_id`, `member_id`, `tracking_id`), every alignment step and a timestamp, as JSON
- **Thresholds and the CI gate:** pass or fail limits on any value; `climval check` runs the checks of a YAML or TOML file and exits with 0, 1 or 2; a GitHub Action writes the job summary and uploads the report
- **Export:** HTML report, JSON and Markdown; tables in Jupyter
- **CLI:** compare two datasets or run a configuration file from the terminal
- **Extensible:** add your own metrics with `BaseMetric`

## Quickstart

```python
import climval

reference, candidate = climval.demo_data()   # small synthetic fields, no download

result = climval.compare(
    reference,                   # 4 degree grid, -180..180, standard calendar, K
    candidate,                   # 5 degree grid, 0..360, 360_day calendar, degC
    variable="tas",
    region=climval.Box(lat=(35, 72), lon=(-10, 40)),
    by="season",
)

result.summary()                  # values and the alignment steps
result.to_dataframe()             # long table: sample, group, metric, value, units
result["rmse"]                    # one number: space_time sample, whole period
result.check({"rmse": {"max": 2.0}, "pearson_r": {"min": 0.9, "sample": "climatology"}})
result.export("report.html")      # or .json, .md
```

The demo fields differ in grid, longitude convention, latitude order,
calendar and units, so every alignment step has work to do. Results on them
are flagged as synthetic in every output.

### On Pangeo data in the cloud

ERA5 from Google's ARCO-ERA5 store against a CMIP6 model from the Pangeo
catalogue, read anonymously (`pip install "climval[cloud,catalog]"`). One year
reads about 150 MB:

```python
import intake
import xarray as xr
import climval

era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    storage_options={"token": "anon"},
    chunks={},
)
cat = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
query = cat.search(source_id="MPI-ESM1-2-LR", experiment_id="historical", table_id="Amon",
                   variable_id="tas", member_id="r1i1p1f1")
cmip = query.to_dataset_dict(storage_options={"token": "anon"}, progressbar=False).popitem()[1]

result = climval.compare(
    era5,
    cmip.squeeze(drop=True),              # drop the size-1 member_id and dcpp_init_year
    variable=("2m_temperature", "tas"),
    region=climval.Box(lat=(35, 72), lon=(-10, 40)),
    period=("2014", "2014"),              # 6-hourly ERA5 is averaged to monthly means
    by="season",
)
```

Both datasets are compared by default on the grid with the larger cells,
measured as latitude spacing times longitude spacing (`target="coarser"`),
here the 1.875 degree MPI-ESM1-2-LR grid. One axis of that grid can still be
finer than the other input's along the same axis. Pass `target="reference"`,
`target="candidate"` or a grid of your own to compare elsewhere; the grid
used is recorded. The same call is available as an accessor:
`cmip.climval.compare(era5, variable=("2m_temperature", "tas"))`. The
[examples](https://github.com/northflowlabs/climval/tree/main/examples) score
five CMIP6 models against ERA5 and run the CI gate on real data.

### How alignment works

| Step | What climval does |
|---|---|
| Coordinates | Finds latitude, longitude and time through cf_xarray when installed, else CF attributes and common names; renames them to `lat`, `lon`, `time`; sorts latitude ascending. Two-dimensional latitude and longitude (curvilinear grids) are accepted only with `regrid="conservative"`, which uses xESMF. Latitude and longitude on a shared dimension are station data |
| Longitude | Expresses both inputs in one frame: the target grid's convention, or a frame that keeps the requested region in one piece |
| Region | Cuts the box before regridding. The grid being regridded keeps a margin of one cell; for bilinear and nearest on a global grid it also gets a wrapped column at each end when the box reaches the edge of the longitude frame. Conservative remapping gets no wrapped columns and tells xESMF the source is periodic instead. A regionmask polygon is applied after regridding, on the target grid |
| Time | Finds each input's native frequency from time bounds (`time_bnds`), else a `frequency` attribute (CMIP `mon`, `day`, `3hr` and so on), else the time steps. With bounds, each step is labelled by the midpoint of its interval, so values stamped at the end of their interval (ERA5 accumulations at 00 UTC) count for the right day. Selects the period on calendar fields, aggregates the finer input to the coarser frequency (or to `freq=`) with a time mean, and matches periods by label (year and month for monthly data), not by timestamp. Sub-daily data keep their step: hourly against 3-hourly is compared in 3-hour bins. An aggregated value counts only where at least `min_coverage` (default 0.8) of its native steps are valid, per grid cell. This makes `noleap`, `360_day`, `all_leap`, `julian` and standard calendars comparable; daily comparisons between `360_day` and another calendar warn and suggest `freq="MS"`. When the time axis alone cannot tell the frequency (a single time step, one month per year, irregular steps) and the answer changes the result, it raises and asks for `freq=`, time bounds or a `frequency` attribute. With no common period it raises |
| Regrid | `bilinear` and `nearest` use `xarray.interp`; `conservative` uses xESMF and is the method that averages a fine grid over coarse cells (bilinear onto a coarser grid samples points; climval warns and the provenance says so). Target points within half a source cell beyond the outermost source centre take the edge value, so polar rows are kept; points farther out, or inside a hole in the source grid, become NaN. Grids that do not overlap raise `NoOverlapError` |
| Stations | When one input is station data (a CSV with latitude and longitude columns that do not form a full grid, or any DataArray with `lat` and `lon` on a `station` dimension), the other is sampled at the stations (bilinear or nearest) and weights are uniform |
| Units | Converts the candidate into the reference's units: K, degC and degF; kg m-2 s-1, kg m-2 h-1, kg m-2 d-1, mm/h and mm/day; Pa and hPa; fractions and percent, built in. A bare `C` (the coulomb in UDUNITS) is rejected. pint handles other units when installed. Unit symbols are matched with their case ("MB" is not "mb"). Unknown mismatches raise unless you pass `units="ignore"` or declare the units |
| Missing values | A point counts only where both inputs are valid |
| Weights | Cell area in every spatial mean: `area=`, or the field named by the reference's `cell_measures`, else computed from the target grid's latitude and longitude bounds (`lat_bnds`, `lon_bnds`, or midpoints between centres). This equals cos(latitude) on a uniform grid and is close to the true cell areas on Gaussian grids (within about 1% at the polar rows); pass `lat_bnds` or `area=` for exact areas. `weights="cos_lat"` asks for cos(latitude) |

### What each number is computed over

| Sample | Points | Result |
|---|---|---|
| `space_time` (default) | every valid point in space and time, weighted by cell area and period length | one value per group |
| `climatology` (default) | the time-mean fields of the group (means weighted by period length), compared over space, weighted by cell area | one value per group |
| `space` | the spatial pattern at each time step | a time series |
| `time` | the time series at each grid point, weighted by period length | a map per group |

For monthly and yearly comparisons each time step counts in proportion to its length in days, in the reference's calendar, the same weighting used to aggregate months into years. Daily and sub-daily steps count equally.

Groups: `by="season"` uses contiguous seasons by default: a December belongs to the following year's DJF, and seasons missing one of their months (the partial winters at the start and end of a period) are left out and listed in the provenance. `seasons="calendar"` pools every December, January and February instead.

Extra dimensions such as an ensemble `member` are kept, so one call scores every member.

## Installation

**Core (NumPy, SciPy, pandas, xarray, dask, flox, cftime, PyYAML):**
```bash
pip install climval
```

**Cloud data (Zarr, fsspec, gcsfs, s3fs):**
```bash
pip install "climval[cloud]"
```

**Other extras:** `catalog` (intake-esm), `netcdf` (netCDF4, h5netcdf), `units` (pint, cf_xarray), `regions` (regionmask), `docs` (MkDocs Material), `all` (everything above except `docs`). `cftime` is kept as an extra that installs nothing new, because cftime is a core dependency. Conservative regridding needs xESMF, which depends on the ESMF library and is best installed with conda: `conda install -c conda-forge xesmf`.

The lower bounds in `pyproject.toml` are tested in CI: one job installs every core dependency at exactly its lower bound on Python 3.10, and another resolves the core dependencies and the extras (except `regrid`) to their lowest allowed versions.

**Development:**
```bash
git clone https://github.com/northflowlabs/climval
cd climval
pip install -e ".[dev]"
pytest
```

## CLI

```bash
# Align and compare two datasets
climval compare era5.zarr tas_mpi.nc \
  --variable t2m,tas \
  --lat 35,72 --lon -10,40 \
  --period 1995 2014 \
  --by season \
  --output results/report.html

# Southern regions: negative edges work with --lat and --lon
# (--lat -60,-30 or --lat -60 -30), or write --region=-60,-30,-10,40

# Compare two synthetic example fields
climval demo

# 0.1 interface: models with data files
climval run --reference era5 --ref-path era5.nc \
  --candidates cmip6-mpi --candidate-paths mpi.nc

# List available metrics, presets, preset details
climval list-metrics
climval list-presets
climval info --model era5
```

## CI gate

A configuration file declares checks and limits:

```yaml
# climval.yml
version: 1
checks:
  - name: europe-tas
    reference: data/era5_t2m_monthly_europe.nc
    candidate:
      catalog: https://storage.googleapis.com/cmip6/pangeo-cmip6.json
      query: {source_id: NorESM2-LM, experiment_id: historical, table_id: Amon,
              variable_id: tas, member_id: r1i1p1f1}
      storage_options: {token: anon}
    variable: {reference: t2m, candidate: tas}
    region: {lat: [35, 72], lon: [-10, 40]}
    period: ["1995", "2014"]
    by: season
    thresholds:
      mean_bias: {abs_max: 2.0, group: "*"}
      pearson_r: {min: 0.95, sample: climatology}
```

```bash
climval check climval.yml --output-dir climval-report
```

The command prints each limit with its value, writes a JSON result, a
Markdown summary and an HTML report, and exits with **0** when every limit
holds, **1** when a limit fails and **2** when the configuration or the data
cannot be used. In GitHub Actions:

```yaml
- uses: northflowlabs/climval@v0.2.0
  with:
    config: climval.yml
    version: "0.2.0"
    extras: cloud,catalog
```

`version` names the climval release to install. At a release tag the action
installs that release by default; pinned to a commit SHA, it installs the
newest release unless `version` is set, so setting it keeps the gate fixed.
The action writes the summary to the job page and uploads the report as an
artifact. See [Thresholds and the CI gate](https://northflowlabs.github.io/climval/how-to/ci-gate/)
and [the GitHub Action](https://northflowlabs.github.io/climval/how-to/github-action/).

## Metrics

Differences are candidate minus reference. Means, variances and covariances are weighted; standard deviations are population values.

| Metric | Name | Direction | Definition |
|--------|------|-----------|-----------|
| Mean bias | `mean_bias` | smaller magnitude is better | weighted mean of the difference |
| Mean absolute error | `mae` | lower is better | weighted mean of the absolute difference |
| Root mean square error | `rmse` | lower is better | square root of the weighted mean squared difference |
| Centred RMSE | `crmse` | lower is better | RMSE after removing each field's mean (Taylor 2001) |
| Normalised RMSE | `nrmse` | lower is better | RMSE divided by the reference's standard deviation |
| Pearson correlation | `pearson_r` | higher is better | weighted covariance over the product of standard deviations |
| Spearman correlation | `spearman_r` | higher is better | weighted Pearson correlation of unweighted ranks (by name, not computed by default) |
| Standard deviation ratio | `std_ratio` | best at 1 | candidate over reference |
| Taylor skill score | `taylor_skill_score` | higher is better | Taylor (2001) eq. 5 with R0 = 1 |
| Kling-Gupta efficiency | `kge` | higher is better | Gupta et al. (2009), eq. 9 |
| Percentile bias (P95, P5) | `percentile_bias_p95` | smaller magnitude is better | 100 (Q_c - Q_r) / abs(Q_r) of weighted percentiles; any percentile by name, for example `percentile_bias_p99_9` |

KGE and percentile bias divide by a reference value, so they depend on where the unit scale has its zero. `climval.compare` computes both on an absolute scale: temperatures in degC or degF are shifted to K for these two metrics (recorded in the provenance), so the result is the same whatever the temperature unit.

How the metrics are tested: `rmse`, `mae`, `mean_bias`, `pearson_r` and `spearman_r` are compared with xskillscore's `rmse`, `mae`, `me`, `pearson_r` and `spearman_r` at a relative tolerance of 1e-10, over three sets of dimensions, with and without weights and with and without missing values (Spearman with missing values against SciPy, because xskillscore returns NaN there). `crmse`, `nrmse`, `std_ratio`, `taylor_skill_score` and `kge`, which xskillscore does not have, are compared with independent NumPy definitions at 1e-9 with random weights; percentile bias with `numpy.percentile` at 1e-9, and weighted percentiles with integer weights against repeated values.

### Memory

Every metric is a chunk-wise dask reduction except these:

- Percentile bias needs the whole sample. A sample that fits in dask's `array.chunk-size` (128 MiB, 16.8 million float64 values) is gathered into one chunk and computed exactly. A larger one, such as the space-time sample of a year of hourly 0.25 degree data, is estimated in the same single pass from weighted histograms whose bins are 2**-20 (about 1e-6) of the value wide, so memory depends on the number of distinct bins, not on the sample size. It estimates the inverse of the weighted empirical distribution function to within one bin width, clipped to the sample's range; NumPy's linear method is a different estimator, and for equal weights at most one sorted sample value lies between the two. `PercentileBias(95, method="exact")` forces the exact computation; the provenance records which method each sample used (`metric_methods`).
- Spearman r ranks the whole sample and so gathers it into one chunk; for the space-time sample of large data that is the whole dataset. climval warns when a sample exceeds `array.chunk-size`. Spearman r is not computed by default.
- Custom metrics that implement only the NumPy `compute` also get each sample in one chunk.

### Custom metrics

```python
import numpy as np
from climval.metrics import BaseMetric
from climval.metrics.stats import wmean

class MaxAbsError(BaseMetric):
    """Largest absolute difference (NumPy; climval gathers each sample)."""
    name = "max_abs_error"
    units_kind = "data"

    def compute(self, reference, candidate, weights=None):
        return float(np.max(np.abs(candidate - reference)))

class MeanSquaredError(BaseMetric):
    """Weighted mean squared difference (xarray; stays lazy on dask)."""
    name = "mse"

    def reduce(self, reference, candidate, weights, dims):
        return wmean((candidate - reference) ** 2, weights, dims)

result = climval.compare(ref, cand, metrics=["rmse", MaxAbsError(), MeanSquaredError()])
```

## Model Presets

| Preset | Type | Spatial res. | Temporal res. | Description |
|--------|------|-------------|--------------|-------------|
| `era5` | Reanalysis | ~31 km | Hourly | ECMWF ERA5 Global Reanalysis |
| `era5-land` | Reanalysis | ~9 km | Hourly | ERA5-Land High-Resolution |
| `cmip6-mpi` | GCM | ~100 km | Monthly | MPI-ESM1-2-HR (Max Planck) |
| `cmip6-hadgem` | GCM | ~100 km | Monthly | HadGEM3-GC31-LL (Met Office) |
| `cordex-eur` | RCM | ~12 km | Daily | EURO-CORDEX 0.11° |

Presets describe datasets for the 0.1 `BenchmarkSuite` interface. A model gets its data from a path or URL, or from an xarray object:

```python
from climval import BenchmarkSuite, load_model

era5 = load_model("era5", name="ERA5", path="data/era5_tas.nc")
mpi = load_model(name="MPI-ESM1-2-HR", variables=["tas"], data=mpi_dataset)

suite = BenchmarkSuite(name="Europe").register(era5, role="reference").register(mpi)
report = suite.run(region=climval.Box(lat=(35, 72), lon=(-10, 40)))
report.export("results/report.html")
```

`run()` uses `climval.compare` for each candidate and variable. Domains given to `load_model` (`lat_range`, `lon_range`, `time_start`, `time_end`) are applied as the region and period of the comparison (the overlap of both models' domains), unless `run()` gets `region=` or `period=`; a model without them uses all its data. Without data `run()` raises; `run(synthetic=True)` scores random arrays for pipeline checks only and flags them.

## Standard Variables

Follows [CF Conventions v1.10](https://cfconventions.org/). Built-in definitions:

| CF Name | Long Name | Units |
|---------|-----------|-------|
| `tas` | Near-Surface Air Temperature | K |
| `pr` | Precipitation | kg m⁻² s⁻¹ |
| `psl` | Sea Level Pressure | Pa |
| `ua` | Eastward Wind | m s⁻¹ |
| `va` | Northward Wind | m s⁻¹ |
| `hurs` | Near-Surface Relative Humidity | % |

## Export Formats

| Format | Extension | Use case |
|--------|-----------|----------|
| JSON | `.json` | Machine-readable values and provenance, pipeline integration |
| HTML | `.html` | Self-contained report with the alignment steps, sharing |
| Markdown | `.md` | GitHub/GitLab, documentation |

## Architecture

```
climval/
├── api.py              # compare(), demo(), demo_data()
├── result.py           # Comparison: values, tables, export, check
├── align/
│   ├── inputs.py       # DataArray, Dataset, path or URL; bounds, frequency, stations
│   ├── coords.py       # CF coordinate detection, longitude and latitude order
│   ├── region.py       # Box (antimeridian aware), regionmask regions
│   ├── temporal.py     # periods, calendars, frequencies, aggregation
│   ├── regrid.py       # bilinear, nearest, conservative (xESMF), station sampling
│   ├── units.py        # unit table, pint fallback
│   └── weights.py      # cell area from bounds, given areas, cos(latitude)
├── metrics/
│   ├── stats.py        # weighted metrics, BaseMetric, registry
│   └── engine.py       # samples and groups
├── provenance.py       # versions, input identity, hashes
├── thresholds.py       # pass or fail limits
├── gate/               # climval check: configuration, runs, reports
├── accessor.py         # .climval accessor on DataArray and Dataset
├── core/               # 0.1 interface: load_model, BenchmarkSuite, reports
├── models/schema.py    # typed data models (CF convention)
└── cli.py              # climval CLI
```

## Related tools

Climate model evaluation has mature tools. climval works alongside them.

### ESMValTool

[ESMValTool](https://www.esmvaltool.org/) runs recipe-driven diagnostics and performance metrics across CMIP ensembles, with CMOR checks and fixes in its preprocessor and a large set of observational datasets ([Righi et al. 2020](https://doi.org/10.5194/gmd-13-1179-2020); [Eyring et al. 2020](https://doi.org/10.5194/gmd-13-3383-2020)). Recipes are written in YAML. If you run full diagnostic campaigns against CMIP5 or CMIP6, ESMValTool is the right tool.

### PCMDI Metrics Package (PMP)

[PMP](https://github.com/PCMDI/pcmdi_metrics), developed at the Program for Climate Model Diagnosis and Intercomparison (Lawrence Livermore National Laboratory), computes a standard set of performance metrics for CMIP models against observational references ([Lee et al. 2024](https://doi.org/10.5194/gmd-17-3919-2024)), building on the summary statistics of Gleckler et al. (2008).

### xskillscore

[xskillscore](https://github.com/xarray-contrib/xskillscore) provides deterministic and probabilistic verification metrics on xarray objects. It expects inputs that are already aligned; climval does the alignment, and for the metrics both libraries have (RMSE, MAE, mean error, Pearson r, Spearman r) uses the same definitions, which the test suite checks against xskillscore.

### ILAMB

[ILAMB](https://www.ilamb.org/) benchmarks land models against a curated collection of observations with a scoring system ([Collier et al. 2018](https://doi.org/10.1029/2018MS001354)).

### ClimateLearn

[ClimateLearn](https://github.com/aditya-grover/climate-learn) is a Python library for machine learning on climate data, with data loading for ERA5 and CMIP6 and evaluation of forecasts.

### Where `climval` fits

climval compares two datasets: it aligns them (grid, region, calendar, frequency, units, missing values and weights), computes the metrics and records every step. It has no recipes, no observational archive and no CMOR fixes.

- It installs with `pip install climval`; xESMF, needed only for conservative regridding, is installed from conda-forge.
- It takes xarray objects (NumPy or dask), paths, URLs and intake-esm queries.
- It is used from Python, or from a YAML, TOML or JSON file with `climval check`, which exits 0, 1 or 2 for use in CI, and through a GitHub Action.
- A metric of your own is a subclass of `BaseMetric`.

**Use climval when you need:**
- A reproducible validation step inside a Python pipeline, a notebook or a CI workflow
- To compare data already open in xarray, including dask-backed Zarr in cloud storage
- A record of exactly what was compared and how

**Use ESMValTool when you need:**
- Pre-built diagnostic recipes from the CMIP community
- CMOR checks and fixes and preprocessing across large multi-model ensembles

## References

- Gleckler, P. J., Taylor, K. E., & Doutriaux, C. (2008). Performance metrics for
  climate models. *Journal of Geophysical Research*, 113, D06104.
  https://doi.org/10.1029/2007JD008972

- Taylor, K. E. (2001). Summarizing multiple aspects of model performance in a single
  diagram. *Journal of Geophysical Research*, 106(D7), 7183-7192.
  https://doi.org/10.1029/2000JD900719

- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition
  of the mean squared error and NSE: Implications for improving hydrological modelling.
  *Journal of Hydrology*, 377(1-2), 80-91.
  https://doi.org/10.1016/j.jhydrol.2009.08.003

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/northflowlabs/climval/blob/main/CONTRIBUTING.md).

```bash
git clone https://github.com/northflowlabs/climval
cd climval
pip install -e ".[dev]"
pytest && ruff check climval tests && mypy climval/
```

## Licence

climval is licensed under the [Apache License 2.0](https://github.com/northflowlabs/climval/blob/main/LICENSE).

Copyright 2026 Northflow Technologies AS
