<a href="https://northflow.no"><img src="https://raw.githubusercontent.com/northflowlabs/climval/main/.github/assets/northflow-mark.svg" alt="Northflow" width="40" height="40"></a>

# climval

climval is a Python library that validates climate model output against reference data. It puts both datasets on the same grid, period and units, then computes weighted metrics with xarray, lazily on dask arrays, and records how every number was made.

[Research page](https://research.northflow.no/systems/climval) · [PyPI](https://pypi.org/project/climval/) · [Notebooks](https://github.com/northflowlabs/northflow-notebooks) · [Licence](#licence)

[![PyPI](https://img.shields.io/pypi/v/climval?color=0e6b63)](https://pypi.org/project/climval/)
[![CI](https://github.com/northflowlabs/climval/actions/workflows/ci.yml/badge.svg)](https://github.com/northflowlabs/climval/actions/workflows/ci.yml)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-0e6b63)](https://github.com/northflowlabs/climval/blob/main/LICENSE)

## Why climval

Every lab compares climate models with its own scripts for RMSE, bias and Taylor diagrams, and the results rarely reproduce across groups. Most of the work, and most of the mistakes, sit before the metric: grids that differ, longitudes in 0..360 against -180..180, a `360_day` model calendar against a standard-calendar reanalysis, K against degC, missing values in one dataset and not the other. climval does those steps explicitly, in one call, and writes each one into the result. It takes the data you already have open in Python (xarray, dask, Zarr in cloud storage, intake-esm catalogues) and never loads a whole dataset into memory.

## Features

- **Inputs:** `xarray.DataArray`, `xarray.Dataset`, or a path or URL to NetCDF, Zarr or CSV, local or remote through fsspec (`gs://`, `s3://`, `https://`). Files open lazily with dask chunks
- **Alignment:** CF-aware coordinate detection, longitude convention, latitude order, regions (boxes across the antimeridian, regionmask regions such as IPCC AR6), regridding (bilinear, nearest, conservative through xESMF), calendars and frequencies, unit conversion, paired missing-value masking and area weights. Each step is recorded
- **Metrics:** mean bias, MAE, RMSE, centred RMSE, normalised RMSE, Pearson r, Spearman r, standard deviation ratio, Taylor skill score, Kling-Gupta efficiency and percentile bias, all weighted
- **Samples and groups:** the full space-time sample, the spatial pattern of the climatology, per-time-step patterns and per-grid-point maps; per season, month or year
- **Dask:** inputs stay lazy until compute; one pass computes every metric
- **Provenance:** software versions, input identity (path or URL, coordinate and value hashes, CMIP attributes such as `source_id`, `experiment_id`, `member_id`, `tracking_id`), every alignment step and a timestamp, as JSON
- **Thresholds:** pass or fail limits on any value, for CI gates
- **Export:** HTML report, JSON and Markdown; tables in Jupyter
- **CLI:** compare two datasets from the terminal
- **Extensible:** add your own metrics with `BaseMetric`

## Quickstart

```python
import climval
import xarray as xr

era5 = xr.open_zarr("gs://.../era5_monthly.zarr")    # any lazy Dataset
cmip = xr.open_dataset("tas_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_185001-201412.nc", chunks={})

result = climval.compare(
    era5,
    cmip,
    variable=("t2m", "tas"),                     # names differ between the datasets
    region=climval.Box(lat=(35, 72), lon=(-10, 40)),
    period=("1995", "2014"),
    by="season",
)

result.summary()                  # values and the alignment steps
result.to_dataframe()             # long table: sample, group, metric, value, units
result["rmse"]                    # one number: space-time sample, whole period
result.check({"rmse": {"max": 2.0}, "pearson_r": {"min": 0.9, "sample": "climatology"}})
result.export("report.html")      # or .json, .md
```

The candidate is regridded onto the reference grid by default. Pass `target="candidate"`, `target="coarser"` or a grid of your own to compare elsewhere. The same call is available as an accessor: `cmip.climval.compare(era5, variable=("t2m", "tas"))`.

To try climval without data, `climval.demo()` compares two synthetic fields on different grids, calendars and units. Its results are flagged as synthetic in every output.

### How alignment works

| Step | What climval does |
|---|---|
| Coordinates | Finds latitude, longitude and time through cf_xarray when installed, else CF attributes and common names; renames them to `lat`, `lon`, `time`; sorts latitude ascending. Two-dimensional latitude and longitude (curvilinear grids) are accepted only with `regrid="conservative"`, which uses xESMF |
| Longitude | Expresses both inputs in one frame: the target grid's convention, or a frame that keeps the requested region in one piece |
| Region | Cuts the box before regridding. The grid being regridded keeps a margin of one cell, and a wrapped column at each end when it is global, so the edges interpolate without extrapolation. A regionmask polygon is applied after regridding, on the target grid |
| Time | Selects the period on calendar fields, aggregates the finer input to the coarser frequency (or to `freq=`) with a time mean, drops incomplete periods, and matches periods by label (year and month for monthly data), not by timestamp. This makes `noleap`, `360_day`, `all_leap`, `julian` and standard calendars comparable. With no common period it raises |
| Regrid | `bilinear` and `nearest` use `xarray.interp`; `conservative` uses xESMF. Target points outside the source grid, or inside a hole in it, become NaN |
| Units | Converts the candidate into the reference's units: K, degC and degF; kg m-2 s-1 and mm/day; Pa and hPa; fractions and percent, built in. pint handles other units when installed. Unknown mismatches raise unless you pass `units="ignore"` or declare the units |
| Missing values | A point counts only where both inputs are valid |
| Weights | Cell area (`area=` or the reference's `cell_measures`), else cos(latitude), in every spatial mean |

### What each number is computed over

| Sample | Points | Result |
|---|---|---|
| `space_time` (default) | every valid point in space and time, spatially weighted | one value per group |
| `climatology` (default) | the time-mean fields of the group, compared over space, spatially weighted | one value per group |
| `space` | the spatial pattern at each time step | a time series |
| `time` | the time series at each grid point, equal weight per step | a map per group |

Extra dimensions such as an ensemble `member` are kept, so one call scores every member.

## Installation

**Core (NumPy, SciPy, pandas, xarray, cftime):**
```bash
pip install climval
```

**Cloud data and dask (Zarr, fsspec, gcsfs, s3fs):**
```bash
pip install "climval[cloud]"
```

**Other extras:** `netcdf` (netCDF4, h5netcdf), `units` (pint, cf_xarray), `regions` (regionmask), `viz` (matplotlib, cartopy), `all` (everything above). Conservative regridding needs xESMF, which depends on the ESMF library and is best installed with conda: `conda install -c conda-forge xesmf`.

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
  --region 35,72,-10,40 \
  --period 1995 2014 \
  --by season \
  --output results/report.html

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

## Metrics

Differences are candidate minus reference. Means, variances and covariances are weighted; standard deviations are population values.

| Metric | Name | Direction | Definition |
|--------|------|-----------|-----------|
| Mean bias | `mean_bias` | ↓ lower better (in size) | weighted mean of the difference |
| Mean absolute error | `mae` | ↓ lower better | weighted mean of the absolute difference |
| Root mean square error | `rmse` | ↓ lower better | square root of the weighted mean squared difference |
| Centred RMSE | `crmse` | ↓ lower better | RMSE after removing each field's mean (Taylor 2001) |
| Normalised RMSE | `nrmse` | ↓ lower better | RMSE divided by the reference's standard deviation |
| Pearson correlation | `pearson_r` | ↑ higher better | weighted covariance over the product of standard deviations |
| Spearman correlation | `spearman_r` | ↑ higher better | weighted Pearson correlation of unweighted ranks (by name, not computed by default) |
| Standard deviation ratio | `std_ratio` | best at 1 | candidate over reference |
| Taylor skill score | `taylor_skill_score` | ↑ higher better | Taylor (2001) eq. 5 with R0 = 1 |
| Kling-Gupta efficiency | `kge` | ↑ higher better | Gupta et al. (2009); for ratio-scale variables such as precipitation |
| Percentile bias (P95, P5) | `percentile_bias_p95` | ↓ lower better (in size) | 100 (Q_c - Q_r) / abs(Q_r) of weighted percentiles (Gleckler et al. 2008); use K, not degC |

The weighted metrics agree with xskillscore to 1e-10 in the test suite, with and without weights and missing values.

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

`run()` uses `climval.compare` for each candidate and variable. Without data it raises; `run(synthetic=True)` scores random arrays for pipeline checks only and flags them.

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
│   ├── inputs.py       # DataArray, Dataset, path or URL, opened lazily
│   ├── coords.py       # CF coordinate detection, longitude and latitude order
│   ├── region.py       # Box (antimeridian aware), regionmask regions
│   ├── temporal.py     # periods, calendars, frequencies, aggregation
│   ├── regrid.py       # bilinear, nearest, conservative (xESMF)
│   ├── units.py        # unit table, pint fallback
│   └── weights.py      # cell area, cos(latitude)
├── metrics/
│   ├── stats.py        # weighted metrics, BaseMetric, registry
│   └── engine.py       # samples and groups
├── provenance.py       # versions, input identity, hashes
├── thresholds.py       # pass or fail limits
├── accessor.py         # .climval accessor on DataArray and Dataset
├── core/               # 0.1 interface: load_model, BenchmarkSuite, reports
├── models/schema.py    # typed data models (CF convention)
└── cli.py              # climval CLI
```

## Related tools

Climate model evaluation has mature tools. climval works alongside them.

### ESMValTool

[ESMValTool](https://www.esmvaltool.org/) is the community standard for Earth System Model evaluation, backed by ESA, CMIP and a consortium of European research institutions. If you are running **full diagnostic campaigns** against CMIP5/6 with pre-defined recipes, CMOR fixes and multi-decadal observational datasets, ESMValTool is the right tool.

It is configured through YAML recipes and installed through conda, which suits a dedicated climate modelling team.

### PCMDI Metrics Package (PMP)

[PMP](https://github.com/PCMDI/pcmdi_metrics), developed at Lawrence Livermore National Laboratory, provides a systematic framework for computing standardized performance metrics against observational references. It is widely cited and takes work to deploy outside a national lab.

### xskillscore

[xskillscore](https://github.com/xarray-contrib/xskillscore) provides deterministic and probabilistic verification metrics on xarray objects. It expects inputs that are already aligned; climval does the alignment and uses the same metric definitions, which the test suite checks against xskillscore.

### ClimateLearn

[ClimateLearn](https://github.com/aditya-grover/climate-learn) is a Python library oriented toward machine learning pipelines for climate, providing standardized data loading and evaluation for ERA5 and CMIP6. Relevant if your workflow is ML-first.

### Where `climval` fits

climval sits at a different layer:

| | ESMValTool | PMP | **climval** |
|---|---|---|---|
| Installation | Complex, conda-heavy | Moderate | `pip install climval` |
| Configuration | YAML recipes | Config files | Pure Python API |
| Learning curve | Days to weeks | Days | Minutes |
| CI/CD integration | Difficult | Difficult | Native, with thresholds |
| Custom metrics | Recipe-based | Limited | `class MyMetric(BaseMetric)` |
| Programmatic use | Partial | Partial | First-class, on xarray and dask |
| Scope | Full diagnostic suite | Performance metrics | Aligned, recorded comparison of two datasets |

**Use climval when you need:**
- A reproducible validation step inside a Python pipeline, a notebook or a CI workflow
- To compare data already open in xarray, including dask-backed Zarr in cloud storage
- A record of exactly what was compared and how
- Full programmatic control without YAML configuration or recipe management

**Use ESMValTool when you need:**
- Pre-built diagnostic recipes from the CMIP community
- CMOR fixes and preprocessing across large multi-model ensembles
- Figures and diagnostics that conform to IPCC or CMIP reporting standards

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
