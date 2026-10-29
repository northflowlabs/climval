<a href="https://northflow.no"><img src=".github/assets/northflow-mark.svg" alt="Northflow" width="40" height="40"></a>

# climval

climval is a Python library that validates and benchmarks climate model output against reference datasets.

[Research page](https://research.northflow.no/systems/climval) · [PyPI](https://pypi.org/project/climval/) · [Notebooks](https://github.com/northflowlabs/northflow-notebooks) · [Licence](#licence)

[![PyPI](https://img.shields.io/pypi/v/climval?color=0e6b63)](https://pypi.org/project/climval/)
[![CI](https://github.com/northflowlabs/climval/actions/workflows/ci.yml/badge.svg)](https://github.com/northflowlabs/climval/actions/workflows/ci.yml)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-0e6b63)](LICENSE)

## Why climval

Every lab compares climate models with its own scripts for RMSE, bias and Taylor diagrams, and the results rarely reproduce across groups. climval gives that comparison a typed, composable Python interface that runs the same way on a laptop, in a CI pipeline and in a long research workflow. It works with CMIP6, ERA5, CORDEX and custom simulation data.

## Features

- **Metric library:** RMSE, MAE, mean bias, normalised RMSE, Pearson r, Spearman r, Taylor skill score and percentile bias (P5, P95)
- **Typed data models:** CF-convention variable definitions, spatial and temporal domain objects, and structured `ValidationResult` schemas
- **Presets:** ERA5, ERA5-Land, CMIP6-MPI, CMIP6-HadGEM and CORDEX-EUR
- **NetCDF, CSV and Zarr:** read through `xarray` when real files are provided
- **Synthetic fallback:** structural validation and CI without data files
- **Export:** HTML report, JSON and Markdown
- **CLI:** run comparisons from the terminal
- **Extensible:** add your own metrics, variables and loaders

## Quickstart

```python
from datetime import datetime
import numpy as np
from climval import BenchmarkSuite
from climval.metrics import RMSE, MAE, MeanBias, PearsonCorrelation, TaylorSkillScore
from climval.models.schema import (
    ClimateModel, ModelType, SpatialDomain,
    TemporalDomain, SpatialResolution, TemporalResolution,
    ClimateVariable
)

# Define your reference dataset (e.g. ERA5)
era5 = ClimateModel(
    name="ERA5",
    model_type=ModelType.REANALYSIS,
    spatial_domain=SpatialDomain(
        lat_min=39.0, lat_max=40.0,
        lon_min=-0.5, lon_max=0.5,
    ),
    temporal_domain=TemporalDomain(
        start=datetime(2025, 6, 1),
        end=datetime(2025, 6, 30),
    ),
    spatial_resolution=SpatialResolution.LOW,
    temporal_resolution=TemporalResolution.HOURLY,
    variables=[
        ClimateVariable(name="tas", long_name="Near-Surface Air Temperature",
                        units="K", standard_name="air_temperature")
    ],
    metadata={"source": "Copernicus Climate Data Store (CDS)"},
)

# Define your candidate dataset (e.g. Sentinel-3 SLSTR LST)
sentinel3 = ClimateModel(
    name="Sentinel-3 SLSTR LST",
    model_type=ModelType.OBSERVATION,
    spatial_domain=SpatialDomain(
        lat_min=39.0, lat_max=40.0,
        lon_min=-0.5, lon_max=0.5,
    ),
    temporal_domain=TemporalDomain(
        start=datetime(2025, 6, 1),
        end=datetime(2025, 6, 30),
    ),
    spatial_resolution=SpatialResolution.HIGH,
    temporal_resolution=TemporalResolution.DAILY,
    variables=[
        ClimateVariable(name="tas", long_name="Land Surface Temperature",
                        units="K", standard_name="surface_temperature")
    ],
    metadata={"source": "EOPF Sentinel Zarr Samples Service (EODC)"},
)

# Run benchmark
suite = BenchmarkSuite(name="S3-LST-vs-ERA5")
suite.register(era5, role="reference")
suite.register(sentinel3)

report = suite.run(variables=["tas"], seed=42)
report.summary()
report.export("validation_report.html")
```

Output:

```
════════════════════════════════════════════════════════════════════════
  CLIMVAL: EUROPE-2000-2020
  Reference : ERA5
  Generated : 2025-01-15 14:32 UTC
════════════════════════════════════════════════════════════════════════

  ▸ Candidate: MPI-ESM1-2-HR
  ──────────────────────────────────────────────────────────────────────
  Variable   Metric                              Value  Units
  ──────────────────────────────────────────────────────────────────────
  tas        rmse                               1.2930  K
  tas        mae                                1.0744  K
  tas        mean_bias                         -0.9721  K
  tas        nrmse                              0.0845  dimensionless
  tas        pearson_r                          0.9985  dimensionless
  tas        taylor_skill_score                 0.9969  dimensionless
  tas        percentile_bias_p95               -0.0473  %
  tas        percentile_bias_p5                -0.6058  %
  ...
```

## Installation

**Core (no NetCDF dependencies):**
```bash
pip install climval
```

**With NetCDF/xarray support:**
```bash
pip install "climval[netcdf]"
```

**With visualization:**
```bash
pip install "climval[viz]"
```

**Development:**
```bash
git clone https://github.com/northflowlabs/climval
cd climval
pip install -e ".[dev]"
pytest
```

## CLI

```bash
# Run a validation
climval run \
  --reference era5 \
  --candidates cmip6-mpi cordex-eur \
  --variables tas pr \
  --output results/report.html \
  --name "Europe-2000-2020"

# List available metrics
climval list-metrics

# List model presets
climval list-presets

# Inspect a preset
climval info --model era5
```

## Metrics

| Metric | Name | Direction | Reference |
|--------|------|-----------|-----------|
| Root Mean Square Error | `rmse` | ↓ lower better |  |
| Mean Absolute Error | `mae` | ↓ lower better |  |
| Mean Bias | `mean_bias` | ↓ lower better |  |
| Normalized RMSE | `nrmse` | ↓ lower better |  |
| Pearson Correlation | `pearson_r` | ↑ higher better |  |
| Spearman Correlation | `spearman_r` | ↑ higher better |  |
| Taylor Skill Score | `taylor_skill_score` | ↑ higher better | Taylor (2001) |
| Percentile Bias (P95) | `percentile_bias_p95` | ↓ lower better | Gleckler et al. (2008) |
| Percentile Bias (P5) | `percentile_bias_p5` | ↓ lower better | Gleckler et al. (2008) |

### Custom metrics

```python
from climval.metrics import BaseMetric
import numpy as np

class KGE(BaseMetric):
    """Kling-Gupta Efficiency (Gupta et al., 2009)."""
    name = "kge"
    higher_is_better = True

    def compute(self, reference, candidate, weights=None):
        r     = np.corrcoef(reference, candidate)[0, 1]
        alpha = np.std(candidate) / np.std(reference)
        beta  = np.mean(candidate) / np.mean(reference)
        return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

suite.add_metric(KGE())
```

## Model Presets

| Preset | Type | Spatial res. | Temporal res. | Description |
|--------|------|-------------|--------------|-------------|
| `era5` | Reanalysis | ~31 km | Hourly | ECMWF ERA5 Global Reanalysis |
| `era5-land` | Reanalysis | ~9 km | Hourly | ERA5-Land High-Resolution |
| `cmip6-mpi` | GCM | ~100 km | Monthly | MPI-ESM1-2-HR (Max Planck) |
| `cmip6-hadgem` | GCM | ~100 km | Monthly | HadGEM3-GC31-LL (Met Office) |
| `cordex-eur` | RCM | ~12 km | Daily | EURO-CORDEX 0.11° |

Custom models load from any NetCDF4, CSV, or Zarr source:

```python
from climval import load_model
from climval.models import ModelType

model = load_model(
    name="MyModel-v2",
    path="data/simulation.nc",
    variables=["tas", "pr", "psl"],
    lat_range=(50.0, 60.0),
    lon_range=(10.0, 25.0),
    model_type=ModelType.CUSTOM,
)
```

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
| JSON | `.json` | Machine-readable, pipeline integration |
| HTML | `.html` | Self-contained report, sharing |
| Markdown | `.md` | GitHub/GitLab, documentation |

## Architecture

```
climval/
├── core/
│   ├── loader.py       # Model loader with preset registry
│   ├── suite.py        # BenchmarkSuite orchestrator
│   └── report.py       # BenchmarkReport + multi-format export
├── metrics/
│   └── stats.py        # All metric implementations + registry
├── models/
│   └── schema.py       # Typed data models (CF-convention)
└── cli.py              # climval CLI
```

## Related tools

Climate model evaluation has mature tools. climval works alongside them.

### ESMValTool

[ESMValTool](https://www.esmvaltool.org/) is the community standard for Earth System Model evaluation, backed by ESA, CMIP and a consortium of European research institutions. If you are running **full diagnostic campaigns** against CMIP5/6 with pre-defined recipes, regridding pipelines, and multi-decadal observational datasets, ESMValTool is the right tool.

It is configured through YAML recipes and installed through conda, which suits a dedicated climate modelling team.

### PCMDI Metrics Package (PMP)

[PMP](https://github.com/PCMDI/pcmdi_metrics), developed at Lawrence Livermore National Laboratory, provides a systematic framework for computing standardized performance metrics against observational references. It is widely cited and takes work to deploy outside a national lab.

### ClimateLearn

[ClimateLearn](https://github.com/aditya-grover/climate-learn) is a Python library oriented toward machine learning pipelines for climate, providing standardized data loading and evaluation for ERA5 and CMIP6. Relevant if your workflow is ML-first.

### Where `climval` fits

climval sits at a different layer:

| | ESMValTool | PMP | **climval** |
|---|---|---|---|
| Installation | Complex, conda-heavy | Moderate | `pip install climval` |
| Configuration | YAML recipes | Config files | Pure Python API |
| Learning curve | Days to weeks | Days | Minutes |
| CI/CD integration | Difficult | Difficult | Native |
| Custom metrics | Recipe-based | Limited | `class MyMetric(BaseMetric)` |
| Programmatic use | Partial | Partial | First-class |
| Scope | Full diagnostic suite | Performance metrics | Composable validation |

**Use climval when you need:**
- A reproducible validation step inside a Python pipeline or CI workflow
- A composable API to integrate climate metrics into larger systems
- Rapid prototyping, teaching, or lightweight institutional reporting
- Full programmatic control without YAML configuration or recipe management

**Use ESMValTool when you need:**
- Pre-built diagnostic recipes from the CMIP community
- Automated regridding and unit harmonization across large multi-model ensembles
- Figures and diagnostics that conform to IPCC or CMIP reporting standards

climval is scoped to be the layer you can `import` in three lines.

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

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/northflowlabs/climval
cd climval
pip install -e ".[dev]"
pytest && ruff check . && mypy climval/
```

## Licence

climval is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 Northflow Technologies AS
