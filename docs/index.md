# climval

climval validates climate model output against reference data. It puts both
datasets on the same grid, period and units, computes weighted metrics with
xarray, lazily on dask arrays, and records how every number was made. A
configuration file and the `climval check` command turn the same comparison
into a pass or fail gate for continuous integration.

```python
import climval

result = climval.compare(
    era5,                      # reference: Dataset, DataArray, path or URL
    cmip6_tas,                 # candidate, for example from intake-esm
    variable=("t2m", "tas"),
    region=climval.Box(lat=(35, 72), lon=(-10, 40)),
    period=("1995", "2014"),
    by="season",
)
result.to_dataframe()
result.check({"pearson_r": {"min": 0.95, "sample": "climatology"}})
```

## What it is for

climval is the validation step between model output and a decision: a
notebook that compares a run with a reanalysis, a pipeline that checks every
new simulation, a pull request that must not make a model worse. It works on
the data you already have open in Python, including Zarr stores in cloud
object storage and intake-esm catalogues such as the
[Pangeo CMIP6 collection](https://pangeo-data.github.io/pangeo-cmip6-cloud/).

It is not a diagnostic suite. For recipe-driven, multi-diagnostic campaigns
across CMIP ensembles use [ESMValTool](https://www.esmvaltool.org/); for the
standard PCMDI performance metrics use the
[PCMDI Metrics Package](https://github.com/PCMDI/pcmdi_metrics); for land
model benchmarking use [ILAMB](https://www.ilamb.org/). climval does one
comparison of two datasets carefully and says exactly what it did.

## Where to go next

- [Getting started](getting-started.md): install, the first comparison, the
  result object.
- [Alignment](concepts/alignment.md): what each alignment step does and why.
- [Samples and groups](concepts/samples.md) and [Metrics](concepts/metrics.md):
  what each number is computed over, with formulas.
- [Thresholds and the CI gate](how-to/ci-gate.md) and
  [the GitHub Action](how-to/github-action.md).
- [ERA5 against CMIP6 over Europe](examples/era5_cmip6_europe.ipynb): five
  CMIP6 models against ERA5, read anonymously from public cloud storage.
- [Migrating from 0.1](how-to/migrating.md): what changed, including the
  defaults that change numbers.

climval is open source under the Apache License 2.0 and maintained by
[Northflow Technologies AS](https://northflow.no).
