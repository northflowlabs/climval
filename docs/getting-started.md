# Getting started

## Install

```bash
pip install climval                      # NumPy, SciPy, pandas, xarray, dask, cftime, PyYAML
pip install "climval[cloud]"             # Zarr, fsspec, gcsfs, s3fs
pip install "climval[cloud,catalog]"     # plus intake-esm catalogues
```

Other extras: `netcdf` (netCDF4, h5netcdf), `units` (pint, cf_xarray),
`regions` (regionmask), `all`. Conservative
regridding needs xESMF, which depends on the ESMF library and is best
installed with conda: `conda install -c conda-forge xesmf`.

climval needs Python 3.10 or newer.

## A first comparison without data

`climval.demo()` compares two small synthetic fields that differ in every way
alignment has to handle: grid spacing, longitude convention, latitude order,
calendar and units. Its results are flagged as synthetic in every output.

```python
import climval

result = climval.demo(by="season")
result.summary()
```

## A comparison of your own data

```python
import climval
import xarray as xr

reference = xr.open_dataset("era5_t2m_monthly.nc", chunks={})
candidate = xr.open_dataset("tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001-201412.nc", chunks={})

result = climval.compare(
    reference,
    candidate,
    variable=("t2m", "tas"),          # (reference name, candidate name)
    region=climval.Box(lat=(35, 72), lon=(-10, 40)),
    period=("1995", "2014"),
    by="season",
)
```

`compare` puts both inputs on one grid (by default the coarser of the two),
on common months and in the reference's units (see
[Alignment](concepts/alignment.md)), computes every metric in one dask pass
and returns a [`Comparison`](reference/compare.md). Spatial means are
weighted by cell area and monthly steps by their length in days; with
`by="season"` a December counts in the following winter's DJF.

## The result

```python
result.summary()            # printed table and the alignment steps
result.to_dataframe()       # long table: sample, group, metric, value, units
result["rmse"]              # one value: space_time sample, group "all"
result.value("pearson_r", sample="climatology", group="JJA")
result.field("time")        # per-grid-point maps, one per group
result.reference            # the aligned reference field
result.candidate            # the aligned candidate field
result.provenance           # inputs, steps, versions, as a dict
result.export("report.html")   # or .json, .md
```

In Jupyter the result displays as a table.

## Pass or fail

```python
verdict = result.check({
    "rmse": {"max": 2.0},
    "mean_bias": {"abs_max": 1.0, "group": "*"},
    "pearson_r": {"min": 0.95, "sample": "climatology"},
})
verdict.passed
print(verdict.summary())
```

The same limits, written in a configuration file, run from the command line
with `climval check` and from GitHub Actions; see
[Thresholds and the CI gate](how-to/ci-gate.md).

## From the command line

```bash
climval compare era5.nc tas.nc --variable t2m,tas --by season --output report.html
climval check climval.yml --output-dir climval-report
climval demo
climval list-metrics
```
