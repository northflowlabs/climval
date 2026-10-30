# Pangeo and cloud data

climval reads data lazily and never loads a whole dataset. On cloud data the
usual pattern is: open the store with xarray, select what you need, and pass
the Dataset or DataArray to `climval.compare`. Only the chunks that hold the
selected region and period are read, when the metrics are computed.

```bash
pip install "climval[cloud]"
```

## Open the store yourself

```python
import xarray as xr
import climval

era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    storage_options={"token": "anon"},   # public bucket, no credentials
    chunks={},                           # keep the stored chunks
)
t2m = era5["2m_temperature"].sel(time=slice("2010-01-01", "2010-12-31"))
```

Opening the store yourself is the most reliable route for remote Zarr: you
choose the engine, the storage options and the chunks, and you can check
sizes before anything is read.

## Paths and URLs

`compare` also accepts paths and URLs. NetCDF and Zarr open lazily, local or
remote through fsspec (`gs://`, `s3://`, `https://`). Storage options go in
`open_kwargs`:

```python
result = climval.compare(
    "gs://my-bucket/reference.zarr",
    "s3://my-bucket/candidate.zarr",
    variable="tas",
    open_kwargs={"storage_options": {"anon": True}},
)
```

`open_kwargs` applies to both inputs. When they need different options, open
them yourself.

## Check sizes before you read

Cloud stores are chunked, and a chunk is read as a whole. A store chunked as
one global field per time step reads the whole globe even for a small region.
Look at the chunking and estimate the bytes first:

```python
da = era5["2m_temperature"]
print(da.encoding["chunks"], da.dtype)
print(da.sel(time=slice("2010", "2010")).nbytes / 1e6, "MB uncompressed for one year")
```

With gcsfs the compressed size of each chunk is in the bucket listing:
`gcsfs.GCSFileSystem(token="anon").ls(path, detail=True)`. The
[Europe example](../examples/era5_cmip6_europe.ipynb) does this before
reading.

## Keep memory modest

- Subset in space and time before computing; `compare` does this for its
  region and period, and dask reads only the chunks involved.
- `compare(..., compute=False)` returns a lazy result; `.compute()` runs one
  dask pass for every metric and sample.
- Limit the threads when memory is tight:
  `with dask.config.set(scheduler="threads", num_workers=4): result.compute()`.
- On a cluster, create a `dask.distributed.Client` first; climval uses the
  active scheduler.

## Sub-daily data

Hourly or 6-hourly data are aggregated to the common frequency with a time
mean before matching (see [Calendars and frequencies](calendars.md)). For
long periods it is often cheaper to compute daily or monthly means once,
store them, and compare those.
