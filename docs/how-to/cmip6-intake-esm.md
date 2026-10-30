# CMIP6 through intake-esm

The [Pangeo CMIP6 collection](https://pangeo-data.github.io/pangeo-cmip6-cloud/)
holds CMIP6 output as Zarr stores in Google Cloud Storage, indexed by an
intake-esm catalogue. Reading it needs no credentials.

```bash
pip install "climval[cloud,catalog]"
```

## Query the catalogue

```python
import intake

cat = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
query = cat.search(
    experiment_id="historical",
    table_id="Amon",
    variable_id="tas",
    source_id=["MPI-ESM1-2-LR", "NorESM2-LM"],
    member_id="r1i1p1f1",
)
query.df[["source_id", "member_id", "grid_label", "zstore"]]
```

## Open and compare

```python
dsets = query.to_dataset_dict(
    xarray_open_kwargs={"consolidated": True, "chunks": {}},
    storage_options={"token": "anon"},
    progressbar=False,
)
for key, ds in dsets.items():
    tas = ds["tas"].squeeze(drop=True)     # drop the size-1 member_id and dcpp_init_year
    result = climval.compare(era5_monthly, tas, region=box, period=("1995", "2014"), by="season")
```

intake-esm adds `member_id` (and `dcpp_init_year`) as dimensions. Squeeze
them for one member, or keep `member_id` to score every member in one call:
it is an extra dimension and climval keeps it.

## Several models in one call

Models come on different grids and calendars, so they cannot be stacked
directly. Align each one with the reference first, then stack the aligned
candidates:

```python
import pandas as pd
import xarray as xr

aligned = {}
for source, tas in models.items():
    pair = climval.align(era5_monthly, tas, region=box, period=("1995", "2014"), target="reference")
    aligned[source] = pair.candidate
stack = xr.concat(list(aligned.values()), dim=pd.Index(list(aligned), name="model"))
result = climval.compare(pair.reference, stack, by="season")
```

The [Europe example](../examples/era5_cmip6_europe.ipynb) runs this for five
models.

## In a configuration file

`climval check` takes a catalogue query as an input:

```yaml
candidate:
  catalog: https://storage.googleapis.com/cmip6/pangeo-cmip6.json
  query:
    source_id: NorESM2-LM
    experiment_id: historical
    table_id: Amon
    variable_id: tas
    member_id: r1i1p1f1
  storage_options: {token: anon}
```

The query must match exactly one dataset after intake-esm's aggregation.
Dimensions of size one that come from catalogue columns (`member_id`,
`dcpp_init_year`) are squeezed; a query for several members keeps
`member_id` as a dimension and gives one value per member.

## Cite the data

CMIP6 output is licensed under CC BY 4.0. Cite each model's `source_id` and
institution (the dataset attributes `source_id`, `institution_id` and
`institution`, recorded in climval's provenance), the CMIP6 overview
(Eyring et al. 2016, <https://doi.org/10.5194/gmd-9-1937-2016>) and include
the WCRP acknowledgement.
