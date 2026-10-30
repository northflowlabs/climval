# Provenance

Every `Comparison` carries a JSON-serialisable record of how its numbers were
made, in `result.provenance`. It is written into every JSON, HTML and Markdown
export and into the JSON result of `climval check`.

| Key | Content |
|---|---|
| `created` | UTC timestamp |
| `environment` | climval, Python and platform, and the versions of NumPy, xarray, pandas, SciPy, dask, cftime, Zarr, fsspec and the optional libraries in use |
| `inputs` | for each side: kind (path, URL, Dataset, DataArray), location, variable, dimensions, shape, dtype, chunks, coordinate ranges, a SHA-256 of the coordinates, variable attributes (units, standard name, cell methods) and CF or CMIP global attributes such as `source_id`, `experiment_id`, `member_id`, `variant_label`, `institution_id` and `tracking_id` |
| `options` | the options of the call (`region`, `period`, `regrid`, `target`, `time`, `freq`, `min_coverage`, `units`, `weights`, `by`, `seasons`, `samples`, `metrics`) |
| `steps` | one record per alignment step, in order: `inputs`, `coordinates`, `target_grid`, `longitude`, `region`, `time`, `regrid`, `region_mask` (regionmask regions), `units`, `missing`, `weights` |
| `metric_notes` | notes on the metrics, for example the seasons left out with `seasons="contiguous"` and the Kelvin shift for KGE and percentile bias |
| `metric_methods` | per sample, how each percentile was computed (`exact` or `histogram`) |
| `aligned_values_sha256` | hashes of a strided sample of the aligned reference and candidate, taken during the one compute pass |

Input values are hashed before alignment only when they are already in memory;
lazy inputs are not read for a hash, and the aligned-value hashes stand in for
them. The hashes use values rounded to 36 mantissa bits, so they are stable
across platforms but change when the data or the alignment changes.

```python
import json

print(json.dumps(result.provenance["steps"], indent=2))
for line in climval.result.describe_steps(result.provenance["steps"]):
    print(line)
```

## What provenance is for

- **Reproducing a number.** The options and steps say exactly which points
  were compared; the input records say which data.
- **Reviewing a comparison.** A reader can see that a `360_day` model was
  matched by calendar month, which grid the comparison used and which input
  was regridded onto it, or that units were converted from degC to K.
- **Comparing runs.** Two JSON results can be compared field by field, for
  example in a CI job that keeps the previous result as an artifact.
