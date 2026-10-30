# Samples and groups

A metric value always depends on which points it pools. climval names these
point sets *samples* and computes each requested sample for every group.

## Samples

Let \(r\) and \(c\) be the aligned reference and candidate, \(w_s\) the
spatial weight of grid cell \(s\) and \(t\) a time step. Points where either
input is missing carry zero weight.

| Sample | Points pooled | Weights | Result |
|---|---|---|---|
| `space_time` | every valid point \((s, t)\) of the group | \(w_s\) | one value per group |
| `climatology` | the time-mean fields \(\bar r_s, \bar c_s\) of the group, compared over \(s\) | \(w_s\) | one value per group |
| `space` | the cells \(s\) at each time step | \(w_s\) | a time series |
| `time` | the time steps \(t\) at each cell | equal | a map per group |

The default is `samples=("space_time", "climatology")`.

`climatology` is the statistic of a Taylor diagram (Taylor 2001) and of
Gleckler et al. (2008): both fields are averaged over the paired time steps of
the group first, then their spatial patterns are compared. For a free-running
simulation this is usually the sample to read, because it does not ask the
model to reproduce the observed sequence of weather.

`space_time` compares every time step at every point. For a free-running
model it mixes model error with internal variability; it suits forecasts,
nudged runs, reanalysis intercomparisons and downscaling against the driving
data.

`space` gives the pattern statistics at each time step, for example a time
series of pattern correlation. `time` gives per-point temporal statistics,
for example a map of mean bias or of temporal correlation.

## Groups

`by="season"` adds the groups DJF, MAM, JJA and SON, `by="month"` adds Jan to
Dec and `by="year"` adds one group per year. The group `all` holds the whole
period and is always present. A group selects time steps by their calendar
month or year; the `climatology` of group JJA is the mean of the June, July
and August fields of the period, and DJF holds the December, January and
February fields.

## Extra dimensions

Dimensions other than `lat`, `lon` and `time` are kept. A candidate with a
`member` or `model` dimension gives one value per member or model, and
`to_dataframe()` adds a column for that dimension.

```python
result = climval.compare(era5, ensemble, by="season")   # ensemble has dim "member"
table = result.to_dataframe()
table[(table["sample"] == "climatology") & (table["metric"] == "pearson_r")]
```

## Reading values

```python
result.value("rmse")                                     # space_time, all
result.value("pearson_r", sample="climatology", group="DJF")
result.values("climatology", "JJA")                     # dict of every metric
result.field("space")["pearson_r"]                      # time series
result.field("time")["mean_bias"].sel(group="DJF")      # map
```
