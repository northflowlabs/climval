# Samples and groups

A metric value always depends on which points it pools. climval names these
point sets *samples* and computes each requested sample for every group.

## Samples

Let \(r\) and \(c\) be the aligned reference and candidate, \(w_s\) the
spatial weight of grid cell \(s\) (its area by default), \(t\) a time step
and \(L_t\) its time weight: the length of its period in days, in the
reference's calendar, for monthly and yearly data, and 1 for daily and
sub-daily data. Points where either input is missing carry zero weight.

| Sample | Points pooled | Weights | Result |
|---|---|---|---|
| `space_time` | every valid point \((s, t)\) of the group | \(w_s L_t\) | one value per group |
| `climatology` | the time-mean fields \(\bar r_s, \bar c_s\) of the group (means weighted by \(L_t\)), compared over \(s\) | \(w_s\) | one value per group |
| `space` | the cells \(s\) at each time step | \(w_s\) | a time series |
| `time` | the time steps \(t\) at each cell | \(L_t\) | a map per group |

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
and August fields of the period.

Seasons are contiguous by default (`seasons="contiguous"`): a December
belongs to the DJF of the following year, and a season counts only when all
three of its months are present. For daily and sub-daily data a month counts
as present only when at least `min_coverage` (default 0.8) of its time steps
are, so a single day of December does not complete a winter. For the period 1995 to 2014 that leaves out
January and February 1995 (their December is outside the period) and
December 2014 (its January and February are), so DJF holds 19 winters. The
seasons left out are listed in `result.provenance["metric_notes"]["seasons"]`,
here `DJF 1995` and `DJF 2015`. The group `all` keeps every month.
`seasons="calendar"` instead pools every December, January and February of
the period into DJF, as xarray's `time.season` grouping does.

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
