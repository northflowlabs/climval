# Alignment

Most of the work in a model comparison, and most of the mistakes, come before
the metric. Two datasets rarely share a grid, a longitude convention, a
calendar, a time step or units, and missing values seldom fall on the same
points. climval does these steps explicitly, in a fixed order, and writes each
one into the result's provenance (`result.provenance["steps"]`).

`climval.align(reference, candidate, ...)` runs the steps without computing a
metric and returns an `AlignedPair`; `climval.compare` calls it and then
computes the metrics. On dask arrays every step is lazy and no data value is
read. Alignment reads coordinates and, when an input is a Dataset or a file
whose time, latitude or longitude bounds are stored as dask arrays, those
bounds variables, all of them in one small `dask.compute` call, because the
native frequency and the cell areas come from them. Inputs are cast to
float64, lazily, so lazy and eager runs of float32 data give the same values.

## 1. Inputs

Each side may be an `xarray.DataArray`, an `xarray.Dataset` with `variable=`,
a path or URL to NetCDF, Zarr or CSV, or a list of NetCDF paths. NetCDF and
Zarr paths and URLs open lazily with dask chunks (dask is a core dependency);
CSV tables are read into memory. A Zarr store is recognised by a `.zarr`
suffix (also `.zarr-v3` and similar) or, without one, by its `.zmetadata`,
`.zgroup` or `zarr.json`, so Pangeo CMIP6 store URLs, which end in a version
directory, work as given. `variable` is one name, or a pair
`(reference name, candidate name)` when the names differ, for example
`("t2m", "tas")`.

A CSV whose latitude and longitude columns do not form a full grid, or any
input with `lat` and `lon` on one shared dimension, is station data on a
`station` dimension. The gridded input is then sampled at the stations
(bilinear or nearest), and the stations are always the target.

*Why:* the comparison should start from the data you have, without a copy.

## 2. Coordinates

climval finds latitude, longitude and time through cf_xarray when it is
installed, otherwise through CF attributes (`standard_name`, `units`, `axis`)
and common names. They are renamed to `lat`, `lon` and `time`, and latitude is
sorted ascending. Two-dimensional latitude and longitude (a curvilinear grid)
is detected; such a grid can only be remapped with `regrid="conservative"`
(xESMF) and cannot be the target.

*Why:* every later step needs to know which dimension is which, and arithmetic
in xarray aligns on coordinate names and values.

## 3. Target grid

The comparison happens on one grid: `target="coarser"` (default, the input
with larger cells, measured as latitude spacing times longitude spacing, so
one axis of the target can be finer than the other input's; the reference
when both are equal), `"reference"`,
`"candidate"`, or an xarray object that carries the grid. The other input is
regridded onto it. A curvilinear input can only be the source, so with
`"coarser"` the rectilinear input is the target. The `target_grid` step
records which grid was used.

*Why:* interpolating a coarse field onto a fine grid adds no information,
and the fine field then keeps small-scale detail that the coarse one cannot
have, which counts against the coarse model. The choice changes the numbers,
so it is recorded. When several models are scored against one reference with
the default, each model is compared on whichever of its grid and the
reference grid has the larger cells; pass `target="reference"` or a common grid when all of them
should share one.

## 4. Longitude

Both inputs are expressed in one longitude frame: the target grid's
convention (0..360 or -180..180), or a frame that keeps the requested region
in one piece when it crosses the 0 or 180 meridian.

*Why:* a box such as 10W to 40E is two separate pieces in a 0..360 grid.

## 5. Region

A `climval.Box(lat=(south, north), lon=(west, east))` is cut before
regridding. `west > east` crosses the antimeridian. The target grid is cut
exactly. The grid that will be regridded keeps a margin of one cell, and for
bilinear and nearest a wrapped column at each end when it is global and the
box reaches the edge of the longitude frame, so the points at the edge of the
box have neighbours to interpolate from. The wrapped columns are added after
the time step, so they are aggregated once rather than split off every
native time step.
Conservative remapping gets no wrapped columns; xESMF treats a global source
as periodic instead. A regionmask region (for example
`"ar6.land:NEU"`) contributes its bounding box here and its polygon mask after
regridding, on the target grid. See [Regions](../how-to/regions.md).

*Why:* cutting first keeps the amount of data read small; the margin avoids
extrapolation at the edges.

## 6. Time

The time step works on calendar fields, not timestamps. See
[Calendars and frequencies](../how-to/calendars.md) for the details.

1. Each input's native frequency comes from its time bounds (`time_bnds`,
   CF `bounds`), then a `frequency` attribute (CMIP vocabulary such as `mon`,
   `day`, `3hr`, or a pandas alias), then the time steps. With bounds, steps
   are labelled by the midpoints of their intervals, so values stamped at
   the end of their interval count for the right day. When none of these
   tells the frequency (a single time step, one month per year, irregular
   steps), climval raises and asks for `freq=`, time bounds or a `frequency`
   attribute.
2. The `period` is selected on year, month and day, so `("1995", "2014")`
   means the same months in a `360_day`, `noleap` or standard calendar.
3. The common frequency is the coarser of the two native frequencies, or
   `freq=` (never finer than either). Sub-daily data keep their step,
   written `"<N>h"` for a whole number of hours that divides 24: hourly
   against 3-hourly data is compared in 3-hour bins.
4. A finer input is aggregated to that frequency with a time mean (months
   aggregated to years are weighted by their length). A period's value counts
   only where at least `min_coverage` (default 0.8) of its native steps hold
   a valid value, per grid cell; the rule is recorded. On dask arrays the time
   chunks are first made to hold whole periods; a regular sub-daily series
   (hourly to daily, say) is then summed with one task per chunk, and other
   series are resampled through flox, which also reduces chunk by chunk. The
   result is rechunked to whole horizontal fields and dask's automatic size
   along time. A year of hourly ERA5 at 0.25 degrees, in the time-series,
   tiled or one-hour-per-chunk layout, aggregates to days in a graph of tens
   of thousands of tasks.
5. Each period is labelled by its calendar fields (year and month for monthly
   data) and only labels present in both inputs are kept. With no common
   period climval raises `NoOverlapError`.

Daily and sub-daily comparisons between a `360_day` calendar and any other
calendar warn: days are matched by date label, so 30 February and every 31st
day drop out, and the same label sits at a different point of the seasonal
cycle. `freq="MS"` compares monthly means instead.

*Why:* a model month stamped on day 16 of a `360_day` month and a reanalysis
month stamped on day 1 of a Gregorian month are the same month, and matching
timestamps would miss that.

## 7. Regridding

`regrid="bilinear"` (default) and `"nearest"` use `xarray.interp`.
`"conservative"` (first order, `conservative_normed`) always goes through
xESMF; it is the only option for curvilinear grids and the only one that
averages a fine grid over coarse cells. A source cell reaches half a spacing
beyond its centre, so target points in that half cell take the edge value;
target points farther out, or inside a hole in the source grid, become NaN.
climval does not extrapolate. When no target point lies on the source grid
along latitude or longitude, it raises `NoOverlapError` instead of returning
NaN values. Bilinear or nearest interpolation onto a coarser grid samples the
source at the target points; climval warns and suggests
`regrid="conservative"`. A dask source is rechunked to whole horizontal
fields before interpolation, so a source stored in small spatial tiles does
not turn into millions of tiny chunks.

*Why:* bilinear interpolation is a reasonable default for smooth fields such
as temperature; conservative remapping keeps area integrals and suits fluxes
such as precipitation.

## 8. Units

The candidate is converted into the reference's units. Built in: K, degC and
degF; kg m-2 s-1, kg m-2 h-1, kg m-2 d-1, mm/h and mm/day; Pa and hPa;
fractions and percent. A bare `C` is the coulomb in UDUNITS and CF, so it is
rejected with a hint to write `degC`. pint handles
other units when installed. An unknown mismatch raises `UnitsError` unless you
pass `units="ignore"` or declare the units with
`units={"reference": "K", "candidate": "degC"}`. See [Units](../how-to/units.md).

*Why:* a difference between K and degC is a bias of 273.15, and it happens.

## 9. Missing values

A point counts only where both inputs are valid. Both fields are set to NaN
wherever either one is missing.

*Why:* without a paired mask, a mean bias compares two different sets of
points.

## 10. Weights

Every spatial mean is weighted. `weights="area"` (default) uses a cell area
field when one is given with `area=` or found through the reference's
`cell_measures` attribute and lies on the target grid. Otherwise it computes
the area of each cell from the target grid's latitude and longitude bounds
(`lat_bnds` and `lon_bnds` when the Dataset has them, else midpoints between
neighbouring centres, clipped at the poles; longitude bounds that wrap around
the seam, such as 358.75 to 1.25, count as 2.5 degrees):

\[
A = R^2 \, \Delta\lambda \, (\sin\varphi_{north} - \sin\varphi_{south})
\]

On a uniform grid this is proportional to cos(latitude); on a Gaussian or
otherwise non-uniform grid it is not. `weights="cos_lat"` gives cos(latitude),
`weights=None` equal weights, and a DataArray is used as given (it must not
be negative). Station data get equal weights. The weights actually used are
recorded in the `weights` step.

For monthly and yearly comparisons each time step also carries a time
weight, the length of its period in days in the reference's calendar, so a
February counts less than a January wherever time steps are pooled. Daily
and sub-daily steps count equally.

*Why:* on a latitude-longitude grid a cell at 70N covers about a third of the
area of a cell at the equator.

## Inspecting the alignment

```python
pair = climval.align(reference, candidate, ("t2m", "tas"), region=box, period=("1995", "2014"))
pair.reference, pair.candidate, pair.weights
for line in climval.result.describe_steps(pair.steps):
    print(line)
```

Aligned candidates on one common grid (for example all aligned with
`target="reference"`) can be stacked along a new `model` dimension and passed
to `compare`; extra dimensions are kept. See
[CMIP6 through intake-esm](../how-to/cmip6-intake-esm.md#several-models-on-one-common-grid).
