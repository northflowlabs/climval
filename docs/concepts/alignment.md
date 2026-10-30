# Alignment

Most of the work in a model comparison, and most of the mistakes, come before
the metric. Two datasets rarely share a grid, a longitude convention, a
calendar, a time step or units, and missing values seldom fall on the same
points. climval does these steps explicitly, in a fixed order, and writes each
one into the result's provenance (`result.provenance["steps"]`).

`climval.align(reference, candidate, ...)` runs the steps without computing a
metric and returns an `AlignedPair`; `climval.compare` calls it and then
computes the metrics. Nothing is loaded into memory during alignment: on dask
arrays every step is lazy.

## 1. Inputs

Each side may be an `xarray.DataArray`, an `xarray.Dataset` with `variable=`,
a path or URL to NetCDF, Zarr or CSV, or a list of NetCDF paths. Paths and
URLs open lazily with dask chunks when dask is installed. `variable` is one
name, or a pair `(reference name, candidate name)` when the names differ, for
example `("t2m", "tas")`.

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

The comparison happens on one grid: `target="reference"`, `"candidate"`,
`"coarser"` (the input with larger cells), or an xarray object that carries
the grid. The other input is regridded onto it. Pass `target` explicitly when
it matters to you; the default is listed in the
[`compare` reference](../reference/compare.md).

*Why:* interpolating a coarse field onto a fine grid adds no information, and
the choice changes the numbers. It is recorded so a reader can see it.

## 4. Longitude

Both inputs are expressed in one longitude frame: the target grid's
convention (0..360 or -180..180), or a frame that keeps the requested region
in one piece when it crosses the 0 or 180 meridian.

*Why:* a box such as 10W to 40E is two separate pieces in a 0..360 grid.

## 5. Region

A `climval.Box(lat=(south, north), lon=(west, east))` is cut before
regridding. `west > east` crosses the antimeridian. The target grid is cut
exactly. The grid that will be regridded keeps a margin of one cell, and a
wrapped column at each end when it is global, so the points at the edge of the
box have neighbours to interpolate from. A regionmask region (for example
`"ar6.land:NEU"`) contributes its bounding box here and its polygon mask after
regridding, on the target grid. See [Regions](../how-to/regions.md).

*Why:* cutting first keeps the amount of data read small; the margin avoids
extrapolation at the edges.

## 6. Time

The time step works on calendar fields, not timestamps. See
[Calendars and frequencies](../how-to/calendars.md) for the details.

1. The `period` is selected on year, month and day, so `("1995", "2014")`
   means the same months in a `360_day`, `noleap` or standard calendar.
2. Each input's native step is classified (hourly, daily, monthly, yearly) and
   the common frequency is the coarser of the two, or `freq=`.
3. A finer input is aggregated to that frequency with a time mean, and
   incomplete periods are dropped.
4. Each period is labelled by its calendar fields (year and month for monthly
   data) and only labels present in both inputs are kept. With no common
   period climval raises `NoOverlapError`.

*Why:* a model month stamped on day 16 of a `360_day` month and a reanalysis
month stamped on day 1 of a Gregorian month are the same month, and matching
timestamps would miss that.

## 7. Regridding

`regrid="bilinear"` (default) and `"nearest"` use `xarray.interp`.
`"conservative"` uses xESMF and is the only option for curvilinear grids.
Target points outside the source grid, or inside a hole in it, become NaN;
climval does not extrapolate.

*Why:* bilinear interpolation is a reasonable default for smooth fields such
as temperature; conservative remapping keeps area integrals and suits fluxes
such as precipitation.

## 8. Units

The candidate is converted into the reference's units. Built in: K, degC and
degF; kg m-2 s-1 and mm/day; Pa and hPa; fractions and percent. pint handles
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
`cell_measures` attribute and lies on the target grid; otherwise it falls back
to a latitude weight that is proportional to cell area on a regular grid.
`weights="cos_lat"` forces cos(latitude), `weights=None` gives equal weights
and a DataArray is used as given. The weights actually used are recorded in
the `weights` step.

*Why:* on a latitude-longitude grid a cell at 70N covers about a third of the
area of a cell at the equator.

## Inspecting the alignment

```python
pair = climval.align(reference, candidate, ("t2m", "tas"), region=box, period=("1995", "2014"))
pair.reference, pair.candidate, pair.weights
for line in climval.result.describe_steps(pair.steps):
    print(line)
```

The aligned pair can be stacked with others, for example several models along
a new `model` dimension, and passed to `compare`; extra dimensions are kept.
The [Europe example](../examples/era5_cmip6_europe.ipynb) does this.
