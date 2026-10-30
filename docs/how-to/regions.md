# Regions

## A latitude and longitude box

```python
europe = climval.Box(lat=(35, 72), lon=(-10, 40), name="Europe")
result = climval.compare(ref, cand, "tas", region=europe)
```

`lat` is `(south, north)`. `lon` is `(west, east)`, read eastward from `west`
to `east`, in either convention: `(-10, 40)` and `(350, 40)` are the same box.
`west > east` crosses the antimeridian:

```python
pacific = climval.Box(lat=(-20, 20), lon=(160, -150))   # 160E eastward to 150W
band = climval.Box(lat=(60, 90))                         # all longitudes
```

A dict works too: `region={"lat": (35, 72), "lon": (-10, 40)}`.

The box is cut before regridding: the target grid exactly, the grid that is
regridded with a margin of one cell (and wrapped columns when it is global)
so the points at the edge have neighbours. The step records how many points
each side kept.

## regionmask regions

With [regionmask](https://regionmask.readthedocs.io) installed
(`pip install "climval[regions]"`), name a region as `"<collection>:<region>"`:

```python
result = climval.compare(ref, cand, "tas", region="ar6.land:NEU")   # IPCC AR6 Northern Europe, land
result = climval.compare(ref, cand, "tas", region="ar6.all:MED")    # Mediterranean, land and sea
```

The collection is an attribute path under `regionmask.defined_regions`
(`ar6.land`, `ar6.all`, `ar6.ocean`, `srex`, `giorgi` and others); the region
is its abbreviation, name or number. climval cuts the region's bounding box
before regridding and applies the polygon mask after regridding, on the
target grid; the `region_mask` step records how many cells lie inside.

You can also pass a regionmask region object with `climval.MaskRegion`:

```python
import regionmask

neu = climval.MaskRegion(regionmask.defined_regions.ar6.land, "NEU")
```

The IPCC AR6 reference regions are described by Iturbide et al. (2020),
<https://doi.org/10.5194/essd-12-2959-2020>.

## Land or sea only

For a land-only comparison without a named region, mask the reference with a
land fraction before the comparison; the paired mask then removes the same
points from the candidate:

```python
land = era5["land_sea_mask"] > 0.5
result = climval.compare(t2m.where(land), tas, region=europe)
```

## In a configuration file

```yaml
region: {lat: [35, 72], lon: [-10, 40], name: Europe}
region: [35, 72, -10, 40]          # south, north, west, east
region: [60, 90]                   # south, north
region: "ar6.land:NEU"
```
