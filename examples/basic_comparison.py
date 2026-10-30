# SPDX-License-Identifier: Apache-2.0
"""
examples/basic_comparison.py
============================
Compare a candidate with a reference on different grids, calendars and units.

The two fields come from climval.demo_data(): a synthetic reference on a 4
degree grid (standard calendar, K) and a synthetic candidate on a 5 degree
grid (360_day calendar, degC, longitudes 0..360, latitude descending). The
scores describe these synthetic fields only. With real data, pass xarray
objects or paths and URLs (NetCDF, Zarr) instead.

Run:
    python examples/basic_comparison.py
"""

import climval

reference, candidate = climval.demo_data()

result = climval.compare(
    reference,
    candidate,
    variable="tas",
    region=climval.Box(lat=(35.0, 72.0), lon=(-10.0, 40.0)),  # Europe
    period=("2000", "2003"),
    by="season",
    names=("synthetic reference", "synthetic candidate"),
)

result.summary()

# Pass or fail limits, as a CI gate would use them
verdict = result.check({"rmse": {"max": 2.0}, "pearson_r": {"min": 0.9, "sample": "climatology"}})
print()
print(verdict.summary())

result.export("results/europe_2000_2003.json")
result.export("results/europe_2000_2003.html")
result.export("results/europe_2000_2003.md")
print("Done. Results written to results/")
