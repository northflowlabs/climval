# SPDX-License-Identifier: Apache-2.0
"""
climval: validate climate model output against reference data
==============================================================

climval aligns a candidate with a reference (grid, longitude convention,
region, calendar, frequency, units, missing values, area weights) and then
computes weighted metrics with xarray, lazily on dask arrays and in one dask
computation. Every result carries a provenance record of what was compared
and how.

Quickstart
----------
    import climval

    result = climval.compare(
        "gs://.../era5_tas.zarr",          # reference: path, URL, Dataset or DataArray
        cmip6_ds,                           # candidate, for example from intake-esm
        variable=("t2m", "tas"),
        region=climval.Box(lat=(35, 72), lon=(-10, 40)),
        period=("1995", "2014"),
        by="season",
    )
    result.summary()
    result.to_dataframe()
    result.export("report.html")

References
----------
- Gleckler et al. (2008), J. Geophys. Res., 113, D06104
- Taylor (2001), J. Geophys. Res., 106, 7183-7192
- Gupta et al. (2009), J. Hydrol., 377, 80-91
"""

__version__ = "0.2.0"
__author__ = "Northflow Technologies AS"
__license__ = "Apache-2.0"

from climval import accessor as _accessor  # noqa: F401  registers .climval
from climval.align import AlignedPair, align
from climval.align.region import Box, MaskRegion
from climval.api import compare, demo, demo_data
from climval.core.loader import load_model
from climval.core.report import BenchmarkReport
from climval.core.suite import BenchmarkSuite
from climval.result import Comparison
from climval.thresholds import ThresholdResult, evaluate_thresholds

__all__ = [
    "compare",
    "Comparison",
    "align",
    "AlignedPair",
    "Box",
    "MaskRegion",
    "demo",
    "demo_data",
    "evaluate_thresholds",
    "ThresholdResult",
    "BenchmarkSuite",
    "load_model",
    "BenchmarkReport",
    "__version__",
]
