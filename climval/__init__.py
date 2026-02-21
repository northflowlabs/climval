"""
climval — Climate Model Validation Framework
=============================================
An open, composable benchmarking framework for comparing climate model outputs.

Designed for researchers, institutions, and infrastructure teams working with
CMIP6, ERA5, CORDEX, and custom climate simulation data. Follows CF Conventions
(v1.10) and implements metrics from the peer-reviewed literature.

Quickstart
----------
    from climval import BenchmarkSuite, load_model

    era5 = load_model("era5", path="data/era5.nc")
    mpi  = load_model("cmip6-mpi", path="data/mpi.nc")

    suite = BenchmarkSuite(name="Europe-2000-2020")
    suite.register(era5, role="reference")
    suite.register(mpi)

    report = suite.run(variables=["tas", "pr"])
    report.summary()
    report.export("results/report.html")

References
----------
- Gleckler et al. (2008), J. Geophys. Res., 113, D06104
- Taylor (2001), J. Geophys. Res., 106, 7183-7192
"""

__version__ = "0.1.0"
__author__ = "Northflow Technologies"
__license__ = "Apache-2.0"

from climval.core.loader import load_model
from climval.core.report import BenchmarkReport
from climval.core.suite import BenchmarkSuite

__all__ = [
    "BenchmarkSuite",
    "load_model",
    "BenchmarkReport",
]
