"""
examples/basic_comparison.py
==============================
Quickstart: compare ERA5 (reference) against two CMIP6 models
using synthetic data (no files required).

Run:
    python examples/basic_comparison.py
"""

from datetime import datetime
from climval import BenchmarkSuite, load_model
from climval.metrics import PercentileBias, TaylorSkillScore

# ── 1. Load models ────────────────────────────────────────────────────────────

era5 = load_model(
    "era5",
    name="ERA5",
    lat_range=(35.0, 72.0),          # Europe
    lon_range=(-10.0, 40.0),
    time_start=datetime(2000, 1, 1),
    time_end=datetime(2020, 12, 31),
)

mpi = load_model(
    "cmip6-mpi",
    name="MPI-ESM1-2-HR",
    lat_range=(35.0, 72.0),
    lon_range=(-10.0, 40.0),
    time_start=datetime(2000, 1, 1),
    time_end=datetime(2020, 12, 31),
)

cordex = load_model(
    "cordex-eur",
    name="CORDEX-EUR-11",
    lat_range=(35.0, 72.0),
    lon_range=(-10.0, 40.0),
    time_start=datetime(2000, 1, 1),
    time_end=datetime(2020, 12, 31),
)

# ── 2. Build suite ────────────────────────────────────────────────────────────

suite = BenchmarkSuite(name="Europe-2000-2020")
suite.register(era5, role="reference")
suite.register(mpi)
suite.register(cordex)

# Add extra metrics (optional — defaults already include RMSE, MAE, bias, r, TSS)
suite.add_metric(PercentileBias(percentile=99.0))

# ── 3. Run ────────────────────────────────────────────────────────────────────

report = suite.run(
    variables=["tas", "pr"],
    n_samples=2000,   # synthetic sample size (ignored when real files are used)
    seed=42,
)

# ── 4. Output ─────────────────────────────────────────────────────────────────

report.summary()
report.export("results/europe_2000_2020.json")
report.export("results/europe_2000_2020.html")
report.export("results/europe_2000_2020.md")

print("Done. Results written to results/")
