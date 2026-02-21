"""
climval.models.schema
================================
Typed data models for climate variables, model metadata, and benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ModelType(str, Enum):
    REANALYSIS = "reanalysis"
    GCM = "gcm"
    RCM = "rcm"
    DOWNSCALED = "downscaled"
    OBSERVATION = "observation"
    CUSTOM = "custom"


class TemporalResolution(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    ANNUAL = "annual"


class SpatialResolution(str, Enum):
    HIGH = "high"       # < 10 km
    MEDIUM = "medium"   # 10–50 km
    LOW = "low"         # > 50 km
    GLOBAL = "global"


@dataclass
class SpatialDomain:
    """Geographic bounding box in decimal degrees."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    description: str | None = None

    def area_deg2(self) -> float:
        return abs(self.lat_max - self.lat_min) * abs(self.lon_max - self.lon_min)

    def __str__(self) -> str:
        return (
            f"[{self.lat_min:.2f}°N–{self.lat_max:.2f}°N, "
            f"{self.lon_min:.2f}°E–{self.lon_max:.2f}°E]"
        )


@dataclass
class TemporalDomain:
    """Time range for the model output."""
    start: datetime
    end: datetime

    def duration_years(self) -> float:
        delta = self.end - self.start
        return delta.days / 365.25

    def __str__(self) -> str:
        return f"{self.start.strftime('%Y-%m')} → {self.end.strftime('%Y-%m')}"


@dataclass
class ClimateVariable:
    """A single climate variable definition."""
    name: str                        # e.g. "tas" (near-surface air temperature)
    long_name: str                   # e.g. "Near-Surface Air Temperature"
    units: str                       # e.g. "K"
    standard_name: str | None = None  # CF convention
    description: str | None = None


STANDARD_VARIABLES: dict[str, ClimateVariable] = {
    "tas": ClimateVariable(
        name="tas",
        long_name="Near-Surface Air Temperature",
        units="K",
        standard_name="air_temperature",
        description="Temperature at 2m above surface",
    ),
    "pr": ClimateVariable(
        name="pr",
        long_name="Precipitation",
        units="kg m-2 s-1",
        standard_name="precipitation_flux",
        description="Total precipitation rate",
    ),
    "psl": ClimateVariable(
        name="psl",
        long_name="Sea Level Pressure",
        units="Pa",
        standard_name="air_pressure_at_mean_sea_level",
    ),
    "ua": ClimateVariable(
        name="ua",
        long_name="Eastward Wind",
        units="m s-1",
        standard_name="eastward_wind",
    ),
    "va": ClimateVariable(
        name="va",
        long_name="Northward Wind",
        units="m s-1",
        standard_name="northward_wind",
    ),
    "hurs": ClimateVariable(
        name="hurs",
        long_name="Near-Surface Relative Humidity",
        units="%",
        standard_name="relative_humidity",
    ),
}


@dataclass
class ClimateModel:
    """
    Represents a loaded climate model or dataset ready for benchmarking.
    """
    name: str
    model_type: ModelType
    variables: list[ClimateVariable]
    spatial_domain: SpatialDomain
    temporal_domain: TemporalDomain
    spatial_resolution: SpatialResolution
    temporal_resolution: TemporalResolution
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        lines = [
            f"Model       : {self.name}",
            f"Type        : {self.model_type.value}",
            f"Variables   : {', '.join(v.name for v in self.variables)}",
            f"Spatial     : {self.spatial_domain}",
            f"Temporal    : {self.temporal_domain}",
            f"Resolution  : spatial={self.spatial_resolution.value}, "
            f"temporal={self.temporal_resolution.value}",
        ]
        return "\n".join(lines)


@dataclass
class MetricResult:
    """Result of a single metric computation between two models."""
    metric_name: str
    variable: str
    value: float
    units: str
    description: str
    reference_model: str
    candidate_model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Full benchmark result for a model pair."""
    reference: str
    candidate: str
    metrics: list[MetricResult] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.utcnow)
    passed: bool | None = None
    notes: str = ""

    def score_summary(self) -> dict[str, float]:
        return {m.metric_name: m.value for m in self.metrics}
