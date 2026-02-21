"""
climval.core.loader
==============================
Unified model loader supporting NetCDF4, CSV, and synthetic data sources.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from climval.models.schema import (
    ClimateModel,
    ClimateVariable,
    ModelType,
    SpatialDomain,
    SpatialResolution,
    TemporalDomain,
    TemporalResolution,
    STANDARD_VARIABLES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known dataset presets
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, Dict[str, Any]] = {
    "era5": {
        "model_type": ModelType.REANALYSIS,
        "spatial_resolution": SpatialResolution.HIGH,
        "temporal_resolution": TemporalResolution.HOURLY,
        "description": "ECMWF ERA5 Global Reanalysis (1940–present)",
    },
    "era5-land": {
        "model_type": ModelType.REANALYSIS,
        "spatial_resolution": SpatialResolution.HIGH,
        "temporal_resolution": TemporalResolution.HOURLY,
        "description": "ECMWF ERA5-Land High-Resolution Land Reanalysis",
    },
    "cmip6-mpi": {
        "model_type": ModelType.GCM,
        "spatial_resolution": SpatialResolution.MEDIUM,
        "temporal_resolution": TemporalResolution.MONTHLY,
        "description": "CMIP6 MPI-ESM1-2-HR (Max Planck Institute)",
    },
    "cmip6-hadgem": {
        "model_type": ModelType.GCM,
        "spatial_resolution": SpatialResolution.MEDIUM,
        "temporal_resolution": TemporalResolution.MONTHLY,
        "description": "CMIP6 HadGEM3-GC31-LL (Met Office)",
    },
    "cordex-eur": {
        "model_type": ModelType.RCM,
        "spatial_resolution": SpatialResolution.HIGH,
        "temporal_resolution": TemporalResolution.DAILY,
        "description": "EURO-CORDEX Regional Climate Model (0.11°)",
    },
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    preset: Optional[str] = None,
    *,
    name: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    model_type: ModelType = ModelType.CUSTOM,
    variables: Optional[List[str]] = None,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
    time_start: Optional[datetime] = None,
    time_end: Optional[datetime] = None,
    spatial_resolution: SpatialResolution = SpatialResolution.MEDIUM,
    temporal_resolution: TemporalResolution = TemporalResolution.MONTHLY,
    metadata: Optional[Dict[str, Any]] = None,
) -> ClimateModel:
    """
    Load a climate model for benchmarking.

    Parameters
    ----------
    preset : str, optional
        Named preset (e.g. "era5", "cmip6-mpi"). Overrides other args if set.
    name : str, optional
        Display name. Defaults to preset name if using a preset.
    path : str or Path, optional
        Path to NetCDF4 or CSV file. If not provided, a synthetic placeholder
        is created (useful for testing and CI).
    variables : list of str, optional
        CF-convention variable names (e.g. ["tas", "pr"]). Defaults to all
        standard variables.
    lat_range : tuple of float
        (lat_min, lat_max) in decimal degrees.
    lon_range : tuple of float
        (lon_min, lon_max) in decimal degrees.
    time_start : datetime, optional
        Start of temporal domain.
    time_end : datetime, optional
        End of temporal domain.
    metadata : dict, optional
        Arbitrary key-value metadata.

    Returns
    -------
    ClimateModel

    Examples
    --------
    >>> era5 = load_model("era5", path="data/era5_tas_2000_2020.nc")
    >>> custom = load_model(
    ...     name="MyModel",
    ...     path="data/mymodel.nc",
    ...     variables=["tas", "pr"],
    ...     lat_range=(35.0, 72.0),
    ...     lon_range=(-10.0, 40.0),
    ... )
    """
    resolved_name = name or preset or "unnamed"
    resolved_meta = dict(metadata or {})

    # Apply preset defaults
    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(_PRESETS.keys())}"
            )
        p = _PRESETS[preset]
        model_type = p["model_type"]
        spatial_resolution = p["spatial_resolution"]
        temporal_resolution = p["temporal_resolution"]
        resolved_meta.setdefault("description", p.get("description", ""))

    # Resolve variables
    var_keys = variables or list(STANDARD_VARIABLES.keys())
    resolved_vars: List[ClimateVariable] = []
    for vk in var_keys:
        if vk in STANDARD_VARIABLES:
            resolved_vars.append(STANDARD_VARIABLES[vk])
        else:
            logger.warning("Unknown variable '%s', creating generic entry.", vk)
            resolved_vars.append(ClimateVariable(name=vk, long_name=vk, units="?"))

    # Spatial / temporal domains
    spatial_domain = SpatialDomain(
        lat_min=lat_range[0],
        lat_max=lat_range[1],
        lon_min=lon_range[0],
        lon_max=lon_range[1],
    )
    time_start = time_start or datetime(1980, 1, 1)
    time_end = time_end or datetime(2020, 12, 31)
    temporal_domain = TemporalDomain(start=time_start, end=time_end)

    # Optionally load data
    if path is not None:
        _check_file(Path(path))
        resolved_meta["source_path"] = str(path)

    model = ClimateModel(
        name=resolved_name,
        model_type=model_type,
        variables=resolved_vars,
        spatial_domain=spatial_domain,
        temporal_domain=temporal_domain,
        spatial_resolution=spatial_resolution,
        temporal_resolution=temporal_resolution,
        source_path=str(path) if path else None,
        metadata=resolved_meta,
    )

    logger.info("Loaded model: %s", resolved_name)
    return model


def _check_file(path: Path) -> None:
    """Validate that a data file exists and has a supported format."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".nc", ".nc4", ".csv", ".zarr"}:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Supported: .nc, .nc4, .csv, .zarr"
        )
    logger.debug("Validated data file: %s (%d bytes)", path, path.stat().st_size)
