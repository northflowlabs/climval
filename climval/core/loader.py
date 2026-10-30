"""
climval.core.loader
==============================
Model descriptions for the 0.1 BenchmarkSuite API.

``load_model`` describes a dataset (name, type, presets) and points at its
data: a local NetCDF, Zarr or CSV path, a remote URL, or an in-memory xarray
object passed as ``data=``. Nothing is read here; the data are opened lazily
when the suite runs. New code can call :func:`climval.compare` directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from climval.models.schema import (
    STANDARD_VARIABLES,
    ClimateModel,
    ClimateVariable,
    ModelType,
    SpatialDomain,
    SpatialResolution,
    TemporalDomain,
    TemporalResolution,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known dataset presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "era5": {
        "model_type": ModelType.REANALYSIS,
        "spatial_resolution": SpatialResolution.HIGH,
        "temporal_resolution": TemporalResolution.HOURLY,
        "description": "ECMWF ERA5 Global Reanalysis (1940 to present)",
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
    preset: str | None = None,
    *,
    name: str | None = None,
    path: str | Path | None = None,
    model_type: ModelType = ModelType.CUSTOM,
    variables: list[str] | None = None,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    spatial_resolution: SpatialResolution = SpatialResolution.MEDIUM,
    temporal_resolution: TemporalResolution = TemporalResolution.MONTHLY,
    metadata: dict[str, Any] | None = None,
    data: Any = None,
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
        NetCDF, Zarr or CSV path, or a remote URL (gs://, s3://, https://).
        Without ``path`` or ``data`` the model is a description only;
        ``BenchmarkSuite.run`` then refuses to score it unless asked for
        synthetic data explicitly.
    variables : list of str, optional
        CF-convention variable names (e.g. ["tas", "pr"]). Defaults to all
        standard variables.
    lat_range : tuple of float, optional
        (lat_min, lat_max) in decimal degrees. When given (or ``lon_range``),
        ``BenchmarkSuite.run`` compares only this region, unless ``region=``
        is passed to ``run``.
    lon_range : tuple of float, optional
        (lon_min, lon_max) in decimal degrees, read eastward.
    time_start : datetime, optional
        Start of the period ``BenchmarkSuite.run`` compares, unless
        ``period=`` is passed to ``run``.
    time_end : datetime, optional
        End of that period.
    metadata : dict, optional
        Arbitrary key-value metadata.
    data : xarray.Dataset or xarray.DataArray, optional
        Data already open in Python (for example from intake-esm).

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
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(_PRESETS.keys())}")
        p = _PRESETS[preset]
        model_type = p["model_type"]
        spatial_resolution = p["spatial_resolution"]
        temporal_resolution = p["temporal_resolution"]
        resolved_meta.setdefault("description", p.get("description", ""))

    # Resolve variables
    var_keys = variables or list(STANDARD_VARIABLES.keys())
    resolved_vars: list[ClimateVariable] = []
    for vk in var_keys:
        if vk in STANDARD_VARIABLES:
            resolved_vars.append(STANDARD_VARIABLES[vk])
        else:
            logger.warning("Unknown variable '%s', creating generic entry.", vk)
            resolved_vars.append(ClimateVariable(name=vk, long_name=vk, units="?"))

    # Spatial / temporal domains. Unset parts are global and 1980 to 2020 in
    # the description, and are not applied when the suite runs.
    region_given = lat_range is not None or lon_range is not None
    period_given = time_start is not None or time_end is not None
    lat_t = lat_range if lat_range is not None else (-90.0, 90.0)
    lon_t = lon_range if lon_range is not None else (-180.0, 180.0)
    spatial_domain = SpatialDomain(
        lat_min=lat_t[0],
        lat_max=lat_t[1],
        lon_min=lon_t[0],
        lon_max=lon_t[1],
    )
    temporal_domain = TemporalDomain(
        start=time_start or datetime(1980, 1, 1), end=time_end or datetime(2020, 12, 31)
    )

    # Optionally load data
    if path is not None:
        _check_file(path)
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
        data=data,
        region_given=region_given,
        period_given=period_given,
    )

    logger.info("Loaded model: %s", resolved_name)
    return model


def _check_file(path: Path | str) -> None:
    """Validate that a local data file exists and has a supported format.

    Remote URLs are not checked here; they are opened when the suite runs.
    """
    if "://" in str(path):
        return
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".nc", ".nc4", ".csv", ".zarr"}:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .nc, .nc4, .csv, .zarr")
    logger.debug("Validated data file: %s (%d bytes)", path, path.stat().st_size)
