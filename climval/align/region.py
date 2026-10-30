"""Regions: latitude and longitude boxes, and regionmask regions.

A :class:`Box` is applied before regridding. Its longitude range may cross
the antimeridian (``Box(lat=(-20, 20), lon=(160, -150))`` covers 160E to
150W). A :class:`MaskRegion` wraps a ``regionmask`` region (for example an
IPCC AR6 land region); its bounding box is applied before regridding and the
polygon mask after regridding, on the target grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from climval.errors import InputError, MissingDependencyError, NoOverlapError


@dataclass(frozen=True)
class Box:
    """A latitude and longitude box in degrees.

    Parameters
    ----------
    lat : (south, north)
    lon : (west, east), optional
        Read eastward from ``west`` to ``east``. ``west > east`` crosses the
        antimeridian. ``None`` keeps all longitudes.
    name : str, optional
    """

    lat: tuple[float, float] = (-90.0, 90.0)
    lon: tuple[float, float] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.lat[0] > self.lat[1]:
            raise InputError(f"Box latitude must be (south, north), got {self.lat}")
        if self.lon is not None and self.lon[0] == self.lon[1]:
            raise InputError(f"Box longitude range is empty: {self.lon}")

    @property
    def lon_width(self) -> float:
        """Eastward width in degrees, 360 for all longitudes."""
        if self.lon is None:
            return 360.0
        w = float(self.lon[1] - self.lon[0])
        if w <= 0:
            w += 360.0
        return min(w, 360.0)

    @property
    def subsets_lon(self) -> bool:
        return self.lon is not None and self.lon_width < 360.0

    @property
    def west(self) -> float:
        """West edge expressed in [-180, 180)."""
        assert self.lon is not None
        return float(np.mod(self.lon[0] + 180.0, 360.0) - 180.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "box",
            "name": self.name,
            "lat": list(self.lat),
            "lon": list(self.lon) if self.lon is not None else None,
            "crosses_antimeridian": bool(self.subsets_lon and self.west + self.lon_width > 180.0),
        }


@dataclass(frozen=True)
class MaskRegion:
    """One region of a ``regionmask.Regions`` collection, used as a polygon mask."""

    regions: Any
    key: int | str

    @property
    def number(self) -> int:
        return int(self.regions.map_keys(self.key))

    @property
    def name(self) -> str:
        return str(self.regions[self.number].name)

    def box(self) -> Box:
        lon_min, lat_min, lon_max, lat_max = (float(b) for b in self.regions[self.number].bounds)
        lon = (lon_min, lon_max) if lon_max - lon_min < 359.0 else None
        return Box(lat=(max(lat_min, -90.0), min(lat_max, 90.0)), lon=lon)

    def mask(self, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
        """Boolean mask on a rectilinear grid, True inside the region."""
        lon_m = np.mod(np.asarray(lon, dtype=float) + 180.0, 360.0) - 180.0
        lon2d, lat2d = np.meshgrid(lon_m, np.asarray(lat, dtype=float))
        numbers = self.regions.mask(
            xr.DataArray(lon2d, dims=("lat", "lon")),
            xr.DataArray(lat2d, dims=("lat", "lon")),
        )
        inside = np.asarray(numbers.values) == self.number
        return xr.DataArray(inside, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "regionmask",
            "regions": str(getattr(self.regions, "name", "")),
            "key": self.key,
            "number": self.number,
            "name": self.name,
            "bounding_box": self.box().to_dict(),
        }


# Also accepted: a climval.models.SpatialDomain (lat_min, lat_max, lon_min, lon_max).
RegionLike = Box | MaskRegion | Mapping[str, Any] | str | Any | None


def as_region(obj: RegionLike) -> Box | MaskRegion | None:
    """Accept a Box, MaskRegion, dict, legacy SpatialDomain or 'ar6.land:NEU'."""
    if obj is None or isinstance(obj, (Box, MaskRegion)):
        return obj
    if isinstance(obj, Mapping):
        lat = tuple(obj.get("lat", (-90.0, 90.0)))
        lon = obj.get("lon")
        return Box(
            lat=(float(lat[0]), float(lat[1])),
            lon=(float(lon[0]), float(lon[1])) if lon is not None else None,
            name=obj.get("name"),
        )
    dom: Any = obj
    if all(hasattr(dom, a) for a in ("lat_min", "lat_max", "lon_min", "lon_max")):
        lon_t = (float(dom.lon_min), float(dom.lon_max))
        return Box(
            lat=(float(dom.lat_min), float(dom.lat_max)),
            lon=None if lon_t[1] - lon_t[0] >= 360.0 else lon_t,
            name=getattr(dom, "description", None),
        )
    if isinstance(obj, str):
        return _regionmask_from_string(obj)
    raise InputError(
        f"Cannot use {obj!r} as a region. Pass climval.Box(lat=..., lon=...), "
        "a dict with 'lat' and 'lon', or a regionmask region such as 'ar6.land:NEU'."
    )


def _regionmask_from_string(spec: str) -> MaskRegion:
    if ":" not in spec:
        raise InputError(
            f"Region string {spec!r} must look like 'ar6.land:NEU' "
            "(regionmask defined_regions path, a colon, then the region key)"
        )
    path, key = spec.split(":", 1)
    try:
        import regionmask
    except ImportError as exc:
        raise MissingDependencyError(
            'Named regions need regionmask. Install it with: pip install "climval[regions]"'
        ) from exc
    obj: Any = regionmask.defined_regions
    for part in path.split("."):
        obj = getattr(obj, part)
    key_value: int | str = int(key) if key.isdigit() else key
    return MaskRegion(regions=obj, key=key_value)


def choose_lon_frame(box: Box | None, target_lon: np.ndarray | None) -> float:
    """Start of the longitude frame [start, start + 360) used for both inputs.

    The frame follows the target grid's convention (0..360 or -180..180)
    when the region fits in it without a break, so a box that crosses the
    seam of that convention stays one contiguous block.
    """
    prefer = -180.0
    if target_lon is not None and np.size(target_lon) and float(np.nanmax(target_lon)) > 180.0:
        prefer = 0.0
    if box is None or not box.subsets_lon:
        return prefer
    west = box.west
    east = west + box.lon_width
    fits = {
        -180.0: east <= 180.0 + 1e-9,
        0.0: west >= 0.0 and east <= 360.0 + 1e-9,
    }
    # west in [-180, 0) can also be written as west + 360 in the 0..360 frame
    if not fits[0.0] and west < 0.0 and east + 360.0 <= 360.0 + 1e-9:
        fits[0.0] = True
    if fits[prefer]:
        return prefer
    other = 0.0 if prefer == -180.0 else -180.0
    if fits[other]:
        return other
    return west


def subset_box(
    da: xr.DataArray,
    box: Box,
    frame_start: float,
    role: str,
    margin_lat: float = 0.0,
    margin_lon: float = 0.0,
    *,
    lon: bool = True,
) -> xr.DataArray:
    """Select the box from an array already expressed in the longitude frame.

    ``lon=False`` cuts latitude only (longitude is cut later, after the
    periodic padding of a global source).
    """
    eps = 1e-9
    if "lat" in da.dims:
        lat = np.asarray(da["lat"].values, dtype=float)
        keep = (lat >= box.lat[0] - margin_lat - eps) & (lat <= box.lat[1] + margin_lat + eps)
        if not keep.any():
            raise NoOverlapError(
                f"The region {box.to_dict()} contains no latitude of the {role} "
                f"(latitude {lat.min():.2f} to {lat.max():.2f})"
            )
        da = da.isel(lat=np.nonzero(keep)[0])
    if lon and box.subsets_lon and "lon" in da.dims:
        lon_values = np.asarray(da["lon"].values, dtype=float)
        west = frame_start + np.mod(box.west - frame_start, 360.0)
        east = west + box.lon_width
        keep = (lon_values >= west - margin_lon - eps) & (lon_values <= east + margin_lon + eps)
        if not keep.any():
            raise NoOverlapError(f"The region {box.to_dict()} contains no longitude of the {role}")
        da = da.isel(lon=np.nonzero(keep)[0])
    return da


def needs_wrap(lon: np.ndarray, box: Box | None, frame_start: float, margin_lon: float) -> bool:
    """True when the box, widened by the margin, reaches beyond the outermost longitudes."""
    if box is None or not box.subsets_lon:
        return True
    v = np.asarray(lon, dtype=float)
    west = frame_start + np.mod(box.west - frame_start, 360.0)
    east = west + box.lon_width
    eps = 1e-9
    return bool(west - margin_lon < v.min() - eps or east + margin_lon > v.max() + eps)
