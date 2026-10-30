"""Provenance: what was compared, how, and with which software.

Every :class:`~climval.Comparison` carries a JSON-serialisable record with

* the climval, Python and library versions;
* for each input its identity: path or URL when it came from storage, and
  always the variable name, dimensions, shape, coordinate ranges, a SHA-256
  of the index coordinates, the variable attributes and the CF/CMIP global
  attributes (source_id, experiment_id, member_id, tracking_id and others);
* a SHA-256 of a strided sample of the raw values for in-memory NumPy inputs,
  and for every input a SHA-256 of a strided sample of the aligned values
  (computed in the same pass as the metrics, so lazy inputs are read once);
* every alignment step with its parameters;
* a UTC timestamp.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xarray as xr

GLOBAL_ATTRS = (
    "source_id",
    "experiment_id",
    "member_id",
    "variant_label",
    "tracking_id",
    "institution_id",
    "activity_id",
    "table_id",
    "grid_label",
    "frequency",
    "mip_era",
    "further_info_url",
    "Conventions",
    "title",
    "institution",
    "source",
    "product",
    "dataset",
    "doi",
)
VARIABLE_ATTRS = ("units", "standard_name", "long_name", "cell_methods", "cell_measures")
LIBRARIES = (
    "numpy",
    "xarray",
    "pandas",
    "scipy",
    "dask",
    "cftime",
    "zarr",
    "fsspec",
    "netCDF4",
    "h5netcdf",
    "cf_xarray",
    "pint",
    "pint-xarray",
    "xesmf",
    "regionmask",
)
SAMPLE_PER_DIM = 16


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for lib in LIBRARIES:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def environment() -> dict[str, Any]:
    from climval import __version__

    return {
        "climval": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
    }


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(obj: Any) -> Any:
    """Convert numpy scalars, arrays, tuples and dates into JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _coord_bytes(values: np.ndarray) -> bytes:
    v = np.asarray(values)
    if v.dtype.kind in "iufb":
        return np.ascontiguousarray(v.astype(np.float64)).tobytes()
    return "|".join(str(x) for x in v.ravel()).encode("utf-8")


def coords_hash(da: xr.DataArray) -> str:
    """SHA-256 over the names and values of the dimension coordinates."""
    h = hashlib.sha256()
    for dim in da.dims:
        h.update(str(dim).encode("utf-8"))
        if dim in da.coords:
            h.update(_coord_bytes(da[dim].values))
        else:
            h.update(str(da.sizes[dim]).encode("utf-8"))
    return h.hexdigest()


def strided_sample(da: xr.DataArray) -> xr.DataArray:
    """Up to 16 evenly spaced indices per dimension (lazy for dask arrays)."""
    index = {
        str(d): np.unique(np.linspace(0, n - 1, min(n, SAMPLE_PER_DIM)).astype(int))
        for d, n in da.sizes.items()
        if n > 0
    }
    out: xr.DataArray = da.isel(index)
    return out


HASH_MANTISSA_BITS = 36


def hash_values(values: np.ndarray) -> str:
    """SHA-256 of values rounded to 36 mantissa bits (about 11 significant digits).

    The rounding keeps the hash stable when the same numbers are summed in a
    different order (NumPy against dask chunks), which changes the last bits.
    """
    v = np.asarray(values, dtype=np.float64)
    mantissa, exponent = np.frexp(v)
    scale = float(2**HASH_MANTISSA_BITS)
    v = np.ldexp(np.round(mantissa * scale) / scale, exponent)
    v = np.ascontiguousarray(np.where(np.isnan(v), np.nan, v))  # one NaN bit pattern
    return hashlib.sha256(v.tobytes()).hexdigest()


def _coord_ranges(da: xr.DataArray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dim in da.dims:
        if dim not in da.coords or da.sizes[dim] == 0:
            continue
        vals = da[dim].values
        if vals.dtype.kind in "iuf":
            out[str(dim)] = [float(np.nanmin(vals)), float(np.nanmax(vals))]
        else:
            out[str(dim)] = [str(vals[0]), str(vals[-1])]
    return out


def in_memory_numpy(da: xr.DataArray) -> bool:
    """True when the values are a NumPy array already in memory."""
    return bool(getattr(da.variable, "_in_memory", False)) and da.chunks is None


def describe_input(
    da: xr.DataArray,
    kind: str,
    location: str | list[str] | None,
    dataset_attrs: dict[str, Any],
) -> dict[str, Any]:
    attrs = {**{k: v for k, v in da.attrs.items() if k in GLOBAL_ATTRS}, **dataset_attrs}
    record: dict[str, Any] = {
        "kind": kind,
        "location": location,
        "variable": str(da.name) if da.name is not None else None,
        "dims": [str(d) for d in da.dims],
        "shape": [int(s) for s in da.shape],
        "dtype": str(da.dtype),
        "lazy": da.chunks is not None or not in_memory_numpy(da),
        "chunks": (
            {str(d): [int(c) for c in cs] for d, cs in zip(da.dims, da.chunks, strict=False)}
            if da.chunks is not None
            else None
        ),
        "coordinate_ranges": _coord_ranges(da),
        "coordinates_sha256": coords_hash(da),
        "variable_attributes": {k: da.attrs[k] for k in VARIABLE_ATTRS if k in da.attrs},
        "global_attributes": {k: attrs[k] for k in GLOBAL_ATTRS if k in attrs},
    }
    if in_memory_numpy(da):
        record["values_sample_sha256"] = hash_values(strided_sample(da).values)
        record["values_sample_basis"] = (
            f"strided sample, up to {SAMPLE_PER_DIM} indices per dimension, "
            f"rounded to {HASH_MANTISSA_BITS} mantissa bits"
        )
    else:
        record["values_sample_sha256"] = None
        record["values_sample_basis"] = (
            "not hashed before alignment to avoid reading lazy data; see aligned_values_sha256"
        )
    return jsonable(record)  # type: ignore[no-any-return]
