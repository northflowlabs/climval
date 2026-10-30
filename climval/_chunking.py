"""Chunk sizes for the dask arrays climval builds.

dask adds tasks per chunk for every operation, so arrays that keep one small
chunk per day or per spatial tile make large graphs. climval rechunks its
intermediate arrays to whole horizontal fields and dask's automatic size along
time at the points where the chunk structure changes (after time aggregation,
before interpolation, after alignment).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

HORIZONTAL_DIMS = ("lat", "lon", "station")


def chunk_limit_bytes() -> int:
    """dask's ``array.chunk-size`` in bytes (128 MiB when dask is not configured)."""
    try:
        import dask
        from dask.utils import parse_bytes

        return int(parse_bytes(dask.config.get("array.chunk-size")))
    except Exception:  # noqa: BLE001
        return 128 * 2**20


def field_chunks(da: xr.DataArray, horizontal: Sequence[str] = HORIZONTAL_DIMS) -> xr.DataArray:
    """``da`` rechunked to whole horizontal fields and automatic chunks along time.

    The horizontal dimensions become one chunk when a float64 field fits in a
    quarter of ``array.chunk-size`` (a 0.25 degree global field is 8.3 MB);
    larger fields keep their chunks. NumPy arrays are returned unchanged.
    """
    if da.chunks is None:
        return da
    spatial = [str(d) for d in da.dims if d in horizontal]
    field_bytes = int(np.prod([da.sizes[d] for d in spatial])) * 8
    spec: dict[str, Any] = {}
    if spatial and field_bytes * 4 <= chunk_limit_bytes():
        spec.update({d: -1 for d in spatial})
    if "time" in da.dims:
        spec["time"] = "auto"
    if not spec:
        return da
    out: xr.DataArray = da.chunk(spec)
    return out
