"""Resolve user inputs (DataArray, Dataset, path or URL) to a lazy DataArray.

Nothing here loads data values. Paths and URLs are opened with dask chunks when
dask is installed (``chunks={}`` keeps the chunking stored in the file), so the
arrays stay lazy until a metric is computed.
"""

from __future__ import annotations

import glob
import importlib.util
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

import xarray as xr

from climval.errors import InputError, MissingDependencyError

InputLike = Union[
    xr.DataArray,
    xr.Dataset,
    str,
    "os.PathLike[str]",
    Sequence[Union[str, "os.PathLike[str]"]],
]

_AREA_NAMES = ("areacella", "areacello", "cell_area", "area")
_BOUNDS_SUFFIXES = ("_bnds", "_bounds", "_bnd")


@dataclass
class ResolvedInput:
    """A resolved input: the variable as a DataArray plus where it came from."""

    data: xr.DataArray
    role: str
    kind: str  # "path", "url", "paths", "dataset", "dataarray"
    location: str | list[str] | None = None
    dataset_attrs: dict[str, Any] = field(default_factory=dict)
    area: xr.DataArray | None = None


def has_dask() -> bool:
    return importlib.util.find_spec("dask") is not None


def _is_url(path: str) -> bool:
    return "://" in path and not path.startswith("file://")


def _is_zarr(path: str) -> bool:
    p = path.rstrip("/\\")
    if p.endswith(".zarr"):
        return True
    if os.path.isdir(p):
        return any(
            os.path.exists(os.path.join(p, marker))
            for marker in (".zmetadata", ".zgroup", "zarr.json")
        )
    return False


def _require(module: str, extra: str, purpose: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise MissingDependencyError(
            f"{purpose} needs the '{module}' package. "
            f'Install it with: pip install "climval[{extra}]"'
        )


def open_path(
    path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    open_kwargs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Open a NetCDF file, a Zarr store, a CSV table or several files lazily.

    Remote locations (``gs://``, ``s3://``, ``https://`` and other fsspec
    protocols) are supported for Zarr and NetCDF when the ``cloud`` extra is
    installed. ``open_kwargs`` are passed to the xarray opener (for example
    ``storage_options`` or ``engine``).
    """
    kwargs: dict[str, Any] = dict(open_kwargs or {})
    if has_dask():
        kwargs.setdefault("chunks", {})

    if not isinstance(path, (str, os.PathLike)):
        paths = [os.fspath(p) for p in path]
        _require("dask", "cloud", "Opening several files")
        return xr.open_mfdataset(paths, combine="by_coords", **kwargs)

    spath = os.fspath(path)
    if any(ch in spath for ch in "*?[") and not _is_url(spath):
        matches = sorted(glob.glob(spath))
        if not matches:
            raise InputError(f"No files match the pattern {spath!r}")
        _require("dask", "cloud", "Opening several files")
        return xr.open_mfdataset(matches, combine="by_coords", **kwargs)

    if _is_zarr(spath):
        _require("zarr", "cloud", "Reading Zarr")
        if _is_url(spath):
            _require("fsspec", "cloud", "Reading remote Zarr")
        kwargs.setdefault("engine", "zarr")
        return xr.open_dataset(spath, **kwargs)

    if spath.lower().endswith(".csv"):
        return _open_csv(spath)

    if _is_url(spath) and _looks_like_file(spath):
        # A remote NetCDF4/HDF5 file: read it through fsspec (byte ranges).
        # Other URLs (for example OPeNDAP endpoints) go to xarray directly.
        _require("fsspec", "cloud", "Reading remote NetCDF")
        import fsspec

        fobj = fsspec.open(spath, mode="rb", **kwargs.pop("storage_options", {})).open()
        kwargs.setdefault("engine", "h5netcdf")
        return xr.open_dataset(fobj, **kwargs)

    if not _is_url(spath) and not os.path.exists(spath):
        raise InputError(f"Data file not found: {spath}")
    return xr.open_dataset(spath, **kwargs)


def _looks_like_file(url: str) -> bool:
    return url.split("?")[0].lower().endswith((".nc", ".nc4", ".h5", ".hdf5"))


def _open_csv(path: str) -> xr.Dataset:
    """Read a long-format CSV table (one row per point) into a Dataset.

    Columns named like coordinates (time, lat or latitude, lon or longitude)
    become dimensions; every other numeric column becomes a variable.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    coord_cols = [
        c
        for c in frame.columns
        if str(c).lower() in {"time", "date", "datetime", "lat", "latitude", "lon", "longitude"}
    ]
    if not coord_cols:
        raise InputError(
            f"{path}: the CSV needs at least one coordinate column "
            "(time, lat or lon) to be compared"
        )
    for c in coord_cols:
        if str(c).lower() in {"time", "date", "datetime"}:
            frame[c] = pd.to_datetime(frame[c])
    out: xr.Dataset = frame.set_index(coord_cols).to_xarray()
    return out


def _data_variables(ds: xr.Dataset) -> list[str]:
    bounds = {str(v.attrs["bounds"]) for v in ds.variables.values() if "bounds" in v.attrs}
    names = []
    for name, var in ds.data_vars.items():
        sname = str(name)
        if sname in bounds or sname.endswith(_BOUNDS_SUFFIXES):
            continue
        if sname in _AREA_NAMES or sname.startswith("vertices"):
            continue
        if var.ndim == 0:
            continue
        names.append(sname)
    return names


def select_variable(ds: xr.Dataset, variable: str | None, role: str) -> xr.DataArray:
    if variable is not None:
        if variable not in ds.data_vars:
            raise InputError(
                f"Variable {variable!r} not found in the {role}. "
                f"Available: {sorted(str(v) for v in ds.data_vars)}"
            )
        return ds[variable]
    names = _data_variables(ds)
    if len(names) == 1:
        return ds[names[0]]
    raise InputError(
        f"The {role} holds {len(names)} data variables {names}; pass variable= to choose one"
    )


def find_area(ds: xr.Dataset, da: xr.DataArray) -> xr.DataArray | None:
    """Return the cell area referenced by ``cell_measures`` or a known name."""
    measures = str(da.attrs.get("cell_measures", ""))
    tokens = measures.replace(":", " : ").split()
    for i, tok in enumerate(tokens):
        if tok == "area" and i + 2 < len(tokens) and tokens[i + 1] == ":":
            name = tokens[i + 2]
            if name in ds.variables:
                return ds[name]
    for name in _AREA_NAMES:
        if name in ds.variables and name != da.name:
            return ds[name]
    return None


def resolve_input(
    obj: Any,
    variable: str | None,
    role: str,
    open_kwargs: Mapping[str, Any] | None = None,
) -> ResolvedInput:
    """Turn a DataArray, Dataset, path, URL or list of paths into a DataArray."""
    if obj is None:
        raise InputError(
            f"No {role} data. climval.compare() needs both a reference and a "
            "candidate: an xarray DataArray, an xarray Dataset (with variable=), "
            "or a path or URL to NetCDF, Zarr or CSV. Since 0.2.0 climval does "
            "not substitute random numbers; for a demonstration on synthetic "
            "data call climval.demo()."
        )
    if isinstance(obj, xr.DataArray):
        da = obj if obj.name is not None else obj.rename(variable or role)
        return ResolvedInput(data=da, role=role, kind="dataarray", dataset_attrs={})
    if isinstance(obj, xr.Dataset):
        da = select_variable(obj, variable, role)
        return ResolvedInput(
            data=da,
            role=role,
            kind="dataset",
            dataset_attrs=dict(obj.attrs),
            area=find_area(obj, da),
        )
    if isinstance(obj, (str, os.PathLike)):
        location = os.fspath(obj)
        ds = open_path(location, open_kwargs)
        da = select_variable(ds, variable, role)
        return ResolvedInput(
            data=da,
            role=role,
            kind="url" if _is_url(location) else "path",
            location=location,
            dataset_attrs=dict(ds.attrs),
            area=find_area(ds, da),
        )
    if isinstance(obj, Sequence) and all(isinstance(p, (str, os.PathLike)) for p in obj):
        locations = [os.fspath(p) for p in obj]
        ds = open_path(locations, open_kwargs)
        da = select_variable(ds, variable, role)
        return ResolvedInput(
            data=da,
            role=role,
            kind="paths",
            location=locations,
            dataset_attrs=dict(ds.attrs),
            area=find_area(ds, da),
        )
    raise InputError(
        f"Cannot use an object of type {type(obj).__name__} as the {role}. "
        "Pass an xarray DataArray or Dataset, or a path or URL."
    )
