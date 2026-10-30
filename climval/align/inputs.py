"""Resolve user inputs (DataArray, Dataset, path or URL) to a DataArray.

NetCDF and Zarr paths and URLs are opened with dask chunks (``chunks={}``
keeps the chunking stored in the file), so their values stay on disk until
a metric is computed. CSV tables are read into memory with pandas.

Alongside the variable, a resolved input carries what the Dataset knows
about it: time, latitude and longitude bounds (CF ``bounds`` attributes or
``*_bnds`` variables), the ``frequency`` attribute and the cell area named
by ``cell_measures``.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re
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

_AREA_NAMES = ("areacella", "areacello", "cell_area")
_ZARR_SUFFIX = re.compile(r"\.zarr([-_.][A-Za-z0-9]+)*$", re.IGNORECASE)
_ZARR_MARKERS = (".zmetadata", ".zgroup", "zarr.json")
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
    bounds: dict[str, xr.DataArray] = field(default_factory=dict)
    """Bounds per axis ("time", "lat", "lon"), dims (axis, 2), when the Dataset has them."""

    @property
    def frequency(self) -> str | None:
        value = self.data.attrs.get("frequency", self.dataset_attrs.get("frequency"))
        return str(value) if value else None


def has_dask() -> bool:
    return importlib.util.find_spec("dask") is not None


def _is_url(path: str) -> bool:
    return "://" in path and not path.startswith("file://")


def _is_zarr(path: str, storage_options: Mapping[str, Any] | None = None) -> bool:
    """True for a Zarr store: a .zarr suffix (also .zarr-v3 and the like) or Zarr metadata.

    Remote URLs without such a suffix (Pangeo CMIP6 stores end in a version
    directory) are probed for .zmetadata, .zgroup or zarr.json through fsspec.
    """
    p = path.rstrip("/\\")
    if _ZARR_SUFFIX.search(p):
        return True
    if not _is_url(p):
        local = p[len("file://") :] if p.startswith("file://") else p
        if os.path.isdir(local):
            return any(os.path.exists(os.path.join(local, m)) for m in _ZARR_MARKERS)
        return False
    if _looks_like_file(p) or importlib.util.find_spec("fsspec") is None:
        return False
    try:
        import fsspec

        fs, root = fsspec.core.url_to_fs(p, **dict(storage_options or {}))
        return any(fs.exists(f"{root.rstrip('/')}/{m}") for m in _ZARR_MARKERS)
    except Exception:  # noqa: BLE001  unreachable store, missing credentials, ...
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
    """Open a NetCDF file, a Zarr store, a CSV table or several files.

    NetCDF and Zarr open with dask chunks (``chunks={}`` unless given), so
    values are read when computed; CSV tables are read into memory.

    Remote locations (``gs://``, ``s3://``, ``https://`` and other fsspec
    protocols) are supported for Zarr and NetCDF when the ``cloud`` extra is
    installed. ``open_kwargs`` are passed to the xarray opener (for example
    ``storage_options`` or ``engine``).
    """
    kwargs: dict[str, Any] = dict(open_kwargs or {})
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

    if spath.lower().endswith(".csv"):
        return _open_csv(spath)

    if kwargs.get("engine") == "zarr" or _is_zarr(spath, kwargs.get("storage_options")):
        _require("zarr", "cloud", "Reading Zarr")
        if _is_url(spath):
            _require("fsspec", "cloud", "Reading remote Zarr")
        kwargs.setdefault("engine", "zarr")
        return xr.open_dataset(spath, **kwargs)

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
    try:
        return xr.open_dataset(spath, **kwargs)
    except (OSError, ValueError) as exc:
        if not _is_url(spath) and not os.path.isdir(spath):
            raise
        raise InputError(
            f"Could not open {spath!r}: {exc}. If it is a Zarr store, pass "
            'open_kwargs={"engine": "zarr"} (and storage_options for credentials).'
        ) from exc


def _looks_like_file(url: str) -> bool:
    return url.split("?")[0].lower().endswith((".nc", ".nc4", ".h5", ".hdf5"))


_TIME_COLS = {"time", "date", "datetime"}
_LAT_COLS = {"lat", "latitude"}
_LON_COLS = {"lon", "longitude"}
_STATION_COLS = ("station", "station_id", "site", "site_id", "id")


def _open_csv(path: str) -> xr.Dataset:
    """Read a long-format CSV table (one row per point) into memory as a Dataset.

    * A column named time, date or datetime becomes the time dimension.
    * With latitude and longitude columns, the rows are gridded data when
      every combination of the latitudes and longitudes occurs, and station
      data otherwise. Station points become a ``station`` dimension with
      ``lat`` and ``lon`` coordinates on it (a ``station``, ``site`` or ``id``
      column names the stations). Station data are compared with gridded
      data by sampling the grid at the station locations.
    * Every other numeric column becomes a variable.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    lower = {str(c).lower(): c for c in frame.columns}
    time_col = next((lower[c] for c in lower if c in _TIME_COLS), None)
    lat_col = next((lower[c] for c in lower if c in _LAT_COLS), None)
    lon_col = next((lower[c] for c in lower if c in _LON_COLS), None)
    if time_col is None and lat_col is None and lon_col is None:
        raise InputError(
            f"{path}: the CSV needs at least one coordinate column "
            "(time, lat or lon) to be compared"
        )
    if time_col is not None:
        frame[time_col] = pd.to_datetime(frame[time_col])
    if lat_col is not None and lon_col is not None:
        n_pairs = len(frame[[lat_col, lon_col]].drop_duplicates())
        if n_pairs != frame[lat_col].nunique() * frame[lon_col].nunique():
            return _csv_stations(frame, time_col, lat_col, lon_col)
    index = [c for c in (time_col, lat_col, lon_col) if c is not None]
    out: xr.Dataset = frame.set_index(index).to_xarray()
    return out


def _csv_stations(frame: Any, time_col: Any, lat_col: Any, lon_col: Any) -> xr.Dataset:
    import pandas as pd

    lower = {str(c).lower(): c for c in frame.columns}
    id_col = next((lower[c] for c in _STATION_COLS if c in lower), None)
    if id_col is None:
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame[[lat_col, lon_col]]))
        frame = frame.assign(_climval_station=codes)
        id_col = "_climval_station"
    if int(frame.groupby(id_col)[[lat_col, lon_col]].nunique().to_numpy().max()) > 1:
        raise InputError("A station has more than one latitude or longitude in the CSV")
    sites = frame.drop_duplicates(id_col).set_index(id_col)[[lat_col, lon_col]]
    value_cols = [
        c
        for c in frame.columns
        if c not in (time_col, lat_col, lon_col, id_col) and pd.api.types.is_numeric_dtype(frame[c])
    ]
    index = [id_col] if time_col is None else [time_col, id_col]
    ds = frame.set_index(index)[value_cols].to_xarray()
    renames = {id_col: "station"}
    if time_col is not None and time_col != "time":
        renames[time_col] = "time"
    ds = ds.rename(renames)
    ids = ds["station"].values
    ds = ds.assign_coords(
        lat=("station", sites.loc[ids, lat_col].to_numpy(dtype=float)),
        lon=("station", sites.loc[ids, lon_col].to_numpy(dtype=float)),
    )
    ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
    out: xr.Dataset = ds
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
        if var.attrs.get("standard_name") == "cell_area":
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
    """Cell area named by the variable's ``cell_measures`` attribute, else a CMIP-style field.

    The fallback looks only for ``areacella``, ``areacello`` or ``cell_area``
    variables, or a variable with ``standard_name = cell_area``. A data
    variable that is merely called ``area`` is not taken as cell area; pass
    ``area=`` to use it.
    """
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
    for vname, var in ds.data_vars.items():
        if vname != da.name and var.attrs.get("standard_name") == "cell_area":
            return ds[vname]
    return None


def find_bounds(ds: xr.Dataset, da: xr.DataArray) -> dict[str, xr.DataArray]:
    """Bounds of the time, latitude and longitude coordinates of ``da`` found in ``ds``.

    Returned per axis with dims (the coordinate's dimension, 2) and that
    dimension's coordinate values, only for one-dimensional coordinates with
    two bounds per point.
    """
    from climval.align.coords import find_coord

    out: dict[str, xr.DataArray] = {}
    for axis in ("time", "lat", "lon"):
        name = find_coord(da, axis)
        if name is None or name not in ds.variables or ds[name].ndim != 1:
            continue
        dim = ds[name].dims[0]
        candidates = [str(ds[name].attrs.get("bounds", ""))] + [
            f"{name}{suffix}" for suffix in _BOUNDS_SUFFIXES
        ]
        for bname in candidates:
            if not bname or bname not in ds.variables:
                continue
            b = ds[bname]
            other = [d for d in b.dims if d != dim]
            if b.ndim == 2 and dim in b.dims and len(other) == 1 and b.sizes[other[0]] == 2:
                b = b.transpose(dim, other[0]).reset_coords(drop=True)
                out[axis] = b.assign_coords({dim: ds[name].values})
                break
    return out


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
            bounds=find_bounds(obj, da),
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
            bounds=find_bounds(ds, da),
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
            bounds=find_bounds(ds, da),
        )
    raise InputError(
        f"Cannot use an object of type {type(obj).__name__} as the {role}. "
        "Pass an xarray DataArray or Dataset, or a path or URL."
    )
