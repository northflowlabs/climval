# SPDX-License-Identifier: Apache-2.0
"""Time alignment across calendars and frequencies.

Two inputs rarely share timestamps: a model in a ``noleap`` or ``360_day``
calendar stamps its monthly means on different days than a reanalysis in
the standard calendar, and one may be daily while the other is monthly.
climval therefore does not match timestamps. It

1. finds each input's native frequency, in this order of evidence:
   time bounds (``time_bnds``, CF ``bounds``), a ``frequency`` attribute
   (CMIP vocabulary such as ``mon``, ``day``, ``3hr`` or a pandas alias),
   then the distribution of time steps. When time bounds are present, each
   step is labelled by the midpoint of its interval, so values stamped at
   the end of their interval (ERA5 accumulations at 00 UTC) fall into the
   right day. Without bounds, a series whose steps are a year apart but
   stamped in the middle of the year is ambiguous (yearly means, or one
   month per year); a single time step says nothing about the period it
   covers; irregular steps have no single frequency. In these cases climval
   raises instead of guessing, when the guess would change the result;
2. selects the requested period on calendar fields (year, month, day), so
   the same period string means the same thing in every calendar;
3. takes the common frequency: the coarser of the two native frequencies,
   or the user's ``freq`` (never finer than either). Sub-daily data keep
   their step: hourly against 3-hourly is compared in 3-hour bins;
4. aggregates any finer input to that frequency with a time mean (monthly
   data aggregated to years is weighted by month length). A period's value
   is kept only where at least ``min_coverage`` (default 0.8) of the
   expected native steps of that period hold a valid value, counted per
   grid cell; periods with fewer time steps than that are dropped;
5. labels every period by its calendar fields (for monthly data the year
   and month) and keeps the labels present in both inputs.

The aligned pair carries the reference's time coordinate. Daily comparisons
across calendars match days by label, so days that exist in only one
calendar (29 February, 30 February, 31st days) drop out; monthly or coarser
comparisons are calendar independent. A daily or sub-daily comparison
between a 360_day calendar and any other calendar warns, because the same
date label falls at a different point of the seasonal cycle.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climval._chunking import chunk_limit_bytes, field_chunks
from climval.errors import AlignmentError, NoOverlapError

DEFAULT_MIN_COVERAGE = 0.8

# Fixed frequencies; sub-daily frequencies are written "<N>h".
_FIXED = {"D": 1.0, "MS": 30.44, "YS": 365.25}
_NAMES_FIXED = {"D": "daily", "MS": "monthly", "YS": "yearly"}
_LABELS_FIXED = {"D": "date", "MS": "year-month", "YS": "year"}

_WORD_ALIASES = {
    "D": "D",
    "1D": "D",
    "d": "D",
    "day": "D",
    "daily": "D",
    "dayPt": "D",
    "MS": "MS",
    "M": "MS",
    "ME": "MS",
    "mon": "MS",
    "monPt": "MS",
    "month": "MS",
    "monthly": "MS",
    "YS": "YS",
    "Y": "YS",
    "YE": "YS",
    "A": "YS",
    "AS": "YS",
    "year": "YS",
    "yearly": "YS",
    "annual": "YS",
    "yr": "YS",
    "yrPt": "YS",
}
_HOURS_RE = re.compile(r"^(\d*)\s*(h|H|hr|hrPt|hour|hours|hourly)$")


def _hours(freq: str) -> int | None:
    m = re.fullmatch(r"(\d+)h", freq)
    return int(m.group(1)) if m else None


def nominal_days(freq: str) -> float:
    hours = _hours(freq)
    return hours / 24.0 if hours is not None else _FIXED[freq]


def freq_name(freq: str) -> str:
    hours = _hours(freq)
    if hours is None:
        return _NAMES_FIXED[freq]
    return "hourly" if hours == 1 else f"{hours}-hourly"


def _label_name(freq: str) -> str:
    hours = _hours(freq)
    if hours is None:
        return _LABELS_FIXED[freq]
    return "date and hour" if hours == 1 else f"date and {hours}-hour bin from 00 UTC"


def normalise_freq(freq: str) -> str:
    """Canonical frequency: '<N>h', 'D', 'MS' or 'YS'."""
    text = str(freq).strip()
    if text in _WORD_ALIASES:
        return _WORD_ALIASES[text]
    m = _HOURS_RE.match(text)
    if m:
        n = int(m.group(1) or 1)
        if 1 <= n <= 24 and 24 % n == 0:
            return "D" if n == 24 else f"{n}h"
    raise AlignmentError(
        f"Unsupported freq {freq!r}; use 'h' or '<N>h' with N dividing 24 (for example "
        "'3h'), 'D', 'MS' (monthly) or 'YS' (yearly)"
    )


def calendar(da: xr.DataArray) -> str:
    try:
        return str(da["time"].dt.calendar)
    except (AttributeError, TypeError):
        return "unknown"


def _is_cftime(da: xr.DataArray) -> bool:
    return bool(da["time"].dtype == object)


def _deltas_days(values: np.ndarray) -> np.ndarray:
    """Differences between consecutive datetimes (numpy or cftime) in days."""
    v = np.asarray(values)
    if v.size < 2:
        return np.zeros(0)
    if v.dtype.kind == "M":
        d = np.diff(v).astype("timedelta64[s]").astype(float)
        return np.asarray(d / 86400.0)
    return np.asarray([(b - a) / timedelta(days=1) for a, b in zip(v[:-1], v[1:], strict=True)])


def _widths_days(bounds: np.ndarray) -> np.ndarray:
    b = np.asarray(bounds)
    if b.dtype.kind == "M":
        return np.asarray((b[:, 1] - b[:, 0]).astype("timedelta64[s]").astype(float) / 86400.0)
    return np.asarray([(u - lo) / timedelta(days=1) for lo, u in b])


def classify_step(step: float | None) -> str | None:
    """Frequency class of a step in days: '<N>h', 'D', 'MS', 'YS' or None."""
    if step is None or not np.isfinite(step) or step <= 0:
        return None
    if step < 0.95:
        hours = step * 24.0
        n = int(round(hours))
        if n >= 1 and abs(hours - n) <= 0.05 * n and 24 % n == 0:
            return f"{n}h"
        return None
    if step <= 1.05:
        return "D"
    if 27.0 <= step <= 32.0:
        return "MS"
    if 359.0 <= step <= 367.0:
        return "YS"
    return None


def _supported_coarser(step: float) -> list[str]:
    options = [f for f in ("D", "MS", "YS") if nominal_days(f) >= 0.95 * step]
    return options


@dataclass
class TimeInfo:
    """Native frequency of one input and where it came from."""

    freq: str | None
    step: float | None  # native step in days, when known
    source: str
    ambiguous: str | None = None  # why the frequency is a guess that may be wrong
    relabelled: bool = False

    def describe(self) -> str:
        if self.freq is not None:
            name = freq_name(self.freq)
        elif self.step is not None:
            name = f"step {self.step:.4g} days"
        else:
            name = "unknown"
        return f"{name} (from {self.source})"


def _bounds_values(bounds: xr.DataArray | None, da: xr.DataArray) -> np.ndarray | None:
    if bounds is None or "time" not in bounds.dims:
        return None
    nv = [d for d in bounds.dims if d != "time"]
    if len(nv) != 1 or bounds.sizes[nv[0]] != 2 or bounds.sizes["time"] != da.sizes["time"]:
        return None
    return np.asarray(bounds.transpose("time", nv[0]).values)


def native_time(
    da: xr.DataArray,
    role: str,
    bounds: xr.DataArray | None = None,
    declared: str | None = None,
) -> tuple[xr.DataArray, TimeInfo]:
    """Native frequency of ``da``; relabels steps to bound midpoints when bounds exist."""
    n = da.sizes["time"]
    b = _bounds_values(bounds, da)
    if b is not None:
        widths = _widths_days(b)
        med = float(np.median(widths))
        freq = classify_step(med)
        consistent = bool(np.all(widths > 0)) and all(classify_step(w) == freq for w in widths)
        if freq is None or not consistent:
            raise AlignmentError(
                f"The {role}'s time bounds span {widths.min():.4g} to {widths.max():.4g} "
                "days, which is not one hourly, daily, monthly or yearly frequency. "
                "Select a regular part of the series first."
            )
        lower = b[:, 0]
        mid = np.asarray([lo + (u - lo) / 2 for lo, u in b]) if b.dtype == object else None
        if mid is None:
            mid = lower + (b[:, 1] - b[:, 0]) / 2
        da = da.assign_coords(time=mid)
        return da, TimeInfo(freq, med, "time bounds", relabelled=True)

    steps = _deltas_days(da["time"].values)
    if declared:
        try:
            freq = normalise_freq(str(declared))
        except AlignmentError:
            freq = None
        if freq is not None:
            if steps.size and float(np.median(steps)) < 0.9 * nominal_days(freq):
                raise AlignmentError(
                    f"The {role} declares frequency {declared!r} but its time steps are "
                    f"{float(np.median(steps)):.4g} days apart. Correct the frequency "
                    "attribute or remove it."
                )
            return da, TimeInfo(freq, nominal_days(freq), f"frequency attribute {declared!r}")

    if n < 2:
        return da, TimeInfo(None, None, "a single time step")
    med = float(np.median(steps))
    if float(np.min(steps)) < 0.5 * med:
        raise AlignmentError(
            f"The {role}'s time steps are irregular ({float(np.min(steps)):.4g} to "
            f"{float(np.max(steps)):.4g} days, median {med:.4g}), so it has no single "
            "native frequency. Add time bounds (time_bnds), set "
            "attrs['frequency'] (for example 'mon' or 'day'), or select a regular part."
        )
    freq = classify_step(med)
    info = TimeInfo(freq, med, "time steps")
    if freq == "YS":
        t = da["time"].dt
        months = np.asarray(t.month.values)
        days = np.asarray(t.day.values)
        at_start = bool(np.all((months == 1) & (days == 1)))
        at_end = bool(np.all((months == 12) & (days >= 30)))
        if not (at_start or at_end):
            info.ambiguous = (
                f"The {role}'s time steps are one year apart but stamped in month "
                f"{int(months[0])}, so they may hold yearly means or one month (or day) "
                "per year"
            )
    return da, info


def _codes(da: xr.DataArray, freq: str) -> np.ndarray:
    t = da["time"].dt
    year = np.asarray(t.year.values, dtype=np.int64)
    if freq == "YS":
        return year
    month = np.asarray(t.month.values, dtype=np.int64)
    if freq == "MS":
        return year * 100 + month
    day = np.asarray(t.day.values, dtype=np.int64)
    if freq == "D":
        out: np.ndarray = year * 10000 + month * 100 + day
        return out
    hours = _hours(freq) or 1
    hour = np.asarray(t.hour.values, dtype=np.int64)
    codes: np.ndarray = year * 1000000 + month * 10000 + day * 100 + (hour // hours) * hours
    return codes


def _parse_bound(value: Any, end: bool) -> int:
    """Day-precision code for a period bound such as '2000', '2000-06' or 2000."""
    if isinstance(value, (int, np.integer)):
        parts = [int(value)]
    elif hasattr(value, "year") and hasattr(value, "month"):
        parts = [int(value.year), int(value.month), int(getattr(value, "day", 1))]
    else:
        text = str(value).strip()
        try:
            parts = [int(p) for p in text.split("T")[0].split("-") if p != ""]
        except ValueError:
            raise AlignmentError(
                f"Cannot read period bound {value!r}; use 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'"
            ) from None
        if not 1 <= len(parts) <= 3:
            raise AlignmentError(f"Cannot read period bound {value!r}")
    year = parts[0]
    month = parts[1] if len(parts) > 1 else (12 if end else 1)
    day = parts[2] if len(parts) > 2 else (31 if end else 1)
    return year * 10000 + month * 100 + day


def select_period(da: xr.DataArray, period: tuple[Any, Any], role: str) -> xr.DataArray:
    start = _parse_bound(period[0], end=False)
    end = _parse_bound(period[1], end=True)
    codes = _codes(da, "D")
    keep = (codes >= start) & (codes <= end)
    if not keep.any():
        raise NoOverlapError(
            f"The period {period[0]} to {period[1]} contains no time step of the {role} "
            f"({_describe_range(da)})"
        )
    return take_time(da, np.nonzero(keep)[0])


def take_time(da: xr.DataArray, idx: np.ndarray) -> xr.DataArray:
    """``da.isel(time=idx)``, with a slice when ``idx`` is a contiguous run.

    A slice keeps dask chunks whole; an index array adds a task per chunk.
    """
    idx = np.asarray(idx)
    n = da.sizes["time"]
    if idx.size and idx[-1] - idx[0] + 1 == idx.size and np.all(np.diff(idx) == 1):
        if idx.size == n:
            return da
        out: xr.DataArray = da.isel(time=slice(int(idx[0]), int(idx[-1]) + 1))
        return out
    out = da.isel(time=idx)
    return out


def _fixed_window(da: xr.DataArray, freq: str) -> int | None:
    """Steps per period when every period of ``freq`` holds the same steps in order.

    True for regular sub-daily series aggregated to days or to coarser
    sub-daily bins (a day has 24 hours in every calendar), when the series
    starts at the beginning of a period: then the k-th window of ``window``
    steps is the k-th period present, and ``coarsen`` gives the same sums as a
    resample. The last period may be partial. None otherwise.
    """
    if freq != "D" and _hours(freq) is None:
        return None
    codes = _codes(da, freq)
    if codes.size < 2:
        return None
    starts = np.flatnonzero(np.concatenate([[True], codes[1:] != codes[:-1]]))
    runs = np.diff(np.concatenate([starts, [codes.size]]))
    window = int(runs[0])
    if window < 2 or np.any(runs[:-1] != window) or runs[-1] > window:
        return None
    if np.unique(codes).size != runs.size:
        return None  # a period occurs twice: not sorted
    if da.chunks is not None:
        chunks = np.asarray(da.chunksizes["time"][:-1])
        if np.any(chunks % window):
            return None
    return window


def _window_block(block: np.ndarray, window: int, axis: int, what: str) -> np.ndarray:
    """NaN-skipping sum, or count of valid values, over consecutive windows along ``axis``."""
    x = np.asarray(block, dtype=np.float64)
    n = x.shape[axis]
    m = -(-n // window)
    if m * window != n:
        pad_shape = list(x.shape)
        pad_shape[axis] = m * window - n
        x = np.concatenate([x, np.full(pad_shape, np.nan)], axis=axis)
    x = x.reshape(x.shape[:axis] + (m, window) + x.shape[axis + 1 :])
    if what == "sum":
        out: np.ndarray = np.nansum(x, axis=axis + 1)
        return out
    counted: np.ndarray = np.sum(~np.isnan(x), axis=axis + 1).astype(np.float64)
    return counted


def _window_reduce(da: xr.DataArray, window: int, what: str, labels: np.ndarray) -> xr.DataArray:
    """Sum or count over windows of ``window`` time steps, one task per dask chunk.

    The time chunks hold whole windows (only the last chunk may end with a
    partial one), so each chunk reduces on its own; the data are read as
    float64. NumPy arrays go through the same function.
    """
    axis = da.get_axis_num("time")
    assert isinstance(axis, int)
    if da.chunks is None:
        data: Any = _window_block(np.asarray(da.values), window, axis, what)
    else:
        import dask.array as dsa

        chunks = list(da.data.chunks)
        chunks[axis] = tuple(-(-c // window) for c in chunks[axis])
        data = dsa.map_blocks(  # type: ignore[no-untyped-call]
            _window_block,
            da.data,
            window,
            axis,
            what,
            chunks=tuple(chunks),
            dtype=np.float64,
            meta=np.array((), dtype=np.float64),
        )
    coords = {name: c for name, c in da.coords.items() if "time" not in c.dims}
    out = xr.DataArray(data, dims=da.dims, coords=coords, name=da.name)
    labelled: xr.DataArray = out.assign_coords(time=labels)
    return labelled


def _period_chunks(da: xr.DataArray, freq: str) -> xr.DataArray:
    """Rechunk a dask array along time so that chunks hold whole periods of ``freq``.

    Chunks are filled with consecutive periods up to dask's ``array.chunk-size``
    (at least one period each), so every chunk reduces on its own and the
    aggregation does not combine partial periods across chunks. Arrays whose
    time chunks are already that large or larger are left alone.
    """
    if da.chunks is None:
        return da
    field = int(np.prod([max(c) for d, c in da.chunksizes.items() if d != "time"]))
    field_bytes = max(1, field) * 8
    per_chunk = max(1, chunk_limit_bytes() // field_bytes)
    if min(da.chunksizes["time"]) >= per_chunk:
        return da
    codes = _codes(da, freq)
    starts = np.flatnonzero(np.concatenate([[True], codes[1:] != codes[:-1]]))
    runs = np.diff(np.concatenate([starts, [codes.size]]))
    chunks: list[int] = []
    current = 0
    for run in runs.tolist():
        if current and current + run > per_chunk:
            chunks.append(current)
            current = 0
        current += run
    if current:
        chunks.append(current)
    out: xr.DataArray = da.chunk({"time": tuple(chunks)})
    return out


def _describe_range(da: xr.DataArray) -> str:
    t = da["time"].values
    if t.size == 0:
        return f"no time step left, calendar {calendar(da)}"
    return f"{t[0]} to {t[-1]}, calendar {calendar(da)}"


def _pandas_freq(freq: str) -> str:
    return freq.replace("h", "H") if _OLD_PANDAS and _hours(freq) is not None else freq


_OLD_PANDAS = tuple(int(p) for p in pd.__version__.split(".")[:2]) < (2, 2)


def period_lengths(labels: np.ndarray, freq: str, cal: str, cftime: bool) -> np.ndarray:
    """Length in days of each period that starts at ``labels`` (consecutive periods)."""
    hours = _hours(freq)
    if hours is not None:
        return np.full(len(labels), hours / 24.0)
    if freq == "D":
        return np.ones(len(labels))
    edges = xr.date_range(
        start=labels[0],
        periods=len(labels) + 1,
        freq=freq,
        calendar=cal if cftime else "standard",
        use_cftime=cftime,
    )
    deltas = edges[1:] - edges[:-1]
    return np.asarray(
        [
            d / timedelta(days=1) if isinstance(d, timedelta) else d / pd.Timedelta(days=1)
            for d in deltas
        ],
        dtype=float,
    )


def label_lengths(da: xr.DataArray, freq: str) -> np.ndarray | None:
    """Length in days of the period each time step stands for (None when all equal)."""
    if freq == "MS":
        return np.asarray(da["time"].dt.days_in_month.values, dtype=float)
    if freq == "YS":
        years = np.asarray(da["time"].dt.year.values)
        cal = calendar(da)
        out = []
        for y in years:
            out.append(_year_length(int(y), cal, _is_cftime(da)))
        return np.asarray(out, dtype=float)
    return None


def _year_length(year: int, cal: str, cftime: bool) -> float:
    if cftime:
        import cftime as cft

        a = cft.datetime(year, 1, 1, calendar=cal)
        b = cft.datetime(year + 1, 1, 1, calendar=cal)
        return float((b - a) / timedelta(days=1))
    return 366.0 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365.0


def _aggregate(
    da: xr.DataArray,
    freq: str,
    native: str | None,
    step: float | None,
    min_coverage: float,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Time mean onto ``freq``; values below ``min_coverage`` set to NaN.

    Dask arrays are first rechunked so that chunks hold whole periods. A
    regular sub-daily series whose periods all hold the same number of steps
    (hourly to daily, say) is then reduced with ``coarsen``, one task per chunk;
    other series are resampled through flox (a core dependency), which also
    reduces each chunk in place instead of building one chunk per period. The
    sums are rechunked to whole fields and dask's automatic size along time
    before any further arithmetic, so later steps do not carry one small chunk
    per day.
    """
    cftime = _is_cftime(da)
    cal = calendar(da)
    rfreq = _pandas_freq(freq)
    da = _period_chunks(da, freq)
    ones = xr.DataArray(np.ones(da.sizes["time"]), dims="time", coords={"time": da["time"].values})
    per_period = ones.resample(time=rfreq).count()
    window = _fixed_window(da, freq) if native != "MS" else None
    if window is not None:
        present = np.asarray(per_period.values) > 0
        period_labels = np.asarray(per_period["time"].values)[present]
        counts = np.asarray(per_period.values, dtype=float)[present]
        total = field_chunks(_window_reduce(da, window, "sum", period_labels))
        valid_amount = field_chunks(_window_reduce(da, window, "count", period_labels))
        out = total / valid_amount.where(valid_amount > 0)
        how = "mean"
    else:
        counts = np.asarray(per_period.values, dtype=float)
        # Sums in float64, so lazy and eager runs of float32 data agree.
        da = da.astype(np.float64) if da.dtype != np.float64 else da
        if native == "MS":
            days = da["time"].dt.days_in_month.astype(float)
            valid = da.notnull()
            num = field_chunks((da * days).resample(time=rfreq).sum(skipna=True, min_count=1))
            den = field_chunks(days.where(valid).resample(time=rfreq).sum(skipna=True, min_count=1))
            out = num / den
            valid_amount = den.fillna(0.0)
            how = "month-length weighted mean"
        else:
            # Two flox reductions (sum and count of the valid values), each one
            # task per input chunk; the results are rechunked to whole fields
            # before any further arithmetic, so that arithmetic has few chunks.
            total = field_chunks(da.resample(time=rfreq).sum(skipna=True))
            valid_amount = field_chunks(da.resample(time=rfreq).count()).astype(np.float64)
            out = total / valid_amount.where(valid_amount > 0)
            how = "mean"
    labels = np.asarray(out["time"].values)
    info: dict[str, Any] = {
        "aggregation": how,
        "periods": int(labels.size),
        "min_coverage": min_coverage,
        "coverage_rule": (
            f"a period value is kept where at least {min_coverage:g} of the expected "
            "native steps of that period hold a valid value, per grid cell"
        ),
    }
    if step is not None and labels.size:
        lengths = period_lengths(labels, freq, cal, cftime)
        if native == "MS":
            expected_steps = np.maximum(np.rint(lengths / 30.44), 1.0)
            expected_amount = lengths
        else:
            expected_steps = np.maximum(np.rint(lengths / step), 1.0)
            expected_amount = expected_steps
        complete = counts >= min_coverage * expected_steps - 1e-9
        dropped = [str(lab) for lab, ok in zip(labels, complete, strict=False) if not ok]
        info["incomplete_periods_dropped"] = dropped
        expected = xr.DataArray(expected_amount, dims="time", coords={"time": out["time"].values})
        out = out.where(valid_amount >= min_coverage * expected - 1e-9)
        if not complete.any():
            raise NoOverlapError(
                f"No {freq_name(freq)} period of the input ({_describe_range(da)}) holds at "
                f"least {min_coverage:g} of its expected time steps, so nothing is left after "
                "aggregation. Lower min_coverage or choose a finer freq="
            )
        if not complete.all():
            out = take_time(out, np.nonzero(complete)[0])
    out = out.assign_attrs(da.attrs)
    out.name = da.name
    return out, info


def align_time(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    freq: str | None = None,
    period: tuple[Any, Any] | None = None,
    mode: str = "intersect",
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    bounds: dict[str, xr.DataArray | None] | None = None,
    declared: dict[str, str | None] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
    """Put both inputs on common periods. See the module docstring."""
    has_r = "time" in reference.dims
    has_c = "time" in candidate.dims
    record: dict[str, Any] = {"step": "time"}
    if not has_r and not has_c:
        if period is not None:
            raise AlignmentError("period= was given but neither input has a time dimension")
        record["action"] = "none: no time dimension"
        return reference, candidate, record
    if has_r != has_c:
        which = "reference" if has_r else "candidate"
        raise AlignmentError(
            f"Only the {which} has a time dimension. Reduce it first "
            "(for example .mean('time') for a climatology) or select one time step."
        )
    if mode not in ("intersect", "exact"):
        raise AlignmentError(f"time must be 'intersect' or 'exact', got {mode!r}")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise AlignmentError(f"min_coverage must be in (0, 1], got {min_coverage!r}")
    bounds = bounds or {}
    declared = declared or {}

    record["calendars"] = {"reference": calendar(reference), "candidate": calendar(candidate)}
    reference, info_r = native_time(
        reference, "reference", bounds.get("reference"), declared.get("reference")
    )
    candidate, info_c = native_time(
        candidate, "candidate", bounds.get("candidate"), declared.get("candidate")
    )
    record["native_frequency"] = {
        "reference": info_r.describe(),
        "candidate": info_c.describe(),
    }
    relabelled = [r for r, i in (("reference", info_r), ("candidate", info_c)) if i.relabelled]
    if relabelled:
        record["relabelled_to_bound_midpoints"] = relabelled
    if period is not None:
        reference = select_period(reference, period, "reference")
        candidate = select_period(candidate, period, "candidate")
        record["period"] = [str(period[0]), str(period[1])]

    infos = {"reference": info_r, "candidate": info_c}
    sizes = {"reference": reference.sizes["time"], "candidate": candidate.sizes["time"]}

    if freq is not None:
        target = normalise_freq(freq)
    else:
        for role, info in infos.items():
            if info.freq is None and info.step is not None:
                options = _supported_coarser(info.step)
                hint = (
                    f"Pass freq={options[0]!r} to aggregate both to that frequency"
                    if options
                    else "Select or aggregate the data first"
                )
                raise AlignmentError(
                    f"The {role}'s time step ({info.step:.4g} days) is not sub-daily, "
                    f"daily, monthly or yearly. {hint}."
                )
        unknown = [r for r, i in infos.items() if i.freq is None]
        if unknown and len(unknown) == 2:
            t_r = str(reference["time"].values[0])
            t_c = str(candidate["time"].values[0])
            if t_r != t_c:
                raise AlignmentError(
                    "Both inputs have a single time step, with different timestamps "
                    f"({t_r} and {t_c}), and no time bounds or frequency attribute, so "
                    "the period each covers is unknown. Pass freq=, add time bounds or "
                    "set attrs['frequency']."
                )
            record["frequency"] = "single time step"
            record["method"] = "one time step on each side with the same timestamp"
            candidate = candidate.assign_coords(time=reference["time"].values)
            record["matched_periods"] = 1
            record["range"] = [t_r, t_r]
            return reference, candidate, record
        if unknown:
            role = unknown[0]
            other = "candidate" if role == "reference" else "reference"
            raise AlignmentError(
                f"The {role} has a single time step and no time bounds or frequency "
                "attribute, so the period it covers (one hour, day, month or year) is "
                f"unknown, and the {other} cannot be aggregated to match it. Pass "
                "freq= (for example freq='MS' when it is a monthly mean), add time "
                "bounds, or set attrs['frequency']."
            )
        target = max((i.freq for i in infos.values() if i.freq), key=nominal_days)
    record["frequency"] = freq_name(target)
    cals = record["calendars"]
    if (target == "D" or _hours(target) is not None) and (
        (cals["reference"] == "360_day") != (cals["candidate"] == "360_day")
    ):
        msg = (
            f"The reference calendar is {cals['reference']} and the candidate calendar is "
            f"{cals['candidate']}. {freq_name(target).capitalize()} values are matched by "
            "date label, so 30 February and every 31st day drop out and the same label "
            "falls at a different point of the seasonal cycle in the two calendars. Pass "
            "freq='MS' to compare monthly means, which do not depend on the calendar."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        record["calendar_note"] = msg

    out = []
    aggregated: list[str] = []
    for role, da in (("reference", reference), ("candidate", candidate)):
        info = infos[role]
        nat, step = info.freq, info.step
        if nat is not None and nominal_days(nat) > nominal_days(target) * 1.0001:
            raise AlignmentError(
                f"freq={freq!r} is finer than the {role}'s {freq_name(nat)} data; "
                "climval aggregates to coarser periods but does not disaggregate"
            )
        if nat is None and step is not None and step > nominal_days(target) * 1.05:
            raise AlignmentError(
                f"The {role}'s time step ({step:.4g} days) is coarser than freq={freq!r}"
            )
        finer = (nat is not None and nominal_days(nat) < nominal_days(target)) or (
            nat is None and step is not None
        )
        if mode == "intersect" and finer and sizes[role] > 1:
            da, agg = _aggregate(da, target, nat, step, min_coverage)
            record[f"{role}_resampled"] = agg
            aggregated.append(role)
        out.append(da)
    reference, candidate = out

    for role, info in infos.items():
        other = "candidate" if role == "reference" else "reference"
        if info.ambiguous and other in aggregated:
            raise AlignmentError(
                f"{info.ambiguous}. climval would average the {other} over whole "
                f"{freq_name(target)} periods to match, which is wrong for the second "
                "reading. Declare the native frequency with attrs['frequency'] ('yr' for "
                "yearly means, 'mon' for monthly values) or add time bounds (time_bnds)."
            )

    hours = [_hours(i.freq) for i in infos.values() if i.freq is not None]
    if (
        _hours(target) is not None
        and len(set(hours)) > 1
        and not any(i.relabelled for i in infos.values())
    ):
        msg = (
            f"The inputs have different sub-daily steps; both are compared in "
            f"{freq_name(target)} bins starting at 00 UTC. Without time bounds climval "
            "assumes each value is stamped at the start or inside its interval; data "
            "stamped at the end of their interval (accumulations) need time bounds."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        record["note"] = msg

    codes_r = _codes(reference, target)
    codes_c = _codes(candidate, target)
    for codes, role in ((codes_r, "reference"), (codes_c, "candidate")):
        if np.unique(codes).size != codes.size:
            raise AlignmentError(
                f"The {role} has several time steps in one {freq_name(target)} period; "
                "pass a coarser freq= so they are aggregated"
            )

    if mode == "exact":
        if not (codes_r.size == codes_c.size and np.array_equal(codes_r, codes_c)):
            raise AlignmentError(
                "time='exact' needs the same periods in both inputs; "
                "use time='intersect' to keep the common ones"
            )
    common = np.intersect1d(codes_r, codes_c)
    if common.size == 0:
        raise NoOverlapError(
            "The inputs share no time period at "
            f"{freq_name(target)} resolution. Reference: {_describe_range(reference)}. "
            f"Candidate: {_describe_range(candidate)}."
        )
    idx_r = np.nonzero(np.isin(codes_r, common))[0]
    idx_c = np.nonzero(np.isin(codes_c, common))[0]
    record["matched_periods"] = int(common.size)
    record["unmatched_dropped"] = {
        "reference": int(codes_r.size - idx_r.size),
        "candidate": int(codes_c.size - idx_c.size),
    }
    record["method"] = (
        f"periods matched by calendar labels ({_label_name(target)}), not by timestamp"
    )
    reference = take_time(reference, idx_r)
    candidate = take_time(candidate, idx_c)
    candidate = candidate.assign_coords(time=reference["time"].values)
    record["range"] = [str(reference["time"].values[0]), str(reference["time"].values[-1])]
    record["target_freq"] = target
    return reference, candidate, record


def time_weights(reference: xr.DataArray, freq: str | None) -> xr.DataArray | None:
    """Weight of each aligned time step: its period length (None when all equal)."""
    if freq is None or "time" not in reference.dims:
        return None
    lengths = label_lengths(reference, freq)
    if lengths is None:
        return None
    return xr.DataArray(lengths, dims="time", coords={"time": reference["time"].values})
