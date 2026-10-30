"""Time alignment across calendars and frequencies.

Two inputs rarely share timestamps: a model in a ``noleap`` or ``360_day``
calendar stamps its monthly means on different days than a reanalysis in
the standard calendar, and one may be daily while the other is monthly.
climval therefore does not match timestamps. It

1. selects the requested period on calendar fields (year, month, day), so
   the same period string means the same thing in every calendar;
2. finds each input's native step and the common frequency, which is the
   coarser of the two or the user's ``freq`` (never finer than either);
3. aggregates any finer input to that frequency with a time mean (monthly
   data aggregated to years is weighted by month length), and drops periods
   that are incomplete (fewer native steps than the period holds in that
   input's own calendar);
4. labels every period by its calendar fields (for monthly data the year
   and month) and keeps the labels present in both inputs.

The aligned pair carries the reference's time coordinate. Daily comparisons
across calendars match days by label, so days that exist in only one
calendar (29 February, 30 February, 31st days) drop out; monthly or coarser
comparisons are calendar independent.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climval.errors import AlignmentError, NoOverlapError

ORDER = ("h", "D", "MS", "YS")
NOMINAL_DAYS = {"h": 1.0 / 24.0, "D": 1.0, "MS": 30.44, "YS": 365.25}
_ALIASES = {
    "h": "h",
    "H": "h",
    "1h": "h",
    "1H": "h",
    "hour": "h",
    "hourly": "h",
    "D": "D",
    "1D": "D",
    "d": "D",
    "day": "D",
    "daily": "D",
    "MS": "MS",
    "M": "MS",
    "ME": "MS",
    "mon": "MS",
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
}
_NAMES = {"h": "hourly", "D": "daily", "MS": "monthly", "YS": "yearly"}
_LABELS = {"h": "date-hour", "D": "date", "MS": "year-month", "YS": "year"}


def normalise_freq(freq: str) -> str:
    try:
        return _ALIASES[freq]
    except KeyError:
        raise AlignmentError(
            f"Unsupported freq {freq!r}; use 'h', 'D', 'MS' (monthly) or 'YS' (yearly)"
        ) from None


def calendar(da: xr.DataArray) -> str:
    try:
        return str(da["time"].dt.calendar)
    except (AttributeError, TypeError):
        return "unknown"


def _is_cftime(da: xr.DataArray) -> bool:
    return bool(da["time"].dtype == object)


def _step_days(da: xr.DataArray) -> float | None:
    if da.sizes["time"] < 2:
        return None
    idx = da.indexes["time"]
    deltas = idx[1:] - idx[:-1]
    seconds = np.asarray(pd.TimedeltaIndex(deltas).total_seconds(), dtype=float)
    return float(np.median(seconds)) / 86400.0


def classify_step(step: float | None) -> str | None:
    """Frequency class of a median time step in days."""
    if step is None:
        return None
    if step < 0.95:
        return "h"
    if step <= 1.05:
        return "D"
    if 27.0 <= step <= 32.0:
        return "MS"
    if 359.0 <= step <= 367.0:
        return "YS"
    return None


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
    hour = np.asarray(t.hour.values, dtype=np.int64)
    codes: np.ndarray = year * 1000000 + month * 10000 + day * 100 + hour
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
    out: xr.DataArray = da.isel(time=np.nonzero(keep)[0])
    return out


def _describe_range(da: xr.DataArray) -> str:
    t = da["time"].values
    return f"{t[0]} to {t[-1]}, calendar {calendar(da)}"


def _period_lengths(labels: np.ndarray, freq: str, cal: str, cftime: bool) -> np.ndarray:
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


def _aggregate(
    da: xr.DataArray, freq: str, native: str | None, step: float | None
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Time mean onto ``freq``, dropping incomplete periods."""
    cftime = _is_cftime(da)
    cal = calendar(da)
    if native == "MS":
        days = da["time"].dt.days_in_month.astype(float)
        num = (da * days).resample(time=freq).sum(skipna=True, min_count=1)
        den = days.where(da.notnull()).resample(time=freq).sum(skipna=True, min_count=1)
        out = num / den
        how = "month-length weighted mean"
    else:
        out = da.resample(time=freq).mean()
        how = "mean"
    ones = xr.DataArray(np.ones(da.sizes["time"]), dims="time", coords={"time": da["time"].values})
    counts = np.asarray(ones.resample(time=freq).count().values, dtype=float)
    labels = np.asarray(out["time"].values)
    info: dict[str, Any] = {"aggregation": how, "periods": int(labels.size)}
    if step is not None and labels.size:
        lengths = _period_lengths(labels, freq, cal, cftime)
        expected = np.maximum(np.rint(lengths / step), 1.0)
        complete = counts >= expected
        dropped = [str(lab) for lab, ok in zip(labels, complete, strict=False) if not ok]
        info["incomplete_periods_dropped"] = dropped
        if not complete.all():
            out = out.isel(time=np.nonzero(complete)[0])
    return out, info


def align_time(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    freq: str | None = None,
    period: tuple[Any, Any] | None = None,
    mode: str = "intersect",
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

    record["calendars"] = {"reference": calendar(reference), "candidate": calendar(candidate)}
    if period is not None:
        reference = select_period(reference, period, "reference")
        candidate = select_period(candidate, period, "candidate")
        record["period"] = [str(period[0]), str(period[1])]

    step_r, step_c = _step_days(reference), _step_days(candidate)
    nat_r, nat_c = classify_step(step_r), classify_step(step_c)
    record["native_frequency"] = {
        "reference": _NAMES.get(nat_r or "", f"step {step_r} days"),
        "candidate": _NAMES.get(nat_c or "", f"step {step_c} days"),
    }

    if freq is not None:
        target = normalise_freq(freq)
    else:
        known = [f for f in (nat_r, nat_c) if f is not None]
        for step, nat, role in ((step_r, nat_r, "reference"), (step_c, nat_c, "candidate")):
            if step is not None and nat is None:
                raise AlignmentError(
                    f"Cannot classify the {role}'s time step ({step:.3g} days). "
                    "Pass freq='D', 'MS' or 'YS' to aggregate to a common frequency."
                )
        target = max(known, key=ORDER.index) if known else "D"
    record["frequency"] = _NAMES[target]

    out = []
    for da, nat, step, role in (
        (reference, nat_r, step_r, "reference"),
        (candidate, nat_c, step_c, "candidate"),
    ):
        native_days = step if step is not None else None
        if nat is not None and ORDER.index(nat) > ORDER.index(target):
            raise AlignmentError(
                f"freq={freq!r} is finer than the {role}'s {_NAMES[nat]} data; "
                "climval aggregates to coarser periods but does not disaggregate"
            )
        if nat is None and native_days is not None and native_days > NOMINAL_DAYS[target] * 1.05:
            raise AlignmentError(
                f"The {role}'s time step ({native_days:.3g} days) is coarser than freq={freq!r}"
            )
        finer = (nat is not None and ORDER.index(nat) < ORDER.index(target)) or (
            nat is None and native_days is not None
        )
        if mode == "intersect" and finer and da.sizes["time"] > 1:
            da, info = _aggregate(da, target, nat, step)
            record[f"{role}_resampled"] = info
        out.append(da)
    reference, candidate = out

    codes_r = _codes(reference, target)
    codes_c = _codes(candidate, target)
    for codes, role in ((codes_r, "reference"), (codes_c, "candidate")):
        if np.unique(codes).size != codes.size:
            raise AlignmentError(
                f"The {role} has several time steps in one {_NAMES[target]} period; "
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
            f"{_NAMES[target]} resolution. Reference: {_describe_range(reference)}. "
            f"Candidate: {_describe_range(candidate)}."
        )
    idx_r = np.nonzero(np.isin(codes_r, common))[0]
    idx_c = np.nonzero(np.isin(codes_c, common))[0]
    record["matched_periods"] = int(common.size)
    record["unmatched_dropped"] = {
        "reference": int(codes_r.size - idx_r.size),
        "candidate": int(codes_c.size - idx_c.size),
    }
    record["method"] = f"periods matched by calendar labels ({_LABELS[target]}), not by timestamp"
    reference = reference.isel(time=idx_r)
    candidate = candidate.isel(time=idx_c)
    candidate = candidate.assign_coords(time=reference["time"].values)
    record["range"] = [str(reference["time"].values[0]), str(reference["time"].values[-1])]
    return reference, candidate, record
