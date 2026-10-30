"""Unit checks and conversion.

The candidate is converted to the reference's units. A small built-in table
covers the common climate cases without any dependency:

* temperature: K, degC, degF
* precipitation flux: kg m-2 s-1, kg m-2 h-1, kg m-2 d-1, mm s-1, mm h-1,
  mm day-1 (liquid water, 1 kg m-2 equals 1 mm). Amounts per month are not
  in the table, because a month has no fixed length.
* pressure: Pa, hPa (mbar), kPa
* fraction: 1, %
* speed: m s-1, km h-1
* energy flux: W m-2

Anything else is converted with pint when it is installed (``units`` extra),
using cf_xarray's CF-aware unit registry when available. Each conversion is
affine (``a * x + b``), so it is applied lazily and recorded with its
coefficients. A mismatch that cannot be converted raises :class:`UnitsError`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import xarray as xr

from climval.errors import UnitsError

# canonical unit: (quantity, scale, offset), so base = value * scale + offset
_TABLE: dict[str, tuple[str, float, float]] = {
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, 273.15),
    "degF": ("temperature", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    "kg m-2 s-1": ("precipitation_flux", 1.0, 0.0),
    "mm h-1": ("precipitation_flux", 1.0 / 3600.0, 0.0),
    "mm day-1": ("precipitation_flux", 1.0 / 86400.0, 0.0),
    "kg m-2 h-1": ("precipitation_flux", 1.0 / 3600.0, 0.0),
    "kg m-2 d-1": ("precipitation_flux", 1.0 / 86400.0, 0.0),
    "Pa": ("pressure", 1.0, 0.0),
    "hPa": ("pressure", 100.0, 0.0),
    "kPa": ("pressure", 1000.0, 0.0),
    "1": ("fraction", 1.0, 0.0),
    "%": ("fraction", 0.01, 0.0),
    "m s-1": ("speed", 1.0, 0.0),
    "km h-1": ("speed", 1.0 / 3.6, 0.0),
    "W m-2": ("energy_flux", 1.0, 0.0),
}

_ALIASES: dict[str, tuple[str, ...]] = {
    "K": ("K", "kelvin", "degK", "deg_K", "degrees_K", "degree_Kelvin"),
    "degC": (
        "degC",
        "deg_C",
        "°C",
        "celsius",
        "degrees_C",
        "degree_C",
        "degree_Celsius",
        "degrees_Celsius",
        "degrees Celsius",
        "deg C",
    ),
    "degF": ("degF", "deg_F", "°F", "fahrenheit", "degrees_F", "degree_Fahrenheit"),
    "kg m-2 s-1": (
        "kg m-2 s-1",
        "kg m^-2 s^-1",
        "kg m**-2 s**-1",
        "kg/m2/s",
        "kg/m^2/s",
        "kg m-2.s-1",
        "kg.m-2.s-1",
        "mm s-1",
        "mm/s",
        "mm s^-1",
    ),
    "mm h-1": ("mm h-1", "mm/h", "mm/hr", "mm hr-1", "mm h^-1"),
    "kg m-2 h-1": ("kg m-2 h-1", "kg m^-2 h^-1", "kg m**-2 h**-1", "kg/m2/h", "kg/m^2/h"),
    "kg m-2 d-1": (
        "kg m-2 d-1",
        "kg m-2 day-1",
        "kg m^-2 d^-1",
        "kg m^-2 day^-1",
        "kg m**-2 d**-1",
        "kg m**-2 day**-1",
        "kg/m2/d",
        "kg/m2/day",
        "kg/m^2/day",
    ),
    "mm day-1": (
        "mm day-1",
        "mm/day",
        "mm d-1",
        "mm/d",
        "mm day^-1",
        "mm.day-1",
        "mm d^-1",
        "mm per day",
    ),
    "Pa": ("Pa", "pascal"),
    "hPa": ("hPa", "mbar", "mb", "millibar", "hectopascal"),
    "kPa": ("kPa", "kilopascal"),
    "1": ("1", "fraction", "dimensionless", "0-1"),
    "%": ("%", "percent", "pct"),
    "m s-1": ("m s-1", "m/s", "m s^-1", "m s**-1", "m.s-1"),
    "km h-1": ("km h-1", "km/h", "kph"),
    "W m-2": ("W m-2", "W/m2", "W m^-2", "W m**-2", "W/m^2", "W.m-2"),
}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


# Spelled-out names are matched without regard to case ("Kelvin", "Percent").
# Symbols are matched exactly, because case carries meaning there: "MB" is
# not "mb", "mm" is not "Mm".
_WORDS = {
    "kelvin": "K",
    "degree_kelvin": "K",
    "celsius": "degC",
    "degree_celsius": "degC",
    "degrees_celsius": "degC",
    "degrees celsius": "degC",
    "fahrenheit": "degF",
    "degree_fahrenheit": "degF",
    "pascal": "Pa",
    "millibar": "hPa",
    "hectopascal": "hPa",
    "kilopascal": "kPa",
    "fraction": "1",
    "dimensionless": "1",
    "percent": "%",
}
_LOOKUP: dict[str, str] = {}
for _canon, _names in _ALIASES.items():
    for _n in _names:
        _LOOKUP[_squash(_n)] = _canon

# Zero of the absolute scale in each temperature unit, for ratio-scale metrics.
_ABSOLUTE_ZERO_OFFSET = {"K": 0.0, "degC": 273.15, "degF": 459.67}


def canonical(units: str | None) -> str | None:
    """Canonical spelling from the built-in table, or None when unknown."""
    if units is None:
        return None
    key = _squash(str(units))
    if key in _LOOKUP:
        return _LOOKUP[key]
    return _WORDS.get(key.lower())


def absolute_offset(units: str | None) -> float:
    """What to add to values in ``units`` so that zero is absolute zero.

    273.15 for degC and 459.67 for degF (degF plus 459.67 is Rankine, whose
    ratios equal those in K); 0 for every other unit, including units pint
    does not know as temperatures.
    """
    canon = canonical(units)
    if canon in _ABSOLUTE_ZERO_OFFSET:
        return _ABSOLUTE_ZERO_OFFSET[canon]
    if units is None or canon is not None:
        return 0.0
    try:
        import pint

        registry: Any
        try:
            from cf_xarray.units import units as registry
        except Exception:  # noqa: BLE001
            registry = pint.UnitRegistry()
        unit = registry.Unit(str(units))
        if unit.dimensionality != registry.Unit("K").dimensionality:
            return 0.0
        zero = registry.Quantity(0.0, unit).to("K").magnitude
        scale = registry.Quantity(1.0, unit).to("K").magnitude - zero
        return float(zero / scale) if scale else 0.0
    except Exception:  # noqa: BLE001  pint missing or unit unknown
        return 0.0


def same_units(a: str, b: str) -> bool:
    ca, cb = canonical(a), canonical(b)
    if ca is not None and cb is not None:
        return ca == cb
    return _squash(a) == _squash(b)


def affine(source: str, target: str) -> tuple[float, float, str]:
    """Coefficients (a, b) with target_value = a * source_value + b, and the method."""
    cs, ct = canonical(source), canonical(target)
    if cs is not None and ct is not None:
        qs, ss, os_ = _TABLE[cs]
        qt, st, ot = _TABLE[ct]
        if qs != qt:
            raise UnitsError(
                f"Cannot convert {source!r} ({qs.replace('_', ' ')}) to "
                f"{target!r} ({qt.replace('_', ' ')})"
            )
        return ss / st, (os_ - ot) / st, "built-in table"
    return _pint_affine(source, target)


def _pint_affine(source: str, target: str) -> tuple[float, float, str]:
    try:
        import pint
    except ImportError:
        raise UnitsError(
            f"Units differ ({source!r} and {target!r}) and are not in climval's "
            'built-in table. Install pint (pip install "climval[units]") to '
            "convert other units, or pass units='ignore' to compare as given."
        ) from None
    registry: Any
    try:
        from cf_xarray.units import units as cf_units

        registry = cf_units
    except Exception:  # noqa: BLE001
        registry = pint.UnitRegistry()
    try:
        zero = registry.Quantity(0.0, source).to(target).magnitude
        one = registry.Quantity(1.0, source).to(target).magnitude
    except Exception as exc:  # noqa: BLE001
        raise UnitsError(
            f"Cannot convert {source!r} to {target!r}: {exc}. Pass units='ignore' "
            "to compare as given, or declare the units with "
            "units={'reference': ..., 'candidate': ...}."
        ) from exc
    return float(one - zero), float(zero), "pint"


def harmonise_units(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    mode: str | Mapping[str, str] = "convert",
) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
    """Check units and convert the candidate into the reference's units.

    ``mode`` is "convert" (default), "strict" (must already match), "ignore",
    or a mapping {"reference": ..., "candidate": ...} that declares the units
    of either input (overriding their attributes) before converting.
    """
    record: dict[str, Any] = {"step": "units"}
    declared: Mapping[str, str] = {}
    if isinstance(mode, Mapping):
        declared = mode
        unknown = set(declared) - {"reference", "candidate"}
        if unknown:
            raise UnitsError(f"units= mapping accepts 'reference' and 'candidate', got {unknown}")
        mode = "convert"
        record["declared"] = dict(declared)
    if mode not in ("convert", "strict", "ignore"):
        raise UnitsError(f"units must be 'convert', 'strict', 'ignore' or a mapping, got {mode!r}")

    ru = declared.get("reference", reference.attrs.get("units"))
    cu = declared.get("candidate", candidate.attrs.get("units"))
    ru = None if ru in (None, "") else str(ru)
    cu = None if cu in (None, "") else str(cu)
    record.update({"reference_units": ru, "candidate_units": cu, "mode": mode})

    if mode == "ignore":
        record["action"] = "not checked (units='ignore')"
        return reference, candidate, record
    if ru is None and cu is None:
        record["action"] = "no units on either input; values compared as given"
        return reference, candidate, record
    if ru is None or cu is None:
        missing = "reference" if ru is None else "candidate"
        raise UnitsError(
            f"The {missing} has no units attribute, so the inputs cannot be "
            f"checked against each other. Declare it with units={{'{missing}': '...'}} "
            "or pass units='ignore'."
        )
    for role, u in (("reference", ru), ("candidate", cu)):
        if _squash(u) == "C":
            raise UnitsError(
                f"The {role} has units 'C', which is the coulomb in UDUNITS and CF. Write "
                f"'degC' for degrees Celsius, or declare it with units={{'{role}': 'degC'}}."
            )
    reference = reference.assign_attrs(units=ru)
    if same_units(ru, cu):
        record["action"] = "units match"
        return reference, candidate.assign_attrs(units=ru), record
    if mode == "strict":
        raise UnitsError(f"Units differ: reference {ru!r}, candidate {cu!r} (units='strict')")
    a, b, how = affine(cu, ru)
    converted = candidate * a + b if b != 0.0 else candidate * a
    converted = converted.assign_attrs({**candidate.attrs, "units": ru})
    converted.name = candidate.name
    record.update(
        {
            "action": f"candidate converted from {cu!r} to {ru!r}",
            "scale": a,
            "offset": b,
            "via": how,
        }
    )
    return reference, converted, record
