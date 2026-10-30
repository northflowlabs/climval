# SPDX-License-Identifier: Apache-2.0
"""Read and validate a ``climval check`` configuration file.

A configuration is YAML (``.yml``, ``.yaml``), TOML (``.toml``) or JSON
(``.json``) with a list of checks. Every check names a reference, a
candidate, a variable and the :func:`climval.compare` options, plus
thresholds on the resulting values::

    version: 1
    defaults:
      by: season
      period: ["1995", "2014"]
    checks:
      - name: europe-tas
        reference: data/era5_tas_monthly.nc
        candidate:
          catalog: https://storage.googleapis.com/cmip6/pangeo-cmip6.json
          query: {source_id: NorESM2-LM, experiment_id: historical,
                  table_id: Amon, variable_id: tas, member_id: r1i1p1f1}
          storage_options: {token: anon}
        variable: {reference: t2m, candidate: tas}
        region: {lat: [35, 72], lon: [-10, 40]}
        thresholds:
          rmse: {max: 3.0}
          mean_bias: {abs_max: 1.5, group: "*"}
          pearson_r: {min: 0.95, sample: climatology}

Everything is validated before any data is opened, so a typo fails fast
with exit code 2 and a message that names the offending key. That includes
the limits (finite numbers, ``min`` not above ``max``, ``abs_max`` not
negative) and the threshold groups, which must be names that the check's
``by`` produces. Every check needs at least one threshold: a check that
gates nothing is a configuration error, not a pass.

Limits may also be written as numeric strings such as ``"1e-3"``: PyYAML
follows YAML 1.1, which reads ``1e-3`` (no decimal point) as a string.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from climval.align.region import Box
from climval.errors import ConfigError, InputError

SUPPORTED_VERSIONS = (1,)

CHECK_OPTIONS = (
    "region",
    "period",
    "freq",
    "min_coverage",
    "by",
    "seasons",
    "regrid",
    "target",
    "time",
    "units",
    "weights",
    "metrics",
    "samples",
    "names",
)
CHECK_KEYS = frozenset(
    {
        "name",
        "description",
        "reference",
        "candidate",
        "variable",
        "variables",
        "thresholds",
        "open_kwargs",
        *CHECK_OPTIONS,
    }
)
DEFAULT_KEYS = CHECK_KEYS - {"name", "description"}
TOP_KEYS = frozenset({"version", "defaults", "checks"})
INPUT_KEYS = frozenset(
    {"path", "catalog", "query", "sel", "storage_options", "open_kwargs", "name"}
)
LIMIT_KEYS = ("min", "max", "abs_max")
THRESHOLD_KEYS = frozenset({"metric", "sample", "group", *LIMIT_KEYS})

_CHOICES: dict[str, tuple[Any, ...]] = {
    "by": ("season", "month", "year", None),
    "seasons": ("contiguous", "calendar"),
    "regrid": ("bilinear", "nearest", "conservative"),
    "target": ("reference", "candidate", "coarser"),
    "time": ("intersect", "exact"),
    "weights": ("area", "cos_lat", None),
}
_SAMPLES = ("space_time", "climatology", "space", "time")
_SCALAR_SAMPLES = ("space_time", "climatology")
_SEASONS = ("DJF", "MAM", "JJA", "SON")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_YEAR = re.compile(r"^-?\d{1,4}$")
_NUMERIC = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass
class InputSpec:
    """Where one side of a check comes from: a path or URL, or a catalogue query."""

    path: str | None = None
    catalog: str | None = None
    query: dict[str, Any] = field(default_factory=dict)
    sel: dict[str, Any] = field(default_factory=dict)
    open_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    given: str | None = None  # the path or catalogue as written in the file

    def describe(self) -> str:
        if self.path is not None:
            text = self.given or self.path
        else:
            q = ", ".join(f"{k}={v}" for k, v in self.query.items())
            catalog = str(self.given or self.catalog)
            text = f"{catalog.rstrip('/').rsplit('/', 1)[-1]} [{q}]"
        if self.sel:
            text += " sel(" + ", ".join(f"{k}={v}" for k, v in self.sel.items()) + ")"
        return text

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.path is not None:
            out["path"] = self.path
        if self.catalog is not None:
            out["catalog"] = self.catalog
            out["query"] = self.query
        if self.sel:
            out["sel"] = self.sel
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class CheckSpec:
    """One check: a comparison per variable and the thresholds that gate it."""

    name: str
    reference: InputSpec
    candidate: InputSpec
    variables: list[str | tuple[str, str] | None]
    options: dict[str, Any]
    thresholds: list[dict[str, Any]]
    description: str | None = None

    def threshold_mapping(self, item: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """One threshold in the form :func:`climval.evaluate_thresholds` takes."""
        spec = {k: v for k, v in item.items() if k != "metric"}
        return {str(item["metric"]): spec}


@dataclass
class GateConfig:
    """A parsed configuration file."""

    path: str | None
    sha256: str | None
    checks: list[CheckSpec]

    def select(self, names: Sequence[str]) -> GateConfig:
        """Keep only the named checks; unknown names are a configuration error."""
        if not names:
            return self
        known = {c.name for c in self.checks}
        missing = [n for n in names if n not in known]
        if missing:
            raise ConfigError(f"No check named {missing}; the configuration has {sorted(known)}")
        keep = [c for c in self.checks if c.name in set(names)]
        return GateConfig(path=self.path, sha256=self.sha256, checks=keep)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _parse_text(text: str, suffix: str, where: str) -> Any:
    if suffix in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError:  # pragma: no cover - PyYAML is a core dependency
            raise ConfigError("Reading YAML needs PyYAML: pip install pyyaml") from None
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{where}: not valid YAML: {exc}") from None
    if suffix == ".toml":
        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover - Python 3.10
            import tomli as tomllib
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{where}: not valid TOML: {exc}") from None
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{where}: not valid JSON: {exc}") from None
    raise ConfigError(f"{where}: use a .yml, .yaml, .toml or .json file, got {suffix!r}")


def load_config(path: str | os.PathLike[str]) -> GateConfig:
    """Read and validate a configuration file.

    Relative data paths in the file are read relative to the file's directory.

    Raises
    ------
    climval.errors.ConfigError
        The file is missing, cannot be parsed, or fails validation.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: {p}")
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ConfigError(f"{p}: the file is not UTF-8 text") from None
    data = _parse_text(text, p.suffix.lower(), str(p))
    config = parse_config(data, base_dir=p.parent)
    config.path = os.fspath(path)
    config.sha256 = hashlib.sha256(raw).hexdigest()
    return config


def parse_config(data: Any, base_dir: str | os.PathLike[str] | None = None) -> GateConfig:
    """Validate an already parsed configuration (a dict) and return a GateConfig."""
    if not isinstance(data, Mapping):
        raise ConfigError("The configuration must be a mapping with a 'checks' list")
    _no_unknown(data, TOP_KEYS, "configuration")
    version = data.get("version", 1)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in SUPPORTED_VERSIONS
    ):
        raise ConfigError(f"version: unsupported configuration version {version!r}; use 1")
    defaults = data.get("defaults", {}) or {}
    if not isinstance(defaults, Mapping):
        raise ConfigError("defaults: must be a mapping")
    _no_unknown(defaults, DEFAULT_KEYS, "defaults")
    checks_raw = data.get("checks")
    if not isinstance(checks_raw, list) or not checks_raw:
        raise ConfigError("checks: must be a non-empty list")
    base = Path(base_dir) if base_dir is not None else None
    checks: list[CheckSpec] = []
    seen: set[str] = set()
    for i, item in enumerate(checks_raw):
        where = f"checks[{i}]"
        if not isinstance(item, Mapping):
            raise ConfigError(f"{where}: must be a mapping")
        _no_unknown(item, CHECK_KEYS, where)
        merged = {**defaults, **item}
        if "variable" in item and "variables" in defaults:
            merged.pop("variables")
        if "variables" in item and "variable" in defaults:
            merged.pop("variable")
        spec = _parse_check(merged, where, base, index=i)
        if spec.name in seen:
            raise ConfigError(f"{where}.name: duplicate check name {spec.name!r}")
        seen.add(spec.name)
        checks.append(spec)
    return GateConfig(path=None, sha256=None, checks=checks)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _no_unknown(obj: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(str(k) for k in obj if k not in allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}; allowed: {sorted(allowed)}")


def _str_map(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: must be a mapping")
    return {str(k): v for k, v in value.items()}


def _resolve_path(path: str, base: Path | None) -> str:
    if "://" in path or base is None or os.path.isabs(path):
        return path
    return str((base / path).resolve()) if not any(ch in path for ch in "*?[") else str(base / path)


def _parse_input(value: Any, where: str, base: Path | None) -> InputSpec:
    if isinstance(value, (str, os.PathLike)):
        return InputSpec(path=_resolve_path(os.fspath(value), base), given=os.fspath(value))
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: must be a path or URL, or a mapping with 'path' or 'catalog'")
    _no_unknown(value, INPUT_KEYS, where)
    has_path = value.get("path") is not None
    has_cat = value.get("catalog") is not None
    if has_path == has_cat:
        raise ConfigError(f"{where}: give exactly one of 'path' or 'catalog'")
    open_kwargs = _str_map(value.get("open_kwargs"), f"{where}.open_kwargs")
    storage = value.get("storage_options")
    if storage is not None:
        open_kwargs["storage_options"] = _str_map(storage, f"{where}.storage_options")
    spec = InputSpec(
        sel=_str_map(value.get("sel"), f"{where}.sel"),
        open_kwargs=open_kwargs,
        name=str(value["name"]) if value.get("name") is not None else None,
    )
    if has_path:
        spec.path = _resolve_path(str(value["path"]), base)
        spec.given = str(value["path"])
        if "query" in value:
            raise ConfigError(f"{where}.query: only used with 'catalog'")
    else:
        catalog = str(value["catalog"])
        spec.catalog = _resolve_path(catalog, base)
        spec.given = catalog
        query = _str_map(value.get("query"), f"{where}.query")
        if not query:
            raise ConfigError(f"{where}.query: a catalogue input needs a query mapping")
        spec.query = query
    return spec


def _parse_variable(value: Any, where: str) -> str | tuple[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        _no_unknown(value, frozenset({"reference", "candidate"}), where)
        if "reference" not in value or "candidate" not in value:
            raise ConfigError(f"{where}: a variable mapping needs 'reference' and 'candidate'")
        return (str(value["reference"]), str(value["candidate"]))
    raise ConfigError(
        f"{where}: a variable is a name, or {{reference: NAME, candidate: NAME}} when the "
        "names differ; use 'variables' for a list"
    )


def _parse_region(value: Any, where: str) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str):
        nums = _numbers(value, where)
        if len(nums) == 2:
            return Box(lat=(nums[0], nums[1]))
        if len(nums) == 4:
            return Box(lat=(nums[0], nums[1]), lon=(nums[2], nums[3]))
        raise ConfigError(f"{where}: a region list is [south, north] or [south, north, west, east]")
    if isinstance(value, Mapping):
        _no_unknown(value, frozenset({"lat", "lon", "name"}), where)
        try:
            lat = _numbers(value.get("lat", [-90, 90]), f"{where}.lat")
            lon = value.get("lon")
            lon_t = _numbers(lon, f"{where}.lon") if lon is not None else None
            if len(lat) != 2 or (lon_t is not None and len(lon_t) != 2):
                raise ConfigError(f"{where}: lat and lon each take two numbers")
            return Box(
                lat=(lat[0], lat[1]),
                lon=(lon_t[0], lon_t[1]) if lon_t is not None else None,
                name=str(value["name"]) if value.get("name") is not None else None,
            )
        except ConfigError:
            raise
        except ValueError as exc:
            raise ConfigError(f"{where}: {exc}") from None
    raise ConfigError(f"{where}: a region is a regionmask name, a list or a mapping")


def _numbers(value: Any, where: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ConfigError(f"{where}: expected a list of numbers")
    if any(isinstance(v, bool) for v in value):
        raise ConfigError(f"{where}: expected numbers, got {list(value)}")
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected numbers, got {list(value)}") from None
    if not all(math.isfinite(v) for v in nums):
        raise ConfigError(f"{where}: expected finite numbers, got {list(value)}")
    return nums


def _limit(value: Any, where: str) -> float:
    """A threshold limit: a number, or a numeric string such as "1e-3"."""
    given = value
    if isinstance(value, str) and _NUMERIC.match(value.strip()):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: must be a number, got {given!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{where}: must be a finite number, got {given!r}")
    return value


def _check_group(group: str, by: str | None, where: str) -> None:
    """A threshold group must be one that the check's ``by`` produces."""
    if group in ("all", "*"):
        return
    if by is None:
        raise ConfigError(f"{where}: group {group!r} needs 'by' (season, month or year)")
    if by == "season":
        known, hint = group in _SEASONS, f"one of {list(_SEASONS)}"
    elif by == "month":
        known, hint = group in _MONTHS, f"one of {list(_MONTHS)}"
    else:
        known, hint = bool(_YEAR.match(group)), "a year such as '2001'"
    if not known:
        raise ConfigError(
            f"{where}: group {group!r} is not a group of by={by!r}; use 'all', '*' or {hint}"
        )


def _parse_thresholds(
    value: Any, where: str, metrics: list[str] | None, samples: list[str], by: str | None
) -> list[dict[str, Any]]:
    from climval.metrics.stats import DEFAULT_METRICS, get_metric
    from climval.thresholds import validate_limits

    if value is None or (isinstance(value, (Mapping, list)) and not value):
        raise ConfigError(
            f"{where}: a check needs at least one threshold, for example "
            "rmse: {max: 2.0}; a check without thresholds would pass whatever the values"
        )
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, Mapping):
        for metric, spec in value.items():
            specs = spec if isinstance(spec, list) else [spec]
            for j, s in enumerate(specs):
                if not isinstance(s, Mapping):
                    raise ConfigError(f"{where}.{metric}: a threshold is a mapping of limits")
                if "metric" in s:
                    raise ConfigError(f"{where}.{metric}: 'metric' is given by the key")
                items.append(
                    (
                        f"{where}.{metric}" + (f"[{j}]" if len(specs) > 1 else ""),
                        {"metric": str(metric), **s},
                    )
                )
    elif isinstance(value, list):
        for j, s in enumerate(value):
            if not isinstance(s, Mapping) or "metric" not in s:
                raise ConfigError(f"{where}[{j}]: a threshold in a list needs 'metric'")
            items.append((f"{where}[{j}]", dict(s)))
    else:
        raise ConfigError(f"{where}: thresholds are a mapping or a list")

    computed = {m.name for m in DEFAULT_METRICS}
    out: list[dict[str, Any]] = []
    for loc, item in items:
        _no_unknown(item, THRESHOLD_KEYS, loc)
        metric = str(item["metric"])
        if metric != "n_points":
            try:
                canonical = get_metric(metric).name
            except ValueError as exc:
                raise ConfigError(f"{loc}: {exc}") from None
            metric = canonical
            if metrics is not None and metric not in metrics:
                raise ConfigError(f"{loc}: metric {metric!r} is not in this check's metrics")
            if metrics is None and metric not in computed:
                raise ConfigError(
                    f"{loc}: metric {metric!r} is not computed by default; list the metrics "
                    f"to compute under this check's 'metrics:', including {metric!r}"
                )
        limits = {k: _limit(item[k], f"{loc}.{k}") for k in LIMIT_KEYS if k in item}
        if not limits:
            raise ConfigError(f"{loc}: give at least one of {list(LIMIT_KEYS)}")
        try:
            validate_limits(limits, loc)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
        sample = str(item.get("sample", "space_time"))
        if sample not in _SCALAR_SAMPLES:
            raise ConfigError(f"{loc}.sample: thresholds apply to {list(_SCALAR_SAMPLES)}")
        if sample not in samples:
            raise ConfigError(f"{loc}.sample: {sample!r} is not in this check's samples")
        clean: dict[str, Any] = {"metric": metric, "sample": sample, **limits}
        if "group" in item:
            group = item["group"]
            if isinstance(group, bool) or not isinstance(group, (str, int)):
                raise ConfigError(f"{loc}.group: must be a group name, got {group!r}")
            clean["group"] = str(group)
            _check_group(clean["group"], by, f"{loc}.group")
        out.append(clean)
    return out


def _parse_check(item: Mapping[str, Any], where: str, base: Path | None, index: int) -> CheckSpec:
    for key in ("reference", "candidate"):
        if item.get(key) is None:
            raise ConfigError(f"{where}: '{key}' is required")
    reference = _parse_input(item["reference"], f"{where}.reference", base)
    candidate = _parse_input(item["candidate"], f"{where}.candidate", base)

    if "variable" in item and "variables" in item:
        raise ConfigError(f"{where}: give 'variable' or 'variables', not both")
    if "variables" in item:
        raw = item["variables"]
        if not isinstance(raw, list) or not raw:
            raise ConfigError(f"{where}.variables: must be a non-empty list")
        variables = [_parse_variable(v, f"{where}.variables[{j}]") for j, v in enumerate(raw)]
    else:
        variables = [_parse_variable(item.get("variable"), f"{where}.variable")]

    options: dict[str, Any] = {}
    for key in CHECK_OPTIONS:
        if key not in item:
            continue
        value = item[key]
        loc = f"{where}.{key}"
        if key in _CHOICES:
            if isinstance(value, str) and value.lower() == "none":
                value = None
            if value not in _CHOICES[key]:
                raise ConfigError(f"{loc}: must be one of {list(_CHOICES[key])}, got {value!r}")
        elif key == "region":
            try:
                value = _parse_region(value, loc)
            except InputError as exc:
                raise ConfigError(f"{loc}: {exc}") from None
        elif key == "period":
            if not isinstance(value, list) or len(value) != 2:
                raise ConfigError(f"{loc}: must be [start, end], for example ['1995', '2014']")
            value = (str(value[0]), str(value[1]))
        elif key == "freq":
            from climval.align.temporal import normalise_freq
            from climval.errors import AlignmentError

            try:
                normalise_freq(str(value))
            except AlignmentError as exc:
                raise ConfigError(f"{loc}: {exc}") from None
            value = str(value)
        elif key == "min_coverage":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{loc}: must be a number in (0, 1], got {value!r}")
            if not 0 < value <= 1:
                raise ConfigError(f"{loc}: must be in (0, 1], got {value!r}")
            value = float(value)
        elif key == "units":
            if isinstance(value, Mapping):
                _no_unknown(value, frozenset({"reference", "candidate"}), loc)
                value = {str(k): str(v) for k, v in value.items()}
            elif value not in ("convert", "strict", "ignore"):
                raise ConfigError(f"{loc}: 'convert', 'strict', 'ignore' or a mapping")
        elif key == "metrics":
            from climval.metrics.stats import get_metric

            if not isinstance(value, list) or not value:
                raise ConfigError(f"{loc}: must be a non-empty list of metric names")
            try:
                value = [get_metric(str(m)).name for m in value]
            except ValueError as exc:
                raise ConfigError(f"{loc}: {exc}") from None
        elif key == "samples":
            if not isinstance(value, list) or not value:
                raise ConfigError(f"{loc}: must be a non-empty list")
            bad = [s for s in value if s not in _SAMPLES]
            if bad:
                raise ConfigError(f"{loc}: unknown samples {bad}; choose from {list(_SAMPLES)}")
            value = [str(s) for s in value]
        elif key == "names":
            if not isinstance(value, list) or len(value) != 2:
                raise ConfigError(f"{loc}: must be [reference name, candidate name]")
            value = (str(value[0]), str(value[1]))
        options[key] = value
    if "open_kwargs" in item:
        options["open_kwargs"] = _str_map(item["open_kwargs"], f"{where}.open_kwargs")

    samples = list(options.get("samples", ["space_time", "climatology"]))
    thresholds = _parse_thresholds(
        item.get("thresholds"),
        f"{where}.thresholds",
        options.get("metrics"),
        samples,
        options.get("by"),
    )

    name = item.get("name")
    if name is None:
        name = f"check-{index + 1}"
    return CheckSpec(
        name=str(name),
        reference=reference,
        candidate=candidate,
        variables=variables,
        options=options,
        thresholds=thresholds,
        description=str(item["description"]) if item.get("description") is not None else None,
    )
