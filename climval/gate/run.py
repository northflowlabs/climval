"""Run the checks of a :class:`~climval.gate.config.GateConfig`.

Each check runs :func:`climval.compare` once per variable and evaluates its
thresholds with :func:`climval.evaluate_thresholds`. A check that cannot run
(missing file, failed catalogue query, alignment error) is recorded as an
error and the remaining checks still run.

Exit codes: 0 when every check passes, 1 when a threshold fails and nothing
else went wrong, 2 when the configuration is invalid or any check could not
run.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from climval.errors import ConfigError, InputError
from climval.gate.config import CheckSpec, GateConfig, InputSpec
from climval.provenance import environment, jsonable, timestamp

if TYPE_CHECKING:
    from climval.result import Comparison
    from climval.thresholds import ThresholdResult

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

_CORE_DIMS = {"time", "lat", "lon", "latitude", "longitude", "x", "y", "i", "j", "nlat", "nlon"}


@dataclass
class CheckOutcome:
    """The result of one check for one variable."""

    check: str
    variable: str
    status: str  # "pass", "fail" or "error"
    reference: str
    candidate: str
    description: str | None = None
    comparison: Comparison | None = None
    thresholds: ThresholdResult | None = None
    error: str | None = None
    error_type: str | None = None
    detail: str | None = None
    seconds: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.check if not self.variable else f"{self.check} ({self.variable})"

    @property
    def n_checks(self) -> int:
        return len(self.thresholds.checks) if self.thresholds is not None else 0

    @property
    def n_failed(self) -> int:
        return len(self.thresholds.failures) if self.thresholds is not None else 0

    def to_dict(self, include_comparison: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "check": self.check,
            "variable": self.variable,
            "description": self.description,
            "status": self.status,
            "reference": self.reference,
            "candidate": self.candidate,
            "inputs": self.inputs,
            "seconds": round(self.seconds, 3),
            "thresholds": self.thresholds.to_dict()["checks"] if self.thresholds else [],
        }
        if self.error is not None:
            out["error"] = {"type": self.error_type, "message": self.error}
        if include_comparison and self.comparison is not None:
            out["comparison"] = self.comparison.to_dict()
        return out


@dataclass
class GateReport:
    """Everything a ``climval check`` run produced."""

    outcomes: list[CheckOutcome]
    config_path: str | None = None
    config_sha256: str | None = None
    created: str = field(default_factory=timestamp)
    config_error: str | None = None

    @property
    def exit_code(self) -> int:
        if self.config_error is not None or any(o.status == "error" for o in self.outcomes):
            return EXIT_ERROR
        if any(o.status == "fail" for o in self.outcomes):
            return EXIT_FAIL
        return EXIT_PASS

    @property
    def passed(self) -> bool:
        return self.exit_code == EXIT_PASS

    @property
    def status(self) -> str:
        return {EXIT_PASS: "pass", EXIT_FAIL: "fail", EXIT_ERROR: "error"}[self.exit_code]

    def counts(self) -> dict[str, int]:
        return {s: sum(o.status == s for o in self.outcomes) for s in ("pass", "fail", "error")}

    def to_dict(self, include_comparison: bool = True) -> dict[str, Any]:
        from climval import __version__

        return jsonable(  # type: ignore[no-any-return]
            {
                "climval": __version__,
                "created": self.created,
                "config": {"path": self.config_path, "sha256": self.config_sha256},
                "status": self.status,
                "passed": self.passed,
                "exit_code": self.exit_code,
                "counts": self.counts(),
                "config_error": self.config_error,
                "environment": environment(),
                "checks": [o.to_dict(include_comparison) for o in self.outcomes],
            }
        )

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            _write(path, text)
        return text

    def summary_text(self) -> str:
        from climval.gate.render import render_text

        return render_text(self)

    def to_markdown(self) -> str:
        from climval.gate.render import render_markdown

        return render_markdown(self)

    def to_html(self) -> str:
        from climval.gate.render import render_html

        return render_html(self)


def _write(path: str | Path, text: str, append: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if append else "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Opening inputs
# ---------------------------------------------------------------------------


class _Catalogs:
    """Open each intake-esm catalogue once per run."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, url: str) -> Any:
        if url not in self._cache:
            try:
                from intake_esm.core import esm_datastore
            except ImportError:
                from climval.errors import MissingDependencyError

                raise MissingDependencyError(
                    "A catalogue input needs intake-esm. "
                    'Install it with: pip install "climval[catalog]"'
                ) from None
            self._cache[url] = esm_datastore(url)  # type: ignore[arg-type]
        return self._cache[url]


def _split_open_kwargs(spec: InputSpec, check_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**check_kwargs, **spec.open_kwargs}
    if "storage_options" in check_kwargs and "storage_options" in spec.open_kwargs:
        merged["storage_options"] = {
            **check_kwargs["storage_options"],
            **spec.open_kwargs["storage_options"],
        }
    return merged


def open_input(
    spec: InputSpec, check_kwargs: Mapping[str, Any], catalogs: _Catalogs
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Open one side of a check lazily; returns the Dataset and a description."""
    from climval.align.inputs import open_path

    kwargs = _split_open_kwargs(spec, check_kwargs)
    info: dict[str, Any] = spec.to_dict()
    if spec.path is not None:
        ds = open_path(spec.path, kwargs)
    else:
        assert spec.catalog is not None
        cat = catalogs.get(spec.catalog)
        sub = cat.search(**spec.query)
        if len(sub.df) == 0:
            raise InputError(f"The catalogue query {spec.query} matched no dataset")
        storage = kwargs.pop("storage_options", None)
        fmt = getattr(getattr(sub.esmcat.assets, "format", None), "value", None)
        if storage is None and str(fmt).startswith("zarr"):
            # intake-esm passes storage_options={} to zarr by default, which zarr 3
            # rejects for local stores; None means "no options" for both.
            backend = dict(kwargs.get("backend_kwargs", {}))
            backend.setdefault("storage_options", None)
            kwargs["backend_kwargs"] = backend
        open_kw: dict[str, Any] = {"xarray_open_kwargs": kwargs, "progressbar": False}
        if storage is not None:
            open_kw["storage_options"] = storage
        dsets = sub.to_dataset_dict(**open_kw)
        if len(dsets) != 1:
            raise InputError(
                f"The catalogue query {spec.query} matched {len(dsets)} datasets "
                f"({sorted(dsets)}); narrow it to one, for example with source_id, "
                "member_id, grid_label and table_id"
            )
        key, ds = next(iter(dsets.items()))
        info["catalog_key"] = key
        info["catalog_assets"] = [str(z) for z in sub.df[sub.esmcat.assets.column_name]]
        columns = set(sub.df.columns)
        squeezed = [str(d) for d in ds.dims if d in columns and ds.sizes[d] == 1]
        if squeezed:
            ds = ds.isel({d: 0 for d in squeezed})
            info["squeezed"] = squeezed
    if spec.sel:
        try:
            ds = ds.sel(spec.sel)
        except (KeyError, ValueError) as exc:
            raise InputError(f"sel={spec.sel} failed: {exc}") from None
    return ds, info


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _variable_label(variable: str | tuple[str, str] | None) -> str:
    if variable is None:
        return ""
    if isinstance(variable, tuple):
        return f"{variable[0]}/{variable[1]}"
    return variable


def _evaluate(comparison: Comparison, check: CheckSpec) -> ThresholdResult:
    from climval.thresholds import ThresholdResult, evaluate_thresholds

    checks = []
    for item in check.thresholds:
        checks.extend(evaluate_thresholds(comparison, check.threshold_mapping(item)).checks)
    return ThresholdResult(checks=checks)


def run_check(
    check: CheckSpec, catalogs: _Catalogs | None = None, verbose: bool = False
) -> list[CheckOutcome]:
    """Run one check: one outcome per variable."""
    import climval

    catalogs = catalogs or _Catalogs()
    outcomes = []
    options = dict(check.options)
    check_kwargs = options.pop("open_kwargs", {})
    names = options.pop("names", None) or (
        check.reference.name or _short(check.reference),
        check.candidate.name or _short(check.candidate),
    )
    for variable in check.variables:
        start = time.perf_counter()
        outcome = CheckOutcome(
            check=check.name,
            variable=_variable_label(variable),
            status="error",
            reference=check.reference.describe(),
            candidate=check.candidate.describe(),
            description=check.description,
        )
        try:
            ref_ds, ref_info = open_input(check.reference, check_kwargs, catalogs)
            can_ds, can_info = open_input(check.candidate, check_kwargs, catalogs)
            outcome.inputs = {"reference": ref_info, "candidate": can_info}
            comparison = climval.compare(ref_ds, can_ds, variable, names=names, **options)
            for role, info in outcome.inputs.items():
                record = comparison.provenance["inputs"][role]
                record["location"] = info.get("path") or info.get("catalog_assets")
                record["source"] = jsonable(info)
            outcome.comparison = comparison
            outcome.thresholds = _evaluate(comparison, check)
            outcome.status = "pass" if outcome.thresholds.passed else "fail"
        except Exception as exc:  # noqa: BLE001 - every failure becomes an error outcome
            outcome.status = "error"
            outcome.error_type = type(exc).__name__
            outcome.error = _message(exc)
            if verbose:
                outcome.detail = traceback.format_exc()
        outcome.seconds = time.perf_counter() - start
        outcomes.append(outcome)
    return outcomes


def _message(exc: BaseException) -> str:
    """The message plus its causes, so a wrapped error (intake-esm) shows its origin."""
    parts = [" ".join(str(exc).split())]
    seen = {id(exc)}
    cause = exc.__cause__ or exc.__context__
    while cause is not None and id(cause) not in seen and len(parts) < 4:
        seen.add(id(cause))
        parts.append(f"caused by {type(cause).__name__}: {' '.join(str(cause).split())}")
        cause = cause.__cause__ or cause.__context__
    return "; ".join(parts)


def _short(spec: InputSpec) -> str:
    if spec.path is not None:
        return Path(spec.path.rstrip("/")).name if "://" not in spec.path else spec.path
    parts = [str(spec.query[k]) for k in ("source_id", "member_id") if k in spec.query]
    return " ".join(parts) if parts else "catalogue query"


def run(
    config: GateConfig,
    verbose: bool = False,
    progress: Callable[[str], None] | None = None,
) -> GateReport:
    """Run every check of ``config`` and return a :class:`GateReport`."""
    catalogs = _Catalogs()
    outcomes: list[CheckOutcome] = []
    for check in config.checks:
        if progress is not None:
            progress(f"running {check.name}")
        outcomes.extend(run_check(check, catalogs, verbose=verbose))
    return GateReport(outcomes=outcomes, config_path=config.path, config_sha256=config.sha256)


def run_file(path: str | Path, only: list[str] | None = None, verbose: bool = False) -> GateReport:
    """Load a configuration file and run it. A configuration error gives exit code 2."""
    from climval.gate.config import load_config

    try:
        config = load_config(path).select(only or [])
    except ConfigError as exc:
        return GateReport(outcomes=[], config_path=str(path), config_error=str(exc))
    return run(config, verbose=verbose)
