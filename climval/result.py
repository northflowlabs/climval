"""The :class:`Comparison` result returned by :func:`climval.compare`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from climval.align import AlignedPair
from climval.metrics.engine import SAMPLES, SCALAR_SAMPLES, metric_units, scalar_rows
from climval.metrics.stats import BaseMetric
from climval.provenance import hash_values, jsonable, strided_sample

if TYPE_CHECKING:
    import pandas as pd

    from climval.thresholds import ThresholdResult

SYNTHETIC_NOTICE = (
    "Synthetic data: these scores were computed on generated example fields, "
    "not on model output. They are not a validation."
)


class Comparison:
    """Metric values for one candidate against one reference, with provenance.

    Attributes
    ----------
    reference_name, candidate_name : str
    variable : str
        The reference's variable name.
    units : str or None
        Units of the aligned data (the reference's units).
    metrics : list of BaseMetric
    samples : dict of str to xarray.Dataset
        One Dataset per sample ("space_time", "climatology", "space",
        "time"), a variable per metric plus ``n_points``, and a ``group``
        dimension ("all" plus one entry per season, month or year).
    aligned : AlignedPair
        The aligned reference and candidate, for inspection.
    provenance : dict
        JSON-serialisable record of inputs, alignment steps and software.
    synthetic : bool
        True only for :func:`climval.demo` results.
    """

    def __init__(
        self,
        *,
        reference_name: str,
        candidate_name: str,
        aligned: AlignedPair,
        metrics: list[BaseMetric],
        samples: dict[str, xr.Dataset],
        provenance: dict[str, Any],
        by: str | None,
        synthetic: bool = False,
    ) -> None:
        self.reference_name = reference_name
        self.candidate_name = candidate_name
        self.aligned = aligned
        self.variable = aligned.variable
        self.units = aligned.units
        self.metrics = metrics
        self.samples = samples
        self.provenance = provenance
        self.by = by
        self.synthetic = synthetic
        self._computed = False

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @property
    def computed(self) -> bool:
        return self._computed

    def compute(self) -> Comparison:
        """Compute every metric and the aligned-value hashes in one ``dask.compute`` call.

        dask shares the tasks that read and align the inputs between all
        samples and metrics, so each input chunk is read once. NumPy inputs
        were aligned in memory already.
        """
        if self._computed:
            return self
        names = list(self.samples)
        hashes = {
            "reference": strided_sample(self.aligned.reference),
            "candidate": strided_sample(self.aligned.candidate),
        }
        lazy: list[Any] = [self.samples[n] for n in names] + list(hashes.values())
        import dask

        done = dask.compute(*lazy)  # type: ignore[attr-defined,no-untyped-call]
        self.samples = dict(zip(names, done[: len(names)], strict=False))
        ref_s, can_s = done[len(names) :]
        self.provenance["aligned_values_sha256"] = {
            "reference": hash_values(ref_s.values),
            "candidate": hash_values(can_s.values),
            "basis": (
                "strided sample of the aligned fields, up to 16 indices per "
                "dimension, rounded to 36 mantissa bits"
            ),
        }
        self._note_undefined_percentiles()
        self._computed = True
        return self

    def _note_undefined_percentiles(self) -> None:
        """Record percentile biases left NaN because the reference quantile is near zero."""
        found: dict[str, dict[str, list[str]]] = {}
        for sample, ds in self.samples.items():
            if "n_points" not in ds:
                continue
            for name in ds.data_vars:
                if not str(name).startswith("percentile_bias"):
                    continue
                undefined = ds[name].isnull() & (ds["n_points"] > 0)
                if "group" not in undefined.dims or not bool(undefined.any()):
                    continue
                groups = [
                    str(g) for g in undefined["group"].values if bool(undefined.sel(group=g).any())
                ]
                found.setdefault(sample, {})[str(name)] = groups
        if found:
            notes = self.provenance.setdefault("metric_notes", {})
            notes["percentile_bias_undefined"] = {
                "rule": (
                    "NaN where the reference percentile is below 1% of the reference's "
                    "weighted mean absolute value (min_relative), for example the 5th "
                    "percentile of daily precipitation, which is near zero"
                ),
                "groups": found,
            }

    def _require_computed(self) -> None:
        if not self._computed:
            self.compute()

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def reference(self) -> xr.DataArray:
        """The aligned reference field."""
        return self.aligned.reference

    @property
    def candidate(self) -> xr.DataArray:
        """The aligned candidate field (on the target grid, in the reference's units)."""
        return self.aligned.candidate

    @property
    def groups(self) -> list[str]:
        first = next(iter(self.samples.values()), None)
        return [str(g) for g in first["group"].values] if first is not None else []

    def value(self, metric: str, sample: str = "space_time", group: str = "all") -> float:
        """One scalar value. Raises when the value has extra dimensions."""
        self._require_computed()
        da = self.samples[sample][metric].sel(group=group)
        if da.ndim:
            raise ValueError(
                f"{metric} has dimensions {da.dims}; use .field({sample!r}) or .to_dataframe()"
            )
        return float(da)

    def values(self, sample: str = "space_time", group: str = "all") -> dict[str, float]:
        """All scalar metric values of one sample and group."""
        return {m.name: self.value(m.name, sample, group) for m in self.metrics}

    def __getitem__(self, metric: str) -> float:
        return self.value(metric)

    def field(self, sample: str) -> xr.Dataset:
        """The full Dataset of one sample (time series for 'space', maps for 'time')."""
        self._require_computed()
        return self.samples[sample]

    def rows(self) -> list[dict[str, Any]]:
        """Long-format rows for the scalar samples."""
        self._require_computed()
        return scalar_rows(self.samples, self.metrics, self.units)

    def to_dataframe(self) -> pd.DataFrame:
        """Long table: sample, group, metric, value, units (plus any extra dims)."""
        import pandas as pd

        return pd.DataFrame(self.rows())

    def check(self, thresholds: Mapping[str, Mapping[str, Any]]) -> ThresholdResult:
        """Evaluate limits; see :func:`climval.evaluate_thresholds`."""
        from climval.thresholds import evaluate_thresholds

        return evaluate_thresholds(self, thresholds)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        self._require_computed()
        return jsonable(  # type: ignore[no-any-return]
            {
                "reference": self.reference_name,
                "candidate": self.candidate_name,
                "variable": self.variable,
                "units": self.units,
                "synthetic_data": self.synthetic,
                "by": self.by,
                "samples": {s: SAMPLES[s] for s in self.samples},
                "metrics": [
                    {**m.summary(), "units": metric_units(m, self.units)} for m in self.metrics
                ],
                "values": self.rows(),
                "provenance": self.provenance,
            }
        )

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    def export(self, path: str | Path) -> None:
        """Write .json, .html or .md, chosen by extension."""
        from climval.core.report import render_comparison_html, render_comparison_markdown

        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".json":
            self.to_json(p)
            return
        if ext == ".html":
            text = render_comparison_html(self)
        elif ext in (".md", ".markdown"):
            text = render_comparison_markdown(self)
        else:
            raise ValueError(f"Unsupported export format '{ext}'. Use .json, .html, or .md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def table(self) -> str:
        """Plain-text table of the scalar values."""
        rows = self.rows()
        extra = sorted(
            {k for r in rows for k in r} - {"sample", "group", "metric", "value", "units"}
        )
        head = ["sample", "group", *extra, "metric", "value", "units"]
        lines = [head]
        for r in rows:
            if r["metric"] == "n_points" and r["group"] != "all":
                continue
            v = r["value"]
            lines.append(
                [
                    r["sample"],
                    r["group"],
                    *[str(r.get(k, "")) for k in extra],
                    r["metric"],
                    "NaN" if v != v else f"{v:.4f}" if r["metric"] != "n_points" else f"{v:.0f}",
                    r["units"],
                ]
            )
        widths = [max(len(str(row[i])) for row in lines) for i in range(len(head))]
        return "\n".join(
            "  ".join(
                str(c).rjust(w) if i == len(head) - 2 else str(c).ljust(w)
                for i, (c, w) in enumerate(zip(row, widths, strict=False))
            )
            for row in lines
        )

    def summary(self) -> None:
        """Print the values and the alignment steps."""
        print(self.summary_text())

    def summary_text(self) -> str:
        bar = "=" * 72
        lines = [
            bar,
            f"  climval: {self.candidate_name} against {self.reference_name}",
            f"  variable {self.variable}" + (f" [{self.units}]" if self.units else ""),
            bar,
        ]
        if self.synthetic:
            lines += [f"  WARNING: {SYNTHETIC_NOTICE}", ""]
        lines += ["  " + line for line in self.table().splitlines()]
        lines += ["", "  Alignment:"]
        lines += [f"    {s}" for s in describe_steps(self.provenance.get("steps", []))]
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "computed" if self._computed else "lazy, call .compute()"
        return (
            f"<climval.Comparison {self.candidate_name} against {self.reference_name}, "
            f"variable {self.variable}, samples {list(self.samples)}, groups "
            f"{self.groups}, {state}>"
        )

    def _repr_html_(self) -> str:
        from climval.core.report import comparison_table_html

        return comparison_table_html(self)


def describe_steps(steps: list[dict[str, Any]]) -> list[str]:
    """One readable line per alignment step."""
    out = []
    for s in steps:
        name = s.get("step", "?")
        detail = {k: v for k, v in s.items() if k != "step"}
        if "action" in detail and len(detail) == 1:
            out.append(f"{name}: {detail['action']}")
            continue
        parts = []
        for k, v in detail.items():
            if isinstance(v, dict) and len(json.dumps(jsonable(v))) > 120:
                v = "{...}"
            parts.append(f"{k}={json.dumps(jsonable(v))}" if not isinstance(v, str) else f"{k}={v}")
        out.append(f"{name}: " + ", ".join(parts))
    return out


__all__ = ["Comparison", "SYNTHETIC_NOTICE", "SCALAR_SAMPLES", "describe_steps"]
