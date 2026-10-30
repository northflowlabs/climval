# SPDX-License-Identifier: Apache-2.0
"""
climval.core.report
===================
Reports: HTML, JSON, Markdown and console output for a
:class:`~climval.Comparison` and for the 0.1 :class:`BenchmarkReport`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from climval.metrics.stats import BaseMetric
from climval.models.schema import BenchmarkResult, ClimateModel

if TYPE_CHECKING:
    from climval.result import Comparison

logger = logging.getLogger(__name__)

_STYLE = """
  :root {
    --bg: #ffffff; --surface: #f5f7f8; --border: #d9dee2; --accent: #0e6b63;
    --text: #1d2327; --muted: #5d6870; --warn: #8a5a00;
    --font: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0d1117; --surface: #161b22; --border: #21262d; --accent: #3fb9a8;
      --text: #c9d1d9; --muted: #8b949e; --warn: #f0b429;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         font-size: 14px; padding: 40px 16px; max-width: 1100px; margin: 0 auto; }
  header { border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }
  h1 { font-size: 22px; color: var(--accent); }
  h2 { font-size: 16px; margin: 32px 0 8px; }
  .meta { color: var(--muted); margin-top: 8px; font-size: 12px; line-height: 1.7; }
  .scroll { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th { text-align: left; padding: 8px 12px; color: var(--muted); font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
  td.value { color: var(--accent); font-weight: 600; text-align: right; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
           background: var(--surface); border: 1px solid var(--border); color: var(--muted); }
  .notice { margin-bottom: 24px; padding: 12px 14px; color: var(--warn);
            border: 1px solid var(--warn); border-radius: 4px; }
  ol { margin: 8px 0 0 20px; line-height: 1.7; }
  pre { background: var(--surface); padding: 12px; overflow-x: auto; font-size: 12px; }
  details { margin-top: 24px; }
  footer { margin-top: 40px; color: var(--muted); font-size: 11px; }
"""

_FOOTER = "climval · Apache-2.0 · Northflow Technologies AS · northflow.no"


def _fmt(value: float, metric: str) -> str:
    if value != value:
        return "NaN"
    return f"{value:.0f}" if metric == "n_points" else f"{value:.4f}"


def _page(title: str, body: str) -> str:
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8"/>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
        f"<title>{escape(title)}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
        f"{body}\n<footer>{_FOOTER}</footer>\n</body>\n</html>"
    )


# ---------------------------------------------------------------------------
# Comparison (0.2)
# ---------------------------------------------------------------------------


def comparison_table_html(c: Comparison) -> str:
    rows = c.rows()
    extra = sorted({k for r in rows for k in r} - {"sample", "group", "metric", "value", "units"})
    head = "".join(
        f"<th>{escape(h)}</th>" for h in ["Sample", "Group", *extra, "Metric", "Value", "Units"]
    )
    body = "".join(
        "<tr>"
        + f"<td>{escape(r['sample'])}</td><td>{escape(r['group'])}</td>"
        + "".join(f"<td>{escape(str(r.get(k, '')))}</td>" for k in extra)
        + f"<td>{escape(r['metric'])}</td>"
        + f"<td class='value'>{_fmt(r['value'], r['metric'])}</td>"
        + f"<td>{escape(r['units'])}</td></tr>\n"
        for r in rows
    )
    return (
        f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>\n{body}</tbody></table></div>"
    )


def render_comparison_html(c: Comparison) -> str:
    from climval.result import SYNTHETIC_NOTICE, describe_steps

    env = c.provenance.get("environment", {})
    samples = "".join(
        f"<li><b>{escape(s)}</b>: {escape(str(ds.attrs.get('description', '')))}</li>"
        for s, ds in c.samples.items()
    )
    steps = "".join(f"<li>{escape(s)}</li>" for s in describe_steps(c.provenance.get("steps", [])))
    notice = f'<p class="notice">{escape(SYNTHETIC_NOTICE)}</p>\n' if c.synthetic else ""
    body = f"""<header>
  <h1>ClimVal: {escape(c.candidate_name)} against {escape(c.reference_name)}</h1>
  <div class="meta">
    Variable <span class="badge">{escape(c.variable)}</span>
    {f'units <span class="badge">{escape(c.units)}</span>' if c.units else ""}<br/>
    Created {escape(str(c.provenance.get("created", "")))} with climval
    {escape(str(env.get("climval", "")))}, Python {escape(str(env.get("python", "")))}
  </div>
</header>
{notice}<h2>Values</h2>
<ul class="meta">{samples}</ul>
{comparison_table_html(c)}
<h2>Alignment</h2>
<ol>{steps}</ol>
<details><summary>Provenance (JSON)</summary>
<pre>{escape(json.dumps(c.provenance, indent=2))}</pre>
</details>"""
    return _page(f"ClimVal: {c.candidate_name}", body)


def render_comparison_markdown(c: Comparison) -> str:
    from climval.result import SYNTHETIC_NOTICE, describe_steps

    rows = c.rows()
    extra = sorted({k for r in rows for k in r} - {"sample", "group", "metric", "value", "units"})
    lines = [
        f"# ClimVal: {c.candidate_name} against {c.reference_name}",
        "",
        f"**Variable:** {c.variable}" + (f" ({c.units})" if c.units else "") + "  ",
        f"**Created:** {c.provenance.get('created', '')}  ",
        f"**climval:** {c.provenance.get('environment', {}).get('climval', '')}",
        "",
    ]
    if c.synthetic:
        lines += [f"> **{SYNTHETIC_NOTICE}**", ""]
    lines += [f"- **{s}**: {ds.attrs.get('description', '')}" for s, ds in c.samples.items()]
    cols = ["Sample", "Group", *extra, "Metric", "Value", "Units"]
    lines += ["", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = [
            r["sample"],
            r["group"],
            *[str(r.get(k, "")) for k in extra],
            r["metric"],
            _fmt(r["value"], r["metric"]),
            r["units"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Alignment", ""]
    lines += [f"1. {s}" for s in describe_steps(c.provenance.get("steps", []))]
    lines += ["", "## Provenance", "", "```json", json.dumps(c.provenance, indent=2), "```", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BenchmarkReport (0.1 compatibility)
# ---------------------------------------------------------------------------


class BenchmarkReport:
    """
    Aggregated output of a BenchmarkSuite run.

    Methods
    -------
    export(path)
        Auto-detect format from extension (.html, .json, .md).
    to_dict()
        Serialize to a plain Python dict.
    summary()
        Print a human-readable table to stdout.
    """

    def __init__(
        self,
        suite_name: str,
        reference: ClimateModel,
        candidates: list[ClimateModel],
        results: list[BenchmarkResult],
        metrics_used: list[BaseMetric],
    ) -> None:
        self.suite_name = suite_name
        self.reference = reference
        self.candidates = candidates
        self.results = results
        self.metrics_used = metrics_used
        self.generated_at: datetime = datetime.now(timezone.utc)

    @property
    def synthetic(self) -> bool:
        """True when any result was computed on synthetic random data."""
        return any(r.synthetic for r in self.results)

    def _synthetic_notice(self) -> str:
        names = ", ".join(r.candidate for r in self.results if r.synthetic)
        return (
            "Synthetic data: scores for "
            f"{names} were computed on random arrays, not on model output. "
            "They are not a validation."
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "generated_at": self.generated_at.isoformat(),
            "reference_model": self.reference.name,
            "candidate_models": [c.name for c in self.candidates],
            "synthetic_data": self.synthetic,
            "results": [
                {
                    "reference": r.reference,
                    "candidate": r.candidate,
                    "computed_at": r.computed_at.isoformat(),
                    "synthetic": r.synthetic,
                    "notes": r.notes,
                    "metrics": [
                        {
                            "metric": m.metric_name,
                            "variable": m.variable,
                            "value": round(m.value, 6) if m.value == m.value else None,
                            "units": m.units,
                            "reference_model": m.reference_model,
                            "candidate_model": m.candidate_model,
                        }
                        for m in r.metrics
                    ],
                    "provenance": {var: comp.provenance for var, comp in r.comparisons.items()},
                }
                for r in self.results
            ],
        }

    def export(self, path: str) -> None:
        """Export to file. Format inferred from extension."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in {".json", ".html", ".md", ".markdown"}:
            raise ValueError(f"Unsupported export format '{ext}'. Use .json, .html, or .md")
        p.parent.mkdir(parents=True, exist_ok=True)
        if ext == ".json":
            text = json.dumps(self.to_dict(), indent=2)
        elif ext == ".html":
            text = self._render_html()
        else:
            text = self._render_markdown()
        p.write_text(text, encoding="utf-8")
        logger.info("Report exported to %s", p)

    def _render_markdown(self) -> str:
        lines = [
            f"# ClimVal Report: {self.suite_name}",
            "",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Reference:** {self.reference.name}  ",
            f"**Candidates:** {', '.join(c.name for c in self.candidates)}",
            "",
        ]
        if self.synthetic:
            lines.extend([f"> **{self._synthetic_notice()}**", ""])
        for result in self.results:
            lines.append(f"## {result.candidate} vs {result.reference}")
            lines.append("")
            lines.append("| Variable | Metric | Value | Units |")
            lines.append("|----------|--------|-------|-------|")
            for m in result.metrics:
                lines.append(
                    f"| {m.variable} | {m.metric_name} | "
                    f"{_fmt(m.value, m.metric_name)} | {m.units} |"
                )
            lines.append("")
        return "\n".join(lines)

    def _render_html(self) -> str:
        rows_html = "".join(
            "<tr>"
            f"<td>{escape(result.candidate)}</td>"
            f"<td>{escape(m.variable)}</td>"
            f"<td>{escape(m.metric_name)}</td>"
            f"<td class='value'>{_fmt(m.value, m.metric_name)}</td>"
            f"<td>{escape(m.units)}</td>"
            "</tr>\n"
            for result in self.results
            for m in result.metrics
        )
        notice_html = (
            f'<p class="notice">{escape(self._synthetic_notice())}</p>\n' if self.synthetic else ""
        )
        badges = "".join(f'<span class="badge">{escape(c.name)}</span> ' for c in self.candidates)
        body = f"""<header>
  <h1>ClimVal: {escape(self.suite_name)}</h1>
  <div class="meta">
    Generated: {self.generated_at.strftime("%Y-%m-%d %H:%M UTC")} &nbsp;·&nbsp;
    Reference: <span class="badge">{escape(self.reference.name)}</span> &nbsp;·&nbsp;
    Candidates: {badges}
  </div>
</header>
{notice_html}<div class="scroll"><table>
  <thead>
    <tr>
      <th>Candidate</th><th>Variable</th><th>Metric</th>
      <th>Value</th><th>Units</th>
    </tr>
  </thead>
  <tbody>
{rows_html}  </tbody>
</table></div>"""
        return _page(f"ClimVal: {self.suite_name}", body)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted summary table to stdout."""
        print(f"\n{'═' * 72}")
        print(f"  CLIMVAL: {self.suite_name.upper()}")
        print(f"  Reference : {self.reference.name}")
        print(f"  Generated : {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'═' * 72}\n")
        if self.synthetic:
            print(f"  WARNING: {self._synthetic_notice()}\n")

        for result in self.results:
            print(f"  ▸ Candidate: {result.candidate}")
            print(f"  {'─' * 66}")
            print(f"  {'Variable':<10} {'Metric':<28} {'Value':>12}  {'Units'}")
            print(f"  {'─' * 66}")
            for m in result.metrics:
                val = f"{m.value:>12.4f}" if m.value == m.value else f"{'NaN':>12}"
                print(f"  {m.variable:<10} {m.metric_name:<28} {val}  {m.units}")
            print()
