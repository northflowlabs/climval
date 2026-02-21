"""
climval.core.report
==============================
BenchmarkReport — structured output with HTML, JSON, and console export.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from climval.metrics.stats import BaseMetric
from climval.models.schema import BenchmarkResult, ClimateModel

logger = logging.getLogger(__name__)


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
        self.generated_at: datetime = datetime.utcnow()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "generated_at": self.generated_at.isoformat(),
            "reference_model": self.reference.name,
            "candidate_models": [c.name for c in self.candidates],
            "results": [
                {
                    "reference": r.reference,
                    "candidate": r.candidate,
                    "computed_at": r.computed_at.isoformat(),
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
                }
                for r in self.results
            ],
        }

    def export(self, path: str) -> None:
        """Export to file. Format inferred from extension."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower()

        if ext == ".json":
            self._export_json(p)
        elif ext == ".html":
            self._export_html(p)
        elif ext in {".md", ".markdown"}:
            self._export_markdown(p)
        else:
            raise ValueError(
                f"Unsupported export format '{ext}'. Use .json, .html, or .md"
            )

        logger.info("Report exported to %s", p)

    def _export_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def _export_markdown(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._render_markdown())

    def _render_markdown(self) -> str:
        lines = [
            f"# ClimVal Report — {self.suite_name}",
            "",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Reference:** {self.reference.name}  ",
            f"**Candidates:** {', '.join(c.name for c in self.candidates)}",
            "",
        ]
        for result in self.results:
            lines.append(f"## {result.candidate} vs {result.reference}")
            lines.append("")
            lines.append("| Variable | Metric | Value | Units |")
            lines.append("|----------|--------|-------|-------|")
            for m in result.metrics:
                val = f"{m.value:.4f}" if m.value == m.value else "NaN"
                lines.append(f"| {m.variable} | {m.metric_name} | {val} | {m.units} |")
            lines.append("")
        return "\n".join(lines)

    def _export_html(self, path: Path) -> None:
        html = self._render_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def _render_html(self) -> str:
        rows_html = ""
        for result in self.results:
            for m in result.metrics:
                val = f"{m.value:.4f}" if m.value == m.value else "NaN"
                rows_html += (
                    f"<tr>"
                    f"<td>{result.candidate}</td>"
                    f"<td>{m.variable}</td>"
                    f"<td>{m.metric_name}</td>"
                    f"<td class='value'>{val}</td>"
                    f"<td>{m.units}</td>"
                    f"</tr>\n"
                )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ClimVal — {self.suite_name}</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #21262d;
    --accent: #388bfd;
    --accent2: #3fb950;
    --text: #c9d1d9;
    --muted: #8b949e;
    --font: 'JetBrains Mono', 'Fira Code', monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font);
          font-size: 14px; padding: 40px; }}
  header {{ border-bottom: 1px solid var(--border); padding-bottom: 24px;
            margin-bottom: 32px; }}
  h1 {{ font-size: 22px; color: var(--accent); letter-spacing: 0.5px; }}
  .meta {{ color: var(--muted); margin-top: 8px; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
  th {{ text-align: left; padding: 10px 14px; color: var(--muted);
        font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
        border-bottom: 1px solid var(--border); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border);
        font-size: 13px; }}
  td.value {{ color: var(--accent2); font-weight: 600; }}
  tr:hover td {{ background: var(--surface); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; background: var(--surface);
            border: 1px solid var(--border); color: var(--muted); }}
  footer {{ margin-top: 40px; color: var(--muted); font-size: 11px; }}
</style>
</head>
<body>
<header>
  <h1>⬡ ClimVal — {self.suite_name}</h1>
  <div class="meta">
    Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
    Reference: <span class="badge">{self.reference.name}</span> &nbsp;·&nbsp;
    Candidates: {' '.join(f'<span class="badge">{c.name}</span>' for c in self.candidates)}
  </div>
</header>
<table>
  <thead>
    <tr>
      <th>Candidate</th><th>Variable</th><th>Metric</th>
      <th>Value</th><th>Units</th>
    </tr>
  </thead>
  <tbody>
{rows_html}  </tbody>
</table>
<footer>climval · open source · Apache-2.0 · northflow.tech</footer>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted summary table to stdout."""
        print(f"\n{'═' * 72}")
        print(f"  CLIMVAL — {self.suite_name.upper()}")
        print(f"  Reference : {self.reference.name}")
        print(f"  Generated : {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'═' * 72}\n")

        for result in self.results:
            print(f"  ▸ Candidate: {result.candidate}")
            print(f"  {'─' * 66}")
            print(f"  {'Variable':<10} {'Metric':<28} {'Value':>12}  {'Units'}")
            print(f"  {'─' * 66}")
            for m in result.metrics:
                val = f"{m.value:>12.4f}" if m.value == m.value else f"{'NaN':>12}"
                print(f"  {m.variable:<10} {m.metric_name:<28} {val}  {m.units}")
            print()
