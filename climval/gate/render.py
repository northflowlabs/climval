# SPDX-License-Identifier: Apache-2.0
"""Console, Markdown and HTML renderings of a :class:`~climval.gate.run.GateReport`.

The Markdown suits a GitHub job summary (``$GITHUB_STEP_SUMMARY``): one
table of checks, then the failing and passing limits of each check, with the
alignment steps folded away.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from climval.gate.run import CheckOutcome, GateReport
    from climval.thresholds import ThresholdCheck

_STATUS = {"pass": "PASS", "fail": "FAIL", "error": "ERROR"}
_VERDICT = {
    "pass": "all checks passed",
    "fail": "threshold failure",
    "error": "configuration or data error",
}


def limit_text(limits: dict[str, float]) -> str:
    parts = []
    if "min" in limits:
        parts.append(f">= {limits['min']:g}")
    if "max" in limits:
        parts.append(f"<= {limits['max']:g}")
    if "abs_max" in limits:
        parts.append(f"abs <= {limits['abs_max']:g}")
    return ", ".join(parts)


def _where(c: ThresholdCheck) -> str:
    extra = "".join(f", {k}={v}" for k, v in c.point.items())
    return f"{c.sample}, {c.group}{extra}"


def _value(c: ThresholdCheck) -> str:
    return "NaN" if c.value != c.value else f"{c.value:.4g}"


def _units(o: CheckOutcome, metric: str) -> str:
    if o.comparison is None:
        return ""
    for r in o.comparison.rows():
        if r["metric"] == metric:
            return str(r["units"])
    return ""


def _headline(report: GateReport) -> str:
    n = report.counts()
    total = len(report.outcomes)
    return (
        f"{_STATUS[report.status]}: {_VERDICT[report.status]} "
        f"({n['pass']} of {_plural(total, 'check')} passed, {n['fail']} failed, "
        f"{_plural(n['error'], 'error')}; exit code {report.exit_code})"
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def render_text(report: GateReport) -> str:
    bar = "=" * 72
    lines = [
        bar,
        "  climval check" + (f": {report.config_path}" if report.config_path else ""),
        bar,
    ]
    if report.config_error is not None:
        lines += [f"  ERROR  configuration: {report.config_error}", "", f"  {_headline(report)}"]
        return "\n".join(lines)
    for o in report.outcomes:
        lines.append(f"  {_STATUS[o.status]:<5}  {o.label}")
        lines.append(f"         reference: {o.reference}")
        lines.append(f"         candidate: {o.candidate}")
        if o.status == "error":
            lines.append(f"         {o.error_type}: {o.error}")
            if o.detail:
                lines += ["         " + line for line in o.detail.rstrip().splitlines()]
        elif o.thresholds is not None:
            if not o.thresholds.checks:
                lines.append("         no thresholds: the comparison ran and nothing was gated")
            for c in o.thresholds.checks:
                mark = "ok  " if c.passed else "FAIL"
                lines.append(
                    f"         {mark} {c.metric} [{_where(c)}] = {_value(c)} "
                    f"(limit {limit_text(c.limits)})"
                )
        lines.append(f"         {o.seconds:.1f} s")
        lines.append("")
    lines.append(f"  {_headline(report)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


_MD_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
_MD_SPECIAL = frozenset("\\`*[]|")


def _md(text: Any) -> str:
    """Text for Markdown: HTML is escaped and Markdown punctuation is backslash
    escaped, so a check named ``<script>`` renders as text in a job summary."""
    out = []
    for ch in " ".join(str(text).splitlines()):
        if ch in _MD_ESCAPES:
            out.append(_MD_ESCAPES[ch])
        elif ch in _MD_SPECIAL:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _code(text: Any) -> str:
    """An inline code span that no backtick in ``text`` can close early.

    A code span shows its content literally, HTML included, so only the fence
    needs care."""
    s = " ".join(str(text).splitlines())
    longest = run = 0
    for ch in s:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def render_markdown(report: GateReport) -> str:
    from climval.result import describe_steps

    lines = ["## climval check", "", f"**{_headline(report)}**", ""]
    if report.config_path:
        sha = f" (sha256 `{report.config_sha256[:12]}`)" if report.config_sha256 else ""
        lines += [f"Configuration: {_code(report.config_path)}{sha}", ""]
    if report.config_error is not None:
        lines += [f"Configuration error: {_md(report.config_error)}", ""]
        return "\n".join(lines)
    lines += [
        "| Status | Check | Variable | Limits passed | Reference | Candidate |",
        "|---|---|---|---|---|---|",
    ]
    for o in report.outcomes:
        passed = f"{o.n_checks - o.n_failed} of {o.n_checks}" if o.status != "error" else ""
        lines.append(
            f"| **{_STATUS[o.status]}** | {_md(o.check)} | {_md(o.variable)} | {passed} | "
            f"{_md(o.reference)} | {_md(o.candidate)} |"
        )
    for o in report.outcomes:
        lines += ["", f"### {_STATUS[o.status]}: {_md(o.label)}", ""]
        if o.description:
            lines += [_md(o.description), ""]
        if o.status == "error":
            lines += [f"{_md(o.error_type)}: {_md(o.error)}", ""]
            continue
        assert o.thresholds is not None and o.comparison is not None
        c0 = o.comparison
        lines += [
            f"{_md(c0.candidate_name)} against {_md(c0.reference_name)}, variable "
            f"{_code(c0.variable)}" + (f" in {_md(c0.units)}" if c0.units else "") + ".",
            "",
        ]
        if o.thresholds.checks:
            lines += [
                "| Result | Metric | Sample, group | Value | Units | Limit |",
                "|---|---|---|---|---|---|",
            ]
            ordered = sorted(o.thresholds.checks, key=lambda c: c.passed)
            for c in ordered:
                lines.append(
                    f"| {'pass' if c.passed else '**fail**'} | {_md(c.metric)} | "
                    f"{_md(_where(c))} | "
                    f"{_value(c)} | {_md(_units(o, c.metric))} | {limit_text(c.limits)} |"
                )
        else:
            lines.append("No thresholds: the comparison ran and nothing was gated.")
        steps = describe_steps(c0.provenance.get("steps", []))
        lines += ["", "<details><summary>Alignment steps</summary>", ""]
        lines += [f"1. {_md(s)}" for s in steps]
        lines += ["", "</details>"]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_EXTRA_STYLE = """
  .pass { color: var(--accent); font-weight: 600; }
  .fail, .error { color: #b3261e; font-weight: 600; }
  @media (prefers-color-scheme: dark) { .fail, .error { color: #ff8a80; } }
  section { margin-top: 40px; }
"""


def render_html(report: GateReport) -> str:
    from climval.core.report import _STYLE, _page, comparison_table_html
    from climval.result import describe_steps

    summary_rows = "".join(
        f"<tr><td class='{o.status}'>{_STATUS[o.status]}</td><td>{escape(o.check)}</td>"
        f"<td>{escape(o.variable)}</td>"
        f"<td>{'' if o.status == 'error' else f'{o.n_checks - o.n_failed} of {o.n_checks}'}</td>"
        f"<td>{escape(o.reference)}</td><td>{escape(o.candidate)}</td></tr>"
        for o in report.outcomes
    )
    sections = []
    for o in report.outcomes:
        parts = [f"<section><h2 class='{o.status}'>{_STATUS[o.status]}: {escape(o.label)}</h2>"]
        if o.description:
            parts.append(f"<p class='meta'>{escape(o.description)}</p>")
        if o.status == "error":
            parts.append(f"<pre>{escape(str(o.error_type))}: {escape(str(o.error))}</pre>")
            if o.detail:
                parts.append(
                    f"<details><summary>Traceback</summary><pre>{escape(o.detail)}</pre></details>"
                )
            parts.append("</section>")
            sections.append("".join(parts))
            continue
        assert o.thresholds is not None and o.comparison is not None
        rows = "".join(
            f"<tr><td class='{'pass' if c.passed else 'fail'}'>{'pass' if c.passed else 'fail'}"
            f"</td><td>{escape(c.metric)}</td><td>{escape(_where(c))}</td>"
            f"<td class='value'>{escape(_value(c))}</td><td>{escape(_units(o, c.metric))}</td>"
            f"<td>{escape(limit_text(c.limits))}</td></tr>"
            for c in sorted(o.thresholds.checks, key=lambda c: c.passed)
        )
        if rows:
            parts.append(
                "<h3>Limits</h3><div class='scroll'><table><thead><tr><th>Result</th>"
                "<th>Metric</th><th>Sample, group</th><th>Value</th><th>Units</th>"
                f"<th>Limit</th></tr></thead><tbody>{rows}</tbody></table></div>"
            )
        else:
            parts.append("<p class='meta'>No thresholds: nothing was gated.</p>")
        steps = "".join(
            f"<li>{escape(s)}</li>" for s in describe_steps(o.comparison.provenance["steps"])
        )
        parts.append(
            "<details><summary>All values</summary>"
            f"{comparison_table_html(o.comparison)}</details>"
            f"<details><summary>Alignment steps</summary><ol>{steps}</ol></details></section>"
        )
        sections.append("".join(parts))
    config = (
        f"Configuration {escape(report.config_path)}"
        + (f", sha256 {escape(report.config_sha256)}" if report.config_sha256 else "")
        if report.config_path
        else ""
    )
    error = f"<pre>{escape(report.config_error)}</pre>" if report.config_error is not None else ""
    body = f"""<header>
  <h1>climval check</h1>
  <div class="meta"><span class="{report.status}">{escape(_headline(report))}</span><br/>
  {config}<br/>Created {escape(report.created)}</div>
</header>
{error}
<div class="scroll"><table><thead><tr><th>Status</th><th>Check</th><th>Variable</th>
<th>Limits passed</th><th>Reference</th><th>Candidate</th></tr></thead>
<tbody>{summary_rows}</tbody></table></div>
{"".join(sections)}"""
    html = _page("climval check", body)
    return html.replace(f"<style>{_STYLE}</style>", f"<style>{_STYLE}{_EXTRA_STYLE}</style>")
