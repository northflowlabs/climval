# SPDX-License-Identifier: Apache-2.0
"""Regression tests from the second release review of the gate, the CLI and thresholds."""

from __future__ import annotations

import importlib
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

import climval
from climval.cli import main
from climval.errors import ConfigError
from climval.gate import EXIT_ERROR, EXIT_PASS, load_config, parse_config, run, run_file

DATA = Path(__file__).parent / "data" / "gate"


@pytest.fixture
def gate_dir(tmp_path: Path) -> Path:
    for name in ("reference.nc", "candidate.nc", "pass.yml", "fail.yml"):
        shutil.copy(DATA / name, tmp_path / name)
    return tmp_path


@pytest.fixture(scope="module")
def comparison() -> climval.Comparison:
    return climval.compare(
        str(DATA / "reference.nc"), str(DATA / "candidate.nc"), ("t2m", "tas"), by="season"
    )


def _check(**extra: Any) -> dict[str, Any]:
    check = {
        "name": "c",
        "reference": "reference.nc",
        "candidate": "candidate.nc",
        "variable": {"reference": "t2m", "candidate": "tas"},
        "thresholds": {"rmse": {"max": 100}},
        **extra,
    }
    return {"version": 1, "checks": [check]}


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


_YAML_CHECK = """checks:
  - name: a
    reference: reference.nc
    candidate: candidate.nc
    variable: {{reference: t2m, candidate: tas}}
{extra}"""


# ---------------------------------------------------------------------------
# M2: limits that cannot be checked are rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max": math.nan}, "finite"),
        ({"max": math.inf}, "finite"),
        ({"min": -math.inf}, "finite"),
        ({"min": 5, "max": 1}, "min 5 is above max 1"),
        ({"abs_max": -0.5}, "negative"),
        ({"max": True}, "must be a number"),
        ({"max": "two"}, "must be a number"),
        ({"max": "1e400"}, "finite"),
    ],
)
def test_bad_limits_are_configuration_errors(limits: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_check(thresholds={"rmse": limits}))


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max": math.nan}, "finite"),
        ({"max": math.inf}, "finite"),
        ({"min": 0.9, "max": 0.1}, "above max"),
        ({"abs_max": -1}, "negative"),
        ({"max": True}, "must be a number"),
        ({"max": "2"}, "must be a number"),
    ],
)
def test_bad_limits_are_rejected_in_python(
    comparison: climval.Comparison, limits: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        comparison.check({"rmse": limits})
    with pytest.raises(ValueError, match=message):
        climval.evaluate_thresholds(comparison, {"rmse": limits})


def test_equal_min_and_max_and_numpy_limits_are_allowed(comparison: climval.Comparison) -> None:
    import numpy as np

    rmse = comparison["rmse"]
    assert comparison.check({"rmse": {"min": rmse, "max": rmse}}).passed
    assert comparison.check({"rmse": {"max": np.float32(100), "min": np.int64(0)}}).passed


@pytest.mark.parametrize(
    ("thresholds", "message"),
    [
        ("{rmse: {max: .nan}}", "finite"),
        ("{rmse: {max: .inf}}", "finite"),
        ("{rmse: {min: 5, max: 1}}", "above max"),
        ("{rmse: {max: 1e400}}", "finite"),
    ],
)
def test_bad_limits_in_yaml_exit_2(
    gate_dir: Path, thresholds: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write(gate_dir, "c.yml", _YAML_CHECK.format(extra=f"    thresholds: {thresholds}\n"))
    assert main(["check", str(cfg)]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "configuration" in out and message in out


def test_yaml_exponent_without_point_is_a_number(gate_dir: Path) -> None:
    # PyYAML (YAML 1.1) reads 1e-3 and 1e2 as strings
    cfg = _write(
        gate_dir,
        "c.yml",
        _YAML_CHECK.format(extra="    thresholds: {rmse: {max: 1e2}, crmse: {min: 1e-3}}\n"),
    )
    config = load_config(cfg)
    assert config.checks[0].thresholds[0]["max"] == 100.0
    assert config.checks[0].thresholds[1]["min"] == 0.001
    assert run(config).exit_code == EXIT_PASS


# ---------------------------------------------------------------------------
# M3: exit code 1 means a threshold failure, nothing else
# ---------------------------------------------------------------------------


def test_unexpected_error_in_check_exits_2(
    gate_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = gate_dir / "is-a-directory"
    out_dir.mkdir()
    code = main(["check", str(gate_dir / "pass.yml"), "--quiet", "--json", str(out_dir)])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "climval check: error:" in err


def test_unexpected_error_after_a_threshold_failure_exits_2(
    gate_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from climval.gate.run import GateReport

    def boom(self: GateReport) -> str:
        raise RuntimeError("renderer broke")

    monkeypatch.setattr(GateReport, "to_markdown", boom)
    assert main(["check", str(gate_dir / "fail.yml"), "--quiet"]) == 1
    md = str(gate_dir / "s.md")
    assert main(["check", str(gate_dir / "fail.yml"), "--quiet", "--markdown", md]) == EXIT_ERROR


# ---------------------------------------------------------------------------
# m9: inputs shared by checks are opened once per run
# ---------------------------------------------------------------------------


def test_shared_inputs_are_opened_once(gate_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = importlib.import_module("climval.align.inputs")

    calls: list[str] = []
    original = inputs.open_path

    def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(inputs, "open_path", counting)
    report = run_file(gate_dir / "pass.yml")
    assert report.exit_code == EXIT_PASS
    assert len(report.outcomes) == 2
    # two checks on the same two files: each file opened once
    assert sorted(Path(c).name for c in calls) == ["candidate.nc", "reference.nc"]

    # and the second check still gives the numbers of a direct call
    north = report.outcomes[1].comparison
    assert north is not None
    direct = climval.compare(
        str(gate_dir / "reference.nc"),
        str(gate_dir / "candidate.nc"),
        ("t2m", "tas"),
        region=climval.Box(lat=(55, 70), lon=(0, 30)),
        period=("2001", "2002"),
    )
    assert north["crmse"] == pytest.approx(direct["crmse"], rel=1e-12)


def test_different_sel_opens_separately(gate_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from climval.gate.config import InputSpec
    from climval.gate.run import _Catalogs

    opened: list[Any] = []
    gate_run = importlib.import_module("climval.gate.run")

    original = gate_run.open_input

    def counting(spec: InputSpec, kwargs: Any, catalogs: Any) -> Any:
        opened.append(spec.sel)
        return original(spec, kwargs, catalogs)

    monkeypatch.setattr(gate_run, "open_input", counting)
    cache = _Catalogs()
    path = str(gate_dir / "candidate.nc")
    cache.open(InputSpec(path=path), {})
    cache.open(InputSpec(path=path), {})
    cache.open(InputSpec(path=path, sel={"lat": 50.0}, open_kwargs={}), {})
    cache.open(InputSpec(path=path), {"chunks": {"time": 12}})
    assert len(opened) == 3


# ---------------------------------------------------------------------------
# m10: groups, empty checks and version validated before data are read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("by", "group", "message"),
    [
        ("season", "XYZ", "not a group of by='season'"),
        ("season", "djf", "not a group of by='season'"),
        ("season", "1", "not a group of by='season'"),
        ("month", 1, "not a group of by='month'"),
        ("month", "January", "not a group of by='month'"),
        ("year", "DJF", "not a group of by='year'"),
        ("none", "DJF", "needs 'by'"),
        ("season", True, "group name"),
    ],
)
def test_unknown_groups_are_configuration_errors(by: str, group: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_check(by=by, thresholds={"rmse": {"max": 100, "group": group}}))


@pytest.mark.parametrize(
    ("by", "group"),
    [("season", "JJA"), ("month", "Jan"), ("year", 2001), ("year", "2001"), ("season", "*")],
)
def test_known_groups_are_accepted(by: str, group: Any) -> None:
    config = parse_config(_check(by=by, thresholds={"rmse": {"max": 100, "group": group}}))
    assert config.checks[0].thresholds[0]["group"] == str(group)


def test_bad_group_file_fails_before_reading_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # no data files at all: the error must come from validation
    cfg = _write(
        tmp_path,
        "c.yml",
        _YAML_CHECK.format(extra="    by: season\n    thresholds: {rmse: {max: 5, group: XYZ}}\n"),
    )
    assert main(["check", str(cfg), "--validate"]) == EXIT_ERROR
    assert "XYZ" in capsys.readouterr().out


def test_no_thresholds_exits_2(gate_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write(gate_dir, "c.yml", _YAML_CHECK.format(extra=""))
    assert main(["check", str(cfg)]) == EXIT_ERROR
    assert "at least one threshold" in capsys.readouterr().out


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_version_must_be_the_integer_1(version: Any) -> None:
    with pytest.raises(ConfigError, match="version"):
        parse_config({**_check(), "version": version})


@pytest.mark.parametrize(
    "region", [[True, 60], {"lat": [False, 60]}, [math.nan, 60], {"lat": [35, math.inf]}]
)
def test_region_numbers_reject_booleans_and_non_finite(region: Any) -> None:
    with pytest.raises(ConfigError, match="region"):
        parse_config(_check(region=region))


# ---------------------------------------------------------------------------
# m10: Markdown escapes HTML; plural forms
# ---------------------------------------------------------------------------

_INJECT = """checks:
  - name: '<script>alert(1)</script>|x'
    description: '<img src=x onerror=alert(2)> | pipe'
    reference: reference.nc
    candidate: {candidate}
    variable: {{reference: t2m, candidate: tas}}
    names: ['<i>ref</i>', '<i>can</i>|y']
    thresholds: {{rmse: {{max: 5}}}}
"""


@pytest.mark.parametrize("candidate", ["candidate.nc", "'<b>missing</b>.nc'"])
def test_markdown_escapes_html(gate_dir: Path, candidate: str) -> None:
    cfg = _write(gate_dir, "inject.yml", _INJECT.format(candidate=candidate))
    report = run_file(cfg)
    md = report.to_markdown()
    for tag in ("<script>", "<img", "<i>", "<b>"):
        assert tag not in md
    assert "&lt;script&gt;alert(1)&lt;/script&gt;\\|x" in md
    assert "&lt;img src=x onerror=alert(2)&gt; \\| pipe" in md
    html = report.to_html()
    assert "<script>" not in html and "<img" not in html


def test_markdown_code_spans_survive_backticks(gate_dir: Path) -> None:
    from climval.gate.render import _code

    assert _code("a`b") == "``a`b``"
    assert _code("`x") == "`` `x ``"
    report = run_file(gate_dir / "pass.yml")
    assert f"Configuration: `{gate_dir / 'pass.yml'}`" in report.to_markdown()


def test_counts_are_pluralised(gate_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = run(load_config(gate_dir / "fail.yml"))
    for text in (report.summary_text(), report.to_markdown(), report.to_html()):
        assert "0 of 1 check passed, 1 failed, 0 errors" in text
        assert "1 checks" not in text
    assert report.outcomes[0].thresholds is not None
    assert "of 2 checks passed" in report.outcomes[0].thresholds.summary()
    assert main(["check", str(gate_dir / "fail.yml"), "--validate", "--quiet"]) == EXIT_PASS
    assert "(1 check: europe-strict)" in capsys.readouterr().out
    two = run(load_config(gate_dir / "pass.yml"))
    assert "2 of 2 checks passed, 0 failed, 0 errors" in two.summary_text()
