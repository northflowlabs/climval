"""climval check: configuration, runs, exit codes, outputs and catalogue inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import xarray as xr

import climval
from climval.cli import main
from climval.errors import ConfigError
from climval.gate import (
    EXIT_ERROR,
    EXIT_FAIL,
    EXIT_PASS,
    load_config,
    parse_config,
    run,
    run_file,
)

DATA = Path(__file__).parent / "data" / "gate"


@pytest.fixture
def gate_dir(tmp_path: Path) -> Path:
    """A copy of the committed gate data, so tests can add files next to it."""
    for name in ("reference.nc", "candidate.nc", "pass.yml", "fail.yml", "error.yml", "pass.toml"):
        shutil.copy(DATA / name, tmp_path / name)
    return tmp_path


def _base(**extra: Any) -> dict[str, Any]:
    check = {
        "name": "c",
        "reference": "reference.nc",
        "candidate": "candidate.nc",
        "variable": {"reference": "t2m", "candidate": "tas"},
        **extra,
    }
    return {"version": 1, "checks": [check]}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_load_yaml_and_toml_agree(gate_dir: Path) -> None:
    y = load_config(gate_dir / "pass.yml")
    t = load_config(gate_dir / "pass.toml")
    assert [c.name for c in y.checks] == ["europe", "north"]
    assert y.checks[0].thresholds == t.checks[0].thresholds
    assert y.checks[0].options["region"] == t.checks[0].options["region"]
    assert y.sha256 and len(y.sha256) == 64


def test_defaults_merge_and_relative_paths(gate_dir: Path) -> None:
    cfg = load_config(gate_dir / "pass.yml")
    north = cfg.checks[1]
    assert north.variables == [("t2m", "tas")]
    assert north.options["names"] == ("synthetic reference", "synthetic model")
    assert north.options["period"] == ("2001", "2002")
    assert Path(str(north.reference.path)).is_absolute()
    assert north.reference.describe() == "reference.nc"
    assert north.options["region"] == climval.Box(lat=(55.0, 70.0), lon=(0.0, 30.0))


def test_threshold_forms_normalise() -> None:
    cfg = parse_config(
        _base(
            by="season",
            thresholds={
                "rmse": [{"max": 2}, {"max": 1, "sample": "climatology"}],
                "bias": {"abs_max": 1, "group": "DJF"},
            },
        )
    )
    th = cfg.checks[0].thresholds
    assert th[0] == {"metric": "rmse", "sample": "space_time", "max": 2}
    assert th[1]["sample"] == "climatology"
    # aliases resolve to the metric's name
    assert th[2] == {"metric": "mean_bias", "sample": "space_time", "abs_max": 1, "group": "DJF"}
    listed = parse_config(_base(thresholds=[{"metric": "pearson_r", "min": 0.9}]))
    assert listed.checks[0].thresholds[0]["metric"] == "pearson_r"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"regrid": "cubic"}, "regrid"),
        ({"colour": "red"}, "unknown keys"),
        ({"region": [10, 20, 30]}, "region list"),
        ({"region": {"lat": [60, 50]}}, "south, north"),
        ({"period": "1995"}, "period"),
        ({"freq": "weekly"}, "freq"),
        ({"metrics": ["rmse", "nonsense"]}, "Unknown metric"),
        ({"samples": ["everything"]}, "unknown samples"),
        ({"thresholds": {"rmse": {"below": 2}}}, "unknown keys"),
        ({"thresholds": {"rmse": {"sample": "space_time"}}}, "at least one"),
        ({"thresholds": {"rmse": {"max": "two"}}}, "must be a number"),
        ({"thresholds": {"rmse": {"max": 2, "sample": "space"}}}, "thresholds apply"),
        ({"thresholds": {"rmse": {"max": 2, "group": "DJF"}}}, "needs 'by'"),
        ({"metrics": ["rmse"], "thresholds": {"mae": {"max": 2}}}, "not in this check"),
        (
            {
                "samples": ["space_time"],
                "thresholds": {"rmse": {"max": 2, "sample": "climatology"}},
            },
            "not in this check's samples",
        ),
        ({"variables": ["tas"]}, "not both"),
        ({"variable": ["t2m", "tas"]}, "variable mapping|a variable is"),
        ({"units": "kelvin"}, "convert"),
        ({"names": ["only one"]}, "names"),
    ],
)
def test_invalid_check_options(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_base(**change))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "mapping"),
        ({"checks": []}, "non-empty"),
        ({"version": 2, "checks": [{}]}, "version"),
        ({"checks": [{"name": "a"}]}, "'reference' is required"),
        (
            {"checks": [{"reference": "a.nc", "candidate": {"path": "b", "catalog": "c"}}]},
            "exactly one",
        ),
        ({"checks": [{"reference": "a.nc", "candidate": {"catalog": "c.json"}}]}, "query"),
        (
            {
                "checks": [
                    {"reference": "a.nc", "candidate": "b.nc", "name": "x"},
                    {"reference": "a.nc", "candidate": "b.nc", "name": "x"},
                ]
            },
            "duplicate",
        ),
        (
            {"defaults": {"name": "x"}, "checks": [{"reference": "a", "candidate": "b"}]},
            "unknown keys",
        ),
    ],
)
def test_invalid_files(data: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yml")
    bad = tmp_path / "bad.yml"
    bad.write_text("checks: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(bad)
    other = tmp_path / "c.ini"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match=".yml"):
        load_config(other)
    js = tmp_path / "c.json"
    js.write_text(json.dumps(_base()), encoding="utf-8")
    assert load_config(js).checks[0].name == "c"


def test_select(gate_dir: Path) -> None:
    cfg = load_config(gate_dir / "pass.yml")
    assert [c.name for c in cfg.select(["north"]).checks] == ["north"]
    with pytest.raises(ConfigError, match="No check named"):
        cfg.select(["south"])


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_run_pass_fail_error(gate_dir: Path) -> None:
    passed = run_file(gate_dir / "pass.yml")
    assert passed.exit_code == EXIT_PASS and passed.passed
    assert [o.status for o in passed.outcomes] == ["pass", "pass"]
    europe = passed.outcomes[0]
    # one row for "all" and one per season for the group="*" limit
    assert sum(c.metric == "mean_bias" for c in europe.thresholds.checks) == 5  # type: ignore[union-attr]
    # the same numbers as calling the engine directly
    direct = climval.compare(
        str(gate_dir / "reference.nc"),
        str(gate_dir / "candidate.nc"),
        ("t2m", "tas"),
        region=climval.Box(lat=(35, 70), lon=(-10, 40)),
        by="season",
    )
    assert europe.comparison is not None
    assert europe.comparison["rmse"] == pytest.approx(direct["rmse"], rel=1e-12)

    failed = run_file(gate_dir / "fail.yml")
    assert failed.exit_code == EXIT_FAIL
    assert [c.metric for c in failed.outcomes[0].thresholds.failures] == ["mean_bias"]  # type: ignore[union-attr]

    errored = run_file(gate_dir / "error.yml")
    assert errored.exit_code == EXIT_ERROR
    assert errored.outcomes[0].error_type == "InputError"


def test_error_wins_over_failure(gate_dir: Path) -> None:
    data = _base(thresholds={"mean_bias": {"abs_max": 0.1}})
    data["checks"].append({**data["checks"][0], "name": "broken", "candidate": "nope.nc"})
    report = run(parse_config(data, base_dir=gate_dir))
    assert [o.status for o in report.outcomes] == ["fail", "error"]
    assert report.exit_code == EXIT_ERROR


def test_threshold_on_missing_group_is_an_error(gate_dir: Path) -> None:
    data = _base(by="season", thresholds={"rmse": {"max": 2, "group": "Winter"}})
    report = run(parse_config(data, base_dir=gate_dir))
    assert report.outcomes[0].status == "error"
    assert "No value" in str(report.outcomes[0].error)


def test_alignment_error_is_reported(gate_dir: Path) -> None:
    data = _base(period=["1950", "1960"], thresholds={"rmse": {"max": 2}})
    report = run(parse_config(data, base_dir=gate_dir), verbose=True)
    o = report.outcomes[0]
    assert o.status == "error" and o.error_type == "NoOverlapError"
    assert o.detail and "Traceback" in o.detail


def test_several_variables_and_sel(gate_dir: Path) -> None:
    ds = xr.open_dataset(gate_dir / "candidate.nc").load()
    ds["tas_copy"] = ds["tas"]
    ds.expand_dims(member=["m1", "m2"]).to_netcdf(gate_dir / "ensemble.nc")
    data = {
        "checks": [
            {
                "name": "multi",
                "reference": "reference.nc",
                "candidate": {"path": "ensemble.nc", "sel": {"member": "m2"}},
                "variables": [
                    {"reference": "t2m", "candidate": "tas"},
                    {"reference": "t2m", "candidate": "tas_copy"},
                ],
                "thresholds": {"rmse": {"max": 2}},
            }
        ]
    }
    report = run(parse_config(data, base_dir=gate_dir))
    assert [o.variable for o in report.outcomes] == ["t2m/tas", "t2m/tas_copy"]
    assert report.exit_code == EXIT_PASS
    assert report.outcomes[0].inputs["candidate"]["sel"] == {"member": "m2"}


def test_check_without_thresholds_passes(gate_dir: Path) -> None:
    report = run(parse_config(_base(), base_dir=gate_dir))
    assert report.exit_code == EXIT_PASS
    assert "nothing was gated" in report.summary_text()


def test_nan_value_fails(gate_dir: Path) -> None:
    # a region outside both grids' valid overlap after masking gives no points
    data = _base(region={"lat": [-80, -60]}, thresholds={"rmse": {"max": 2}})
    report = run(parse_config(data, base_dir=gate_dir))
    assert report.exit_code in (EXIT_FAIL, EXIT_ERROR)


# ---------------------------------------------------------------------------
# Outputs and CLI
# ---------------------------------------------------------------------------


def test_outputs(gate_dir: Path, tmp_path: Path) -> None:
    report = run_file(gate_dir / "fail.yml")
    data = json.loads(report.to_json(tmp_path / "out" / "r.json"))
    assert data["exit_code"] == 1 and data["status"] == "fail"
    assert data["counts"] == {"pass": 0, "fail": 1, "error": 0}
    check = data["checks"][0]
    assert check["thresholds"][0]["passed"] is False
    assert check["comparison"]["provenance"]["inputs"]["candidate"]["location"].endswith(
        "candidate.nc"
    )
    assert data["config"]["sha256"] == report.config_sha256
    md = report.to_markdown()
    assert "## climval check" in md and "**fail**" in md and "abs <= 0.5" in md
    assert "<details><summary>Alignment steps</summary>" in md
    html = report.to_html()
    assert "<!DOCTYPE html>" in html and "europe-strict" in html and "class='fail'" in html
    for text in (md, html, report.summary_text()):
        assert chr(0x2014) not in text and chr(0x2013) not in text


def test_cli_exit_codes(gate_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(gate_dir / "pass.yml"), "--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert main(["check", str(gate_dir / "fail.yml")]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert main(["check", str(gate_dir / "error.yml")]) == 2
    assert main(["check", str(gate_dir / "missing.yml")]) == 2
    assert "configuration" in capsys.readouterr().out


def test_cli_files_and_github_summary(
    gate_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    out = tmp_path / "report"
    code = main(
        [
            "check",
            str(gate_dir / "pass.yml"),
            "--output-dir",
            str(out),
            "--github-summary",
            "--check",
            "north",
        ]
    )
    assert code == 0
    assert {p.name for p in out.iterdir()} == {
        "climval-check.json",
        "climval-check.md",
        "climval-check.html",
    }
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("earlier step\n") and "## climval check" in text
    assert "north" in text and "europe" not in text.split("### ")[-1]
    single = tmp_path / "one.json"
    assert main(["check", str(gate_dir / "pass.yml"), "--json", str(single), "--quiet"]) == 0
    assert json.loads(single.read_text(encoding="utf-8"))["passed"] is True


def test_cli_validate(gate_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(gate_dir / "error.yml"), "--validate"]) == 0
    assert "is valid" in capsys.readouterr().out
    bad = gate_dir / "bad.yml"
    bad.write_text("checks:\n  - reference: a.nc\n", encoding="utf-8")
    assert main(["check", str(bad), "--validate"]) == 2


def test_cli_summary_without_github(
    gate_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["check", str(gate_dir / "pass.yml"), "--github-summary"]) == 0
    assert "GITHUB_STEP_SUMMARY is not set" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# intake-esm catalogue inputs
# ---------------------------------------------------------------------------


def _write_catalog(root: Path) -> Path:
    can = xr.open_dataset(root / "candidate.nc").load()
    rows = []
    for member, shift in (("r1i1p1f1", 0.0), ("r2i1p1f1", 0.5)):
        store = root / f"tas_{member}.zarr"
        (can + shift).assign_attrs(can.attrs).to_zarr(store, mode="w", consolidated=True)
        rows.append(f"SYN,SYNTHETIC-MODEL,historical,{member},Amon,tas,gn,{store.as_posix()}")
    csv = root / "catalog.csv"
    csv.write_text(
        "activity_id,source_id,experiment_id,member_id,table_id,variable_id,grid_label,zstore\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "esmcat_version": "0.1.0",
        "id": "test",
        "description": "synthetic test catalogue",
        "catalog_file": csv.as_posix(),
        "attributes": [
            {"column_name": c, "vocabulary": ""}
            for c in (
                "activity_id",
                "source_id",
                "experiment_id",
                "member_id",
                "table_id",
                "variable_id",
                "grid_label",
            )
        ],
        "assets": {"column_name": "zstore", "format": "zarr"},
        "aggregation_control": {
            "variable_column_name": "variable_id",
            "groupby_attrs": [
                "activity_id",
                "source_id",
                "experiment_id",
                "table_id",
                "grid_label",
            ],
            "aggregations": [
                {"type": "union", "attribute_name": "variable_id"},
                {
                    "type": "join_new",
                    "attribute_name": "member_id",
                    "options": {"coords": "minimal", "compat": "override"},
                },
            ],
        },
    }
    path = root / "catalog.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_catalog_input(gate_dir: Path) -> None:
    pytest.importorskip("intake_esm")
    pytest.importorskip("zarr")
    pytest.importorskip("dask")
    _write_catalog(gate_dir)
    query = {"source_id": "SYNTHETIC-MODEL", "variable_id": "tas", "table_id": "Amon"}
    one = {
        "checks": [
            {
                "name": "catalogue",
                "reference": "reference.nc",
                "candidate": {
                    "catalog": "catalog.json",
                    "query": {**query, "member_id": "r1i1p1f1"},
                },
                "variable": {"reference": "t2m", "candidate": "tas"},
                "region": [35, 70, -10, 40],
                "thresholds": {"rmse": {"max": 2}},
            }
        ]
    }
    report = run(parse_config(one, base_dir=gate_dir))
    o = report.outcomes[0]
    assert o.status == "pass", o.error
    assert o.inputs["candidate"]["squeezed"] == ["member_id"]
    assert o.comparison is not None and "member_id" not in o.comparison.candidate.dims
    assert "SYNTHETIC-MODEL r1i1p1f1" == o.comparison.candidate_name

    # two members stay as an ensemble dimension: one value per member
    both = json.loads(json.dumps(one))
    both["checks"][0]["candidate"]["query"] = query
    report = run(parse_config(both, base_dir=gate_dir))
    o = report.outcomes[0]
    assert o.status == "pass", o.error
    members = {c.point.get("member_id") for c in o.thresholds.checks}  # type: ignore[union-attr]
    assert members == {"r1i1p1f1", "r2i1p1f1"}

    none = json.loads(json.dumps(one))
    none["checks"][0]["candidate"]["query"] = {"source_id": "NOT-A-MODEL"}
    report = run(parse_config(none, base_dir=gate_dir))
    assert report.exit_code == EXIT_ERROR and "matched no dataset" in str(report.outcomes[0].error)


def test_configuration_examples_in_the_docs_are_valid() -> None:
    import re

    import yaml

    root = Path(__file__).parent.parent
    docs = root / "docs"
    if not docs.is_dir():
        pytest.skip("docs/ not present")
    blocks = []
    for page in [root / "README.md", *sorted(docs.rglob("*.md"))]:
        for block in re.findall(r"```yaml\n(.*?)```", page.read_text(encoding="utf-8"), re.S):
            if re.search(r"^checks:", block, re.M):
                blocks.append((page.name, block))
    assert len(blocks) >= 3
    for name, block in blocks:
        config = parse_config(yaml.safe_load(block))
        assert config.checks, name
