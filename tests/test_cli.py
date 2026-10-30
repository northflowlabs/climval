# SPDX-License-Identifier: Apache-2.0
"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import climval
from climval.cli import main


def test_compare_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref, can = climval.demo_data()
    ref.to_dataset().to_netcdf(tmp_path / "ref.nc")
    can.to_dataset().to_netcdf(tmp_path / "can.nc")
    out = tmp_path / "report.json"
    main(
        [
            "compare",
            str(tmp_path / "ref.nc"),
            str(tmp_path / "can.nc"),
            "--variable",
            "tas",
            "--region",
            "35,72,-10,40",
            "--by",
            "season",
            "--output",
            str(out),
        ]
    )
    assert "mean_bias" in capsys.readouterr().out
    assert out.stat().st_size > 0


def test_run_without_data_exits_with_message() -> None:
    with pytest.raises(SystemExit, match="synthetic=True"):
        main(["run", "--reference", "era5", "--candidates", "cmip6-mpi"])


def test_run_synthetic(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.warns(UserWarning, match="synthetic"):
        main(
            [
                "run",
                "--reference",
                "era5",
                "--candidates",
                "cmip6-mpi",
                "--synthetic",
                "--n-samples",
                "50",
            ]
        )
    assert "WARNING" in capsys.readouterr().out


def test_demo_and_lists(capsys: pytest.CaptureFixture[str]) -> None:
    main(["demo"])
    main(["list-metrics"])
    out = capsys.readouterr().out
    assert "not a validation" in out and "spearman_r" in out
