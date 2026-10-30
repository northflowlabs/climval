# SPDX-License-Identifier: Apache-2.0
"""Fixes made before the 0.2.0 release."""

from __future__ import annotations

import io
import sys
from typing import Any

import numpy as np
import pytest
from conftest import make_field, monthly_times

import climval
from climval.cli import main
from climval.errors import ConfigError
from climval.gate import parse_config


def _check(**extra: Any) -> dict[str, Any]:
    check = {
        "name": "c",
        "reference": "reference.nc",
        "candidate": "candidate.nc",
        "variable": {"reference": "t2m", "candidate": "tas"},
        **extra,
    }
    return {"version": 1, "checks": [check]}


# ---------------------------------------------------------------------------
# A threshold on a metric that is not computed by default needs `metrics:`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["spearman_r", "spearman", "percentile_bias_p99_9"])
def test_threshold_outside_default_metrics_needs_metrics(metric: str) -> None:
    with pytest.raises(ConfigError, match="not computed by default; list the metrics"):
        parse_config(_check(thresholds={metric: {"max": 1.0}}))


@pytest.mark.parametrize("metric", ["spearman_r", "percentile_bias_p99_9"])
def test_threshold_outside_default_metrics_accepted_when_listed(metric: str) -> None:
    config = parse_config(_check(metrics=["rmse", metric], thresholds={metric: {"max": 1.0}}))
    assert config.checks[0].thresholds[0]["metric"] == climval.metrics.get_metric(metric).name


@pytest.mark.parametrize("metric", ["rmse", "bias", "percentile_bias_p95", "n_points"])
def test_threshold_on_default_metric_needs_no_metrics(metric: str) -> None:
    parse_config(_check(thresholds={metric: {"min": -10.0}}))


# ---------------------------------------------------------------------------
# Console output on a stream that cannot encode every character
# ---------------------------------------------------------------------------


def test_list_metrics_on_cp1252_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    main(["list-metrics"])
    print("arrow ↓ after main")
    stream.flush()
    out = raw.getvalue().decode("cp1252")
    assert "mean_bias" in out
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert lines["mean_bias"].endswith("smaller magnitude is better")
    assert lines["percentile_bias_p95"].endswith("smaller magnitude is better")
    assert lines["rmse"].endswith("lower is better")
    assert lines["pearson_r"].endswith("higher is better")
    assert "arrow ? after main" in out


def test_run_summary_on_cp1252_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
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
    stream.flush()
    out = raw.getvalue().decode("cp1252")
    assert "Candidate:" in out
    assert out.isascii()


# ---------------------------------------------------------------------------
# n_points counts only points with a positive weight
# ---------------------------------------------------------------------------


def _n_points(res: climval.Comparison, sample: str) -> float:
    df = res.to_dataframe()
    row = df[(df.metric == "n_points") & (df["sample"] == sample) & (df.group == "all")]
    return float(row["value"].iloc[0])


def test_n_points_leaves_out_zero_weights() -> None:
    lat = np.arange(-87.0, 90.0, 6.0)
    lon = np.arange(-177.0, 180.0, 6.0)
    times = monthly_times("2000-01-01", 12)
    ref = make_field(lat, lon, times)
    can = ref + 0.5
    can.attrs = ref.attrs
    area = np.cos(np.deg2rad(ref["lat"])).broadcast_like(ref.isel(time=0, drop=True)).copy()
    area[:5, :] = 0.0
    n_cells = lat.size * lon.size
    weighted_cells = (lat.size - 5) * lon.size

    full = climval.compare(ref, can, area=area * 0 + 1)
    assert _n_points(full, "space_time") == 12 * n_cells

    res = climval.compare(ref, can, area=area)
    assert _n_points(res, "space_time") == 12 * weighted_cells
    assert _n_points(res, "climatology") == weighted_cells
