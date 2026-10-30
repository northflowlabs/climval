# Contributing to climval

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/northflowlabs/climval
cd climval
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                         # all tests
pytest tests/ -v               # verbose
pytest --cov=climval # with coverage
```

## Code standards

- **Ruff** for linting: `ruff check climval/ tests/`
- **Mypy** for types: `mypy climval/`
- All public functions must have docstrings
- All new metrics must have at least one unit test with a known analytical value
- All new code must be covered to at least 90%
- Alignment changes need a unit test for the step and an end-to-end test
- Code must stay lazy on dask: see `tests/test_dask.py`

## Adding a metric

1. Subclass `BaseMetric` in `climval/metrics/stats.py`
2. Set `name`, `higher_is_better` and `units_kind` ("data", "dimensionless" or "percent")
3. Implement `reduce(reference, candidate, weights, dims) -> xr.DataArray` with
   the weighted helpers (`wmean`, `wstd`, `wcorr`), so it stays lazy on dask.
   Inputs are aligned and NaN where either is missing; weights are zero there
4. Register it in `METRIC_REGISTRY`
5. Add a test in `tests/test_metrics.py` with a known analytical value, and
   against xskillscore when it has the metric

```python
class MSE(BaseMetric):
    """Mean squared error."""
    name = "mse"
    higher_is_better = False
    units_kind = None

    def reduce(self, reference, candidate, weights, dims):
        return wmean((candidate - reference) ** 2, weights, dims)
```

## Adding a model preset

Add a new entry to `_PRESETS` in `climval/core/loader.py`:

```python
"my-model": {
    "model_type": ModelType.GCM,
    "spatial_resolution": SpatialResolution.MEDIUM,
    "temporal_resolution": TemporalResolution.MONTHLY,
    "description": "My model description",
},
```

## Pull request checklist

- [ ] Tests pass (`pytest`)
- [ ] Linting passes (`ruff check climval tests`)
- [ ] Type checks pass (`mypy climval/`)
- [ ] Docstrings added for new public API
- [ ] CHANGELOG.md updated

## Commit style

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Kling-Gupta Efficiency metric
fix: correct weighted RMSE normalization
docs: add NetCDF loading example
test: add TSS unit test with known value
```
