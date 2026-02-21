# Contributing to climval

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/northflow-technologies/climval
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

- **Ruff** for linting: `ruff check climval/`
- **Mypy** for types: `mypy climval/`
- All public functions must have docstrings
- All new metrics must have at least one unit test with a known analytical value
- All new code must be covered ≥ 90%

## Adding a metric

1. Subclass `BaseMetric` in `climval/metrics/stats.py`
2. Set `name`, `higher_is_better`
3. Implement `compute(reference, candidate, weights=None) -> float`
4. Register it in `METRIC_REGISTRY`
5. Add a test in `tests/test_benchmark.py` with a known analytical value

```python
class KGE(BaseMetric):
    """Kling-Gupta Efficiency."""
    name = "kge"
    higher_is_better = True

    def compute(self, reference, candidate, weights=None):
        r = np.corrcoef(reference, candidate)[0, 1]
        alpha = np.std(candidate) / np.std(reference)
        beta = np.mean(candidate) / np.mean(reference)
        return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
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
- [ ] Linting passes (`ruff check .`)
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
