# Contributing to climval

Thank you for your interest in contributing. Everyone taking part is expected
to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Bug reports, questions and ideas go in
[GitHub issues](https://github.com/northflowlabs/climval/issues). For a bug,
include the climval version, a short script that shows the problem (the
synthetic fields from `climval.demo_data()` are often enough) and the full
error message.

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

## Documentation

The documentation is built with MkDocs Material from `docs/`; the API
reference comes from the docstrings (NumPy style) through mkdocstrings.

```bash
pip install -e ".[docs]"
mkdocs serve                   # preview at http://127.0.0.1:8000
mkdocs build --strict          # what CI runs
```

The example notebooks in `examples/` are rendered with their stored outputs
and are not executed during the build. If you change one, or change code
that affects its numbers, execute it again and check that every statement in
its text matches the outputs:

```bash
pip install -e ".[cloud,catalog,regions]" matplotlib nbconvert ipykernel
jupyter nbconvert --to notebook --execute --inplace   --ExecutePreprocessor.timeout=1800 examples/*.ipynb
```

They read about 730 MB from public cloud storage in total and write their
reports to `examples/output/`, which is not committed.

## climval check and the Action

`tests/test_gate.py` covers the configuration, the exit codes and the
outputs. The small files in `tests/data/gate/` are synthetic and are written
by `tests/data/gate/make_data.py`. `.github/workflows/action.yml` runs the
action in `action.yml` on them.

## Code standards

- **Ruff** for linting: `ruff check climval/ tests/`
- **Mypy** for types: `mypy climval/`. The result depends on the NumPy type
  stubs; CI runs it on Python 3.11 with the newest releases, so run
  `pip install -U -e ".[dev]"` first for the same result
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
- [ ] Docs build (`mkdocs build --strict`)
- [ ] CHANGELOG.md updated

## Releasing

1. Set the version in `pyproject.toml`.
2. In `CHANGELOG.md`, replace `Unreleased` in the version's heading with the
   release date: `## [0.2.0] - YYYY-MM-DD`.
3. In `CITATION.cff`, set `version` to the same version and `date-released`
   to the same date (`date-released: 2026-11-02`, for example), then check
   the file with `cffconvert --validate`.
4. Set the version in `recipe/recipe.yaml`; its `sha256` is filled in from
   the sdist once it is on PyPI.
5. Commit, then tag and push the tag: `git tag v0.2.0 && git push origin v0.2.0`.

The tag starts `.github/workflows/publish.yml`. It stops before building if
the tag does not match the package version, if the changelog heading has no
date, or if `CITATION.cff` has the wrong version or no `date-released`. It
then tests, builds and publishes to PyPI (trusted publishing), and finally
creates the GitHub Release for the tag with the tag's `CHANGELOG.md` section
as the notes, unless a release for the tag already exists. Zenodo archives
each GitHub Release and mints its DOI.

## Commit style

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Kling-Gupta Efficiency metric
fix: correct weighted RMSE normalization
docs: add NetCDF loading example
test: add TSS unit test with known value
```
