# The GitHub Action

The repository root holds a composite action that installs climval, runs
`climval check` on a configuration file, writes the Markdown summary to the
job summary and uploads the report as an artifact.

## Copy-paste workflow

Save as `.github/workflows/climval.yml`:

```yaml
name: Validate model output

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  climval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Produce or download the candidate here, for example a short model run
      # that writes output/tas_monthly.nc.

      - name: climval check
        uses: northflowlabs/climval@v0.2.0
        with:
          config: climval.yml
          version: "0.2.0"
          extras: cloud,catalog
```

with a `climval.yml` next to it, for example:

```yaml
version: 1
checks:
  - name: europe-tas
    reference: data/era5_t2m_monthly_europe.nc
    candidate: output/tas_monthly.nc
    variable: {reference: t2m, candidate: tas}
    region: {lat: [35, 72], lon: [-10, 40]}
    by: season
    thresholds:
      mean_bias: {abs_max: 2.0, group: "*"}
      pearson_r: {min: 0.95, sample: climatology}
```

The job fails when a threshold fails or a check cannot run. The job summary
shows a table of checks and of every limit; the artifact `climval-report`
holds `climval-check.json`, `climval-check.md` and `climval-check.html`.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `config` | (required) | path to the configuration file |
| `version` | `latest` | climval version from PyPI, `latest`, or `source` to install from the action's own checkout |
| `extras` | empty | pip extras, for example `cloud,catalog` |
| `python-version` | `3.12` | passed to actions/setup-python; empty skips it |
| `checks` | empty | space-separated check names; empty runs all |
| `args` | empty | further arguments for `climval check`, for example `--verbose` |
| `report-dir` | `climval-report` | where the report files go |
| `job-summary` | `true` | write the Markdown summary to the job summary |
| `upload-report` | `true` | upload `report-dir` as an artifact |
| `artifact-name` | `climval-report` | artifact name (make it unique per job in a matrix) |
| `fail-on` | `threshold` | `threshold`: fail on exit codes 1 and 2; `error`: only on 2; `never`: never fail |

## Outputs

| Output | Meaning |
|---|---|
| `exit-code` | 0 passed, 1 threshold failure, 2 configuration or data error |
| `status` | `pass`, `fail` or `error` |
| `report-dir` | directory of the report files |
| `json` | path to the JSON result |

## Patterns

**Report without gating**, then decide in a later step:

```yaml
      - name: climval check
        id: climval
        uses: northflowlabs/climval@v0.2.0
        with:
          config: climval.yml
          fail-on: never
      - name: Comment or label on failure
        if: steps.climval.outputs.status == 'fail'
        run: echo "A threshold failed; see the job summary"
```

**Several configurations in a matrix**, each with its own artifact:

```yaml
    strategy:
      matrix:
        region: [europe, nordic, mediterranean]
    steps:
      - uses: actions/checkout@v4
      - uses: northflowlabs/climval@v0.2.0
        with:
          config: checks/${{ matrix.region }}.yml
          artifact-name: climval-${{ matrix.region }}
```

**Cloud data** needs only the `cloud` extra for public buckets
(`storage_options: {token: anon}` in the configuration) and the `catalog`
extra for intake-esm catalogues. For private buckets, set the credentials
through the usual environment variables or a prior authentication step.

Pin the action to a release tag (`@v0.2.0`) or a commit SHA, and pin
`version` to the same release, so the gate does not change under you.

## How this repository tests the action

The workflow `.github/workflows/action.yml` runs the action from the
repository itself (`uses: ./` with `version: source`) on the small synthetic
files in `tests/data/gate/`: a passing configuration on Linux, Windows and
macOS, a threshold failure that must fail the step with exit code 1, and a
missing file that must report exit code 2.
