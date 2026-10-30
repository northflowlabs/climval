# Command line

```text
climval check CONFIG [--check NAME] [--json PATH] [--markdown PATH] [--html PATH]
                     [--output-dir DIR] [--github-summary] [--validate] [--verbose] [--quiet]
climval compare REFERENCE CANDIDATE [--variable NAME or REF,CAND]
                     [--lat S,N] [--lon W,E] [--region REGION] [--period START END]
                     [--regrid METHOD] [--target GRID] [--freq FREQ]
                     [--min-coverage FRACTION] [--units MODE] [--by GROUPING]
                     [--seasons MODE] [--metrics NAME ...] [--output PATH]
climval demo [--by GROUPING] [--output PATH]
climval run ...            # 0.1 interface: BenchmarkSuite on data files
climval list-metrics
climval list-presets
climval info --model PRESET
```

`climval <command> --help` lists every option. `climval check` is described
in [Thresholds and the CI gate](../how-to/ci-gate.md), including its exit
codes.

## climval compare

| Option | Value |
|---|---|
| `--variable` | a name, or `REF,CAND` when the names differ, for example `t2m,tas` |
| `--lat S,N` | latitude range; negative edges work as written: `--lat -60,-30` or `--lat -60 -30` |
| `--lon W,E` | longitude range read eastward; `W > E` crosses the antimeridian, for example `--lon 170,-170` |
| `--region` | `S,N` or `S,N,W,E` (write `--region=-60,-30,-10,40` when S is negative), or a regionmask name such as `ar6.land:NEU`; not together with `--lat` or `--lon` |
| `--period START END` | for example `--period 1995 2014` |
| `--regrid` | `bilinear` (default), `nearest` or `conservative` (xESMF) |
| `--target` | `coarser` (default), `reference` or `candidate` |
| `--freq` | `h` or `<N>h` (`3h`, `6h`), `D`, `MS` or `YS`; default the coarser native frequency |
| `--min-coverage` | fraction in (0, 1] of a period's native steps that must be valid when aggregating (default `0.8`) |
| `--units` | `convert` (default), `strict` or `ignore` |
| `--by` | `season`, `month` or `year` |
| `--seasons` | `contiguous` (default) or `calendar` |
| `--metrics` | metric names or aliases; default all default metrics |
| `--output` | report path, `.html`, `.json` or `.md` |

```bash
climval compare era5.nc tas.nc --variable t2m,tas --lat -60,-30 --lon -70,-50 \
    --period 1995 2014 --by season --output report.html
```
