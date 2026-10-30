# Command line

```text
climval check CONFIG [--check NAME] [--json PATH] [--markdown PATH] [--html PATH]
                     [--output-dir DIR] [--github-summary] [--validate] [--verbose] [--quiet]
climval compare REFERENCE CANDIDATE [--variable NAME or REF,CAND] [--period START END]
                     [--regrid METHOD] [--target GRID] [--freq FREQ] [--units MODE]
                     [--by GROUPING] [--metrics NAME ...] [--output PATH]
climval demo [--by GROUPING] [--output PATH]
climval run ...            # 0.1 interface: BenchmarkSuite on data files
climval list-metrics
climval list-presets
climval info --model PRESET
```

`climval <command> --help` lists every option. `climval check` is described
in [Thresholds and the CI gate](../how-to/ci-gate.md), including its exit
codes.
