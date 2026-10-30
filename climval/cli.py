"""
climval CLI
===========
Usage:
    climval compare REFERENCE CANDIDATE --variable tas --by season --output report.html
    climval run --reference era5 --ref-path era5.nc --candidates mpi --candidate-paths mpi.nc
    climval demo
    climval list-metrics
    climval list-presets
    climval info --model era5
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

_RANGE_OPTIONS = ("--region", "--lat", "--lon")


def _pair(values: list[str] | None, option: str) -> tuple[float, float] | None:
    """'S,N' or 'S N' (also with negative numbers) as two floats."""
    if not values:
        return None
    parts = [p for v in values for p in v.split(",") if p != ""]
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise SystemExit(f"climval: {option} takes two numbers, got {' '.join(values)}") from None
    if len(numbers) != 2:
        raise SystemExit(f"climval: {option} takes two numbers, got {' '.join(values)}")
    return numbers[0], numbers[1]


def _region(value: str | None, lat: Any = None, lon: Any = None) -> Any:
    """--lat S,N and --lon W,E, or --region S,N[,W,E] or a regionmask name."""
    from climval.align.region import Box

    lat_pair = _pair(lat, "--lat")
    lon_pair = _pair(lon, "--lon")
    if value is not None and (lat_pair or lon_pair):
        raise SystemExit("climval: use either --region or --lat/--lon, not both")
    if lat_pair or lon_pair:
        return Box(lat=lat_pair or (-90.0, 90.0), lon=lon_pair)
    if value is None:
        return None
    if ":" in value:
        return value
    try:
        parts = [float(p) for p in value.split(",")]
    except ValueError:
        raise SystemExit(
            f"climval: --region takes S,N or S,N,W,E or a regionmask name, got {value!r}"
        ) from None
    if len(parts) == 2:
        return Box(lat=(parts[0], parts[1]))
    if len(parts) == 4:
        return Box(lat=(parts[0], parts[1]), lon=(parts[2], parts[3]))
    raise SystemExit("climval: --region takes S,N or S,N,W,E or a regionmask name")


def _join_negative_values(argv: list[str]) -> list[str]:
    """Let '--lat -10,10' through: argparse would read '-10,10' as an option."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if arg in _RANGE_OPTIONS and nxt is not None and nxt.startswith("-") and len(nxt) > 1:
            if "," in nxt and (nxt[1].isdigit() or nxt[1] == "."):
                out.append(f"{arg}={nxt}")
                i += 2
                continue
        out.append(arg)
        i += 1
    return out


def _variable(value: str | None) -> Any:
    if value is None:
        return None
    if "," in value:
        ref, can = value.split(",", 1)
        return (ref, can)
    return value


def cmd_compare(args: argparse.Namespace) -> None:
    import climval

    result = climval.compare(
        args.reference,
        args.candidate,
        _variable(args.variable),
        region=_region(args.region, args.lat, args.lon),
        period=tuple(args.period) if args.period else None,
        regrid=args.regrid,
        target=args.target,
        freq=args.freq,
        min_coverage=args.min_coverage,
        units=args.units,
        by=args.by,
        metrics=args.metrics or None,
    )
    result.summary()
    if args.output:
        result.export(args.output)
        print(f"\n  Report written to {args.output}\n")


def cmd_demo(args: argparse.Namespace) -> None:
    import climval

    result = climval.demo(by=args.by)
    result.summary()
    if args.output:
        result.export(args.output)
        print(f"\n  Report written to {args.output}\n")


def cmd_run(args: argparse.Namespace) -> None:
    from climval.core.loader import load_model
    from climval.core.suite import BenchmarkSuite

    suite = BenchmarkSuite(name=args.name)

    ref = load_model(
        preset=args.reference if args.reference in _KNOWN_PRESETS else None,
        name=args.reference,
        path=args.ref_path or None,
    )
    suite.register(ref, role="reference")

    paths = args.candidate_paths or []
    if paths and len(paths) != len(args.candidates):
        sys.exit("--candidate-paths needs one path per candidate")
    for i, cand_name in enumerate(args.candidates):
        cand = load_model(
            preset=cand_name if cand_name in _KNOWN_PRESETS else None,
            name=cand_name,
            path=paths[i] if paths else None,
        )
        suite.register(cand)

    try:
        report = suite.run(
            variables=args.variables or None,
            n_samples=args.n_samples,
            synthetic=args.synthetic,
        )
    except ValueError as exc:
        sys.exit(f"climval: {exc}")

    report.summary()

    if args.output:
        report.export(args.output)
        print(f"  ✓ Report exported → {args.output}\n")


def cmd_list_metrics(_args: argparse.Namespace) -> None:
    from climval.metrics.stats import METRIC_REGISTRY

    print("\n  Available metrics:\n")
    for name, metric in METRIC_REGISTRY.items():
        direction = "↑ higher better" if metric.higher_is_better else "↓ lower better"
        if name == "std_ratio":
            direction = "best at 1"
        print(f"  {name:<30} {direction}")
    print()


def cmd_list_presets(_args: argparse.Namespace) -> None:
    from climval.core.loader import _PRESETS

    print("\n  Available presets:\n")
    for name, info in _PRESETS.items():
        print(f"  {name:<20} {info.get('description', '')}")
    print()


def cmd_info(args: argparse.Namespace) -> None:
    from climval.core.loader import load_model

    model = load_model(preset=args.model)
    print(f"\n{model.summary()}\n")


_KNOWN_PRESETS = {"era5", "era5-land", "cmip6-mpi", "cmip6-hadgem", "cordex-eur"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="climval",
        description="Validate climate model output against reference data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compare
    cmp_p = sub.add_parser("compare", help="Align and compare two datasets (paths or URLs)")
    cmp_p.add_argument("reference", help="Reference NetCDF, Zarr or CSV path or URL")
    cmp_p.add_argument("candidate", help="Candidate NetCDF, Zarr or CSV path or URL")
    cmp_p.add_argument("--variable", default=None, help="Name, or REF,CAND when they differ")
    cmp_p.add_argument(
        "--lat",
        nargs="+",
        default=None,
        metavar="S,N",
        help="Latitude range, for example --lat -10,10 or --lat -10 10",
    )
    cmp_p.add_argument(
        "--lon",
        nargs="+",
        default=None,
        metavar="W,E",
        help="Longitude range read eastward (W > E crosses the antimeridian)",
    )
    cmp_p.add_argument(
        "--region",
        default=None,
        help="S,N or S,N,W,E (write --region=-60,-30 when S is negative), or e.g. ar6.land:NEU",
    )
    cmp_p.add_argument("--period", nargs=2, default=None, metavar=("START", "END"))
    cmp_p.add_argument(
        "--regrid", default="bilinear", choices=["bilinear", "nearest", "conservative"]
    )
    cmp_p.add_argument(
        "--target", default="coarser", choices=["coarser", "reference", "candidate"]
    )
    cmp_p.add_argument("--freq", default=None, help="h, 3h (any N dividing 24), D, MS or YS")
    cmp_p.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Fraction of a period's steps that must be valid when aggregating (default 0.8)",
    )
    cmp_p.add_argument("--units", default="convert", choices=["convert", "strict", "ignore"])
    cmp_p.add_argument("--by", default=None, choices=["season", "month", "year"])
    cmp_p.add_argument("--metrics", nargs="*", default=None)
    cmp_p.add_argument("--output", default=None, help="Output path (.html/.json/.md)")
    cmp_p.set_defaults(func=cmd_compare)

    # demo
    demo_p = sub.add_parser("demo", help="Compare two synthetic example fields")
    demo_p.add_argument("--by", default=None, choices=["season", "month", "year"])
    demo_p.add_argument("--output", default=None)
    demo_p.set_defaults(func=cmd_demo)

    # run (0.1 interface)
    run_p = sub.add_parser("run", help="Run a BenchmarkSuite comparison (0.1 interface)")
    run_p.add_argument("--reference", required=True, help="Reference model/preset name")
    run_p.add_argument("--ref-path", default=None, help="Path to reference data file")
    run_p.add_argument("--candidates", nargs="+", required=True)
    run_p.add_argument(
        "--candidate-paths", nargs="*", default=None, help="One data path per candidate"
    )
    run_p.add_argument("--variables", nargs="*", default=None)
    run_p.add_argument("--output", default=None, help="Output path (.html/.json/.md)")
    run_p.add_argument("--name", default="benchmark", help="Suite name")
    run_p.add_argument(
        "--synthetic",
        action="store_true",
        help="Score random arrays when no data is given (pipeline checks only)",
    )
    run_p.add_argument(
        "--n-samples", type=int, default=1000, help="Synthetic sample count (with --synthetic)"
    )
    run_p.set_defaults(func=cmd_run)

    # list-metrics
    lm_p = sub.add_parser("list-metrics", help="List available metrics")
    lm_p.set_defaults(func=cmd_list_metrics)

    # list-presets
    lp_p = sub.add_parser("list-presets", help="List known model presets")
    lp_p.set_defaults(func=cmd_list_presets)

    # info
    info_p = sub.add_parser("info", help="Show metadata for a model preset")
    info_p.add_argument("--model", required=True)
    info_p.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(_join_negative_values(list(sys.argv[1:] if argv is None else argv)))
    args.func(args)


if __name__ == "__main__":
    main()
