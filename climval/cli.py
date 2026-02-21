"""
climval CLI
======================
Usage:
    climval run --reference era5 --candidates cmip6-mpi cordex-eur
    climval list-metrics
    climval list-presets
    climval info --model era5
"""

from __future__ import annotations

import argparse
import sys


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

    for cand_name in args.candidates:
        cand = load_model(
            preset=cand_name if cand_name in _KNOWN_PRESETS else None,
            name=cand_name,
        )
        suite.register(cand)

    report = suite.run(
        variables=args.variables or None,
        n_samples=args.n_samples,
    )

    report.summary()

    if args.output:
        report.export(args.output)
        print(f"  ✓ Report exported → {args.output}\n")


def cmd_list_metrics(_args: argparse.Namespace) -> None:
    from climval.metrics.stats import METRIC_REGISTRY

    print("\n  Available metrics:\n")
    for name, metric in METRIC_REGISTRY.items():
        direction = "↑ higher better" if metric.higher_is_better else "↓ lower better"
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
        description="Open benchmarking framework for climate model comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run a benchmark comparison")
    run_p.add_argument("--reference", required=True, help="Reference model/preset name")
    run_p.add_argument("--ref-path", default=None, help="Path to reference data file")
    run_p.add_argument("--candidates", nargs="+", required=True)
    run_p.add_argument("--variables", nargs="*", default=None)
    run_p.add_argument("--output", default=None, help="Output path (.html/.json/.md)")
    run_p.add_argument("--name", default="benchmark", help="Suite name")
    run_p.add_argument("--n-samples", type=int, default=1000,
                       help="Synthetic sample count (if no data files provided)")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
