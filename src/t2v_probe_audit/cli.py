"""Command line entry point.

    python -m t2v_probe_audit manifest    build data/manifest.json if missing
    python -m t2v_probe_audit run         run the three experiments
    python -m t2v_probe_audit analyze     recompute numbers.json and checks.json
    python -m t2v_probe_audit artifacts   render figures and table source data
    python -m t2v_probe_audit all         manifest + run + analyze + artifacts
"""

import argparse
import sys

from . import analysis, artifacts, runner
from .config import GENERATORS
from .manifest import build_manifest


def _common(parser):
    parser.add_argument("--n", type=int, default=60,
                        help="clips sampled per generator (default: 60)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="sampling and probe seed (default: 2026)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker processes (default: min(8, cpus))")
    parser.add_argument("--no-embedding", action="store_true",
                        help="skip DINOv2 (only for quick smoke runs)")


def _run(args):
    runner.run_regimes(args.n, args.seed, args.workers,
                       with_embedding=not args.no_embedding)
    runner.run_density_control(args.n, args.seed, args.workers,
                               with_embedding=not args.no_embedding)
    runner.run_consequence(args.n, args.seed, args.workers)
    return 0


def _analyze(args):
    return analysis.report(analysis.build_numbers())


def _artifacts(args):
    artifacts.build_all()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="t2v-probe-audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="build data/manifest.json")
    p_manifest.add_argument("--force", action="store_true",
                            help="rebuild even if the manifest exists")

    p_run = sub.add_parser("run", help="run the three experiments")
    _common(p_run)

    p_analyze = sub.add_parser("analyze", help="recompute numbers from records")
    p_artifact = sub.add_parser("artifacts", help="render figures and tables")

    p_all = sub.add_parser("all", help="manifest + run + analyze + artifacts")
    _common(p_all)
    p_all.add_argument("--force-manifest", action="store_true")

    args = ap.parse_args(argv)

    if args.command == "manifest":
        build_manifest(force=args.force)
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "artifacts":
        return _artifacts(args)
    if args.command == "all":
        build_manifest(force=args.force_manifest)
        if _run(args) != 0:
            return 1
        if _analyze(args) != 0:
            return 1
        return _artifacts(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
