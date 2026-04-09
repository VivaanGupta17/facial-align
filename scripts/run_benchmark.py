#!/usr/bin/env python3
"""
Run the Facial Align pipeline benchmark suite.

Profiles each stage of the surgical planning pipeline with synthetic data
and generates a performance report. Use for:
- Establishing performance baselines before deployment
- Detecting regressions during CI
- Comparing hardware configurations (CPU vs GPU)

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --iterations 20 --output benchmark_report.md
    python scripts/run_benchmark.py --stages segmentation,mesh_extraction
    python scripts/run_benchmark.py --json  # Machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.benchmark.profiler import (
    DEFAULT_BASELINES,
    PipelineBenchmarkSuite,
    PipelineStage,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Facial Align pipeline benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py                          # Full benchmark, 5 iterations
  python scripts/run_benchmark.py --iterations 20          # More iterations for stability
  python scripts/run_benchmark.py --stages segmentation    # Single stage
  python scripts/run_benchmark.py --json --output out.json # JSON output
  python scripts/run_benchmark.py --volume 256,256,100     # Smaller volume for quick test
        """,
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=5,
        help="Number of timed iterations per stage (default: 5)",
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--stages", "-s", type=str, default=None,
        help="Comma-separated list of stages to benchmark (default: all)",
    )
    parser.add_argument(
        "--volume", type=str, default="512,512,200",
        help="Volume shape as 'X,Y,Z' (default: 512,512,200)",
    )
    parser.add_argument(
        "--structures", type=int, default=15,
        help="Number of segmented structures (default: 15)",
    )
    parser.add_argument(
        "--fragments", type=int, default=4,
        help="Number of fracture fragments (default: 4)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of Markdown",
    )
    parser.add_argument(
        "--check-regression", action="store_true",
        help="Check results against performance baselines",
    )
    args = parser.parse_args()

    # Parse volume shape
    try:
        vol_shape = tuple(int(x) for x in args.volume.split(","))
        assert len(vol_shape) == 3
    except (ValueError, AssertionError):
        print(f"Error: Invalid volume shape '{args.volume}'. Expected format: X,Y,Z")
        sys.exit(1)

    # Parse stages
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
        valid_stages = {e.value for e in PipelineStage}
        for s in stages:
            if s not in valid_stages:
                print(f"Error: Unknown stage '{s}'. Valid stages: {', '.join(sorted(valid_stages))}")
                sys.exit(1)

    print(f"Facial Align Pipeline Benchmark")
    print(f"  Volume: {vol_shape}")
    print(f"  Structures: {args.structures}")
    print(f"  Fragments: {args.fragments}")
    print(f"  Iterations: {args.iterations} (+ {args.warmup} warmup)")
    if stages:
        print(f"  Stages: {', '.join(stages)}")
    print()

    suite = PipelineBenchmarkSuite(
        volume_shape=vol_shape,
        n_structures=args.structures,
        n_fragments=args.fragments,
    )

    print("Running benchmarks...")
    report = suite.run(
        n_iterations=args.iterations,
        warmup=args.warmup,
        stages=stages,
    )

    # Check regressions
    if args.check_regression:
        print("\nRegression check:")
        regressions = []
        for stage_stats in report.stages:
            baseline = DEFAULT_BASELINES.get(stage_stats.stage)
            if baseline:
                warning = baseline.check(stage_stats)
                if warning:
                    regressions.append(warning)
                    print(f"  REGRESSION: {warning}")
                else:
                    print(f"  OK: {stage_stats.stage}")
        if regressions:
            print(f"\n{len(regressions)} regression(s) detected.")
        else:
            print("\nNo regressions detected.")
        print()

    # Output
    if args.json:
        output = json.dumps(report.to_dict(), indent=2)
    else:
        output = report.to_markdown()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output)
        print(f"Report written to {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
