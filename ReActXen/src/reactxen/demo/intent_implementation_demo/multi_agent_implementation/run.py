"""CLI entry point for multi-agent benchmark."""

import argparse
import sys
from pathlib import Path

# Add project root for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from multi_agent_implementation.benchmark_runner import run_benchmark, save_results


def main():
    parser = argparse.ArgumentParser(description="Run PHMForge multi-agent benchmark")
    parser.add_argument(
        "--scenario-file",
        type=str,
        default=str(Path(__file__).parent.parent / "scenarios" / "phm_scenarios.json"),
        help="Path to scenario JSON file",
    )
    parser.add_argument("--model-id", type=int, default=8, help="Model ID (default: 8)")
    parser.add_argument(
        "--model-source",
        type=str,
        default="watsonx",
        choices=["watsonx", "huggingface"],
        help="Model source",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    print(f"Running multi-agent benchmark:")
    print(f"  Scenarios: {args.scenario_file}")
    print(f"  Model: {args.model_source}/{args.model_id}")
    if args.limit:
        print(f"  Limit: {args.limit} scenarios")

    results = run_benchmark(
        scenario_file=args.scenario_file,
        model_id=args.model_id,
        model_source=args.model_source,
        limit=args.limit,
    )

    output_file = save_results(results, args.output_dir)

    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    print(f"\nBenchmark complete: {completed} completed, {failed} failed")
    print(f"Results: {output_file}")


if __name__ == "__main__":
    main()
