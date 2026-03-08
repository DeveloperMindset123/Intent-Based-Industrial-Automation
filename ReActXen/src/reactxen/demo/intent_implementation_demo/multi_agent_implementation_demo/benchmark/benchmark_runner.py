"""Benchmark runner for predictive maintenance scenarios. Limit number of scenarios to load, to test them"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from agents.root_agent import RootAgent


def load_scenarios(scenario_file: str) -> dict:
    """Load scenarios from JSON file."""
    with open(scenario_file, "r") as f:
        return json.load(f)


def run_benchmark(
    scenario_file: str,
    model_id: int = 8,
    model_source: str = "watsonx",
    limit: int = None,
):
    """Run benchmark on scenarios with dynamic model loading."""
    data = load_scenarios(scenario_file)
    print(f"===Loaded scenario files===")
    scenarios = data.get("pdm_scenarios", [])
    print(scenarios)
    if limit:
        scenarios = scenarios[:limit]

    results = []

    for scenario in scenarios:
        try:
            root_agent = RootAgent(
                scenario, model_id=model_id, model_source=model_source
            )
            result = root_agent.run()
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "task_id": scenario.get("task_id"),
                    "error": str(e),
                    "status": "failed",
                }
            )

    return results


if __name__ == "__main__":
    scenario_path = (
        Path(__file__).parent.parent.parent / "scenario" / "my_scenarios.json"
    )
    results = run_benchmark(str(scenario_path), limit=2)
    print(json.dumps(results, indent=2))
