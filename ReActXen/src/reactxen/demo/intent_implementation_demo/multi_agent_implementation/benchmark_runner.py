"""Benchmark runner for multi-agent mode."""

import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add project root for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from multi_agent_implementation.agents.root_agent import RootAgent


def load_scenarios(scenario_file: str) -> list:
    """Load scenarios from JSON file."""
    with open(scenario_file, "r") as f:
        data = json.load(f)
    return data.get("pdm_scenarios", [])


def run_benchmark(
    scenario_file: str,
    model_id: int = 8,
    model_source: str = "watsonx",
    limit: int = None,
) -> list:
    """Run benchmark on scenarios using multi-agent mode."""
    scenarios = load_scenarios(scenario_file)
    if limit:
        scenarios = scenarios[:limit]

    results = []

    for scenario in scenarios:
        task_id = scenario.get("task_id", "unknown")
        category = scenario.get("classification_type", "Unknown")
        dataset = scenario.get("dataset", "")

        print(f"[MultiAgent] Running {task_id} ({category}, {dataset})...")
        start = time.time()

        try:
            root_agent = RootAgent(
                scenario, model_id=model_id, model_source=model_source
            )
            result = root_agent.run()
            result["execution_time"] = round(time.time() - start, 2)
            result["status"] = "completed"
            results.append(result)
        except Exception as e:
            results.append({
                "task_id": task_id,
                "error": str(e),
                "status": "failed",
                "classification_type": category,
                "dataset": dataset,
                "agent_type": "multi_agent",
                "execution_time": round(time.time() - start, 2),
            })

    return results


def save_results(results: list, output_dir: str = None) -> str:
    """Save results to timestamped JSON file."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"multi_agent_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    return str(output_file)
