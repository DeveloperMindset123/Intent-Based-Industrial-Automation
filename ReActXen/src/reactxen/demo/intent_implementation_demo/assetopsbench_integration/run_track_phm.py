"""PHMForge Benchmark Track — MCP-based Predictive Maintenance evaluation.

Runs PHMForge scenarios through MCP servers and collects both task accuracy
and MCP-specific evaluation metrics. Compatible with AssetOpsBench framework.

Usage:
    python run_track_phm.py --scenario_ids "pdm_rul_001,pdm_fault_001"
    python run_track_phm.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phmforge-track")

# Paths
_TRACK_DIR = Path(__file__).parent
_REPO_ROOT = _TRACK_DIR.parent.parent
_PHMFORGE_ROOT = (
    _REPO_ROOT
    / "ReActXen"
    / "src"
    / "reactxen"
    / "demo"
    / "intent_implementation_demo"
)
_SCENARIOS_FILE = _PHMFORGE_ROOT / "scenarios" / "phm_scenarios.json"
_RESULT_DIR = _TRACK_DIR / "track_result"

# Ensure imports work
sys.path.insert(0, str(_PHMFORGE_ROOT))
sys.path.insert(0, str(_PHMFORGE_ROOT / "mcp_servers"))


def load_scenarios(scenario_ids: list[str] | None = None) -> list[dict]:
    """Load PHMForge scenarios, optionally filtering by IDs."""
    with open(_SCENARIOS_FILE) as f:
        data = json.load(f)

    scenarios = data["pdm_scenarios"]
    if scenario_ids:
        scenarios = [s for s in scenarios if s["task_id"] in scenario_ids]
    return scenarios


async def run_scenario(scenario: dict, client: "MCPClient") -> dict:
    """Execute a single scenario through MCP tools and return trajectory."""
    from mcp_eval import MCPEvaluator, validate_tool_args

    evaluator = MCPEvaluator()
    evaluator.start_scenario(scenario["task_id"])

    # Discover tools
    t0 = time.time()
    catalog = await client.discover_tools()
    discovery_ms = (time.time() - t0) * 1000
    evaluator.record_discovery(discovery_ms)

    trajectory = []
    trajectory.append({
        "step": "discovery",
        "tools_found": sum(len(v) for v in catalog.values()),
        "latency_ms": discovery_ms,
    })

    # For each required tool, simulate a call with placeholder args
    # (In production, an LLM agent would generate real args)
    for tool_name in scenario.get("required_tools", []):
        tool_info = client.tool_catalog.get(tool_name, {})
        schema = tool_info.get("input_schema", {})

        t0 = time.time()
        try:
            # Build minimal valid args from schema
            args = _build_placeholder_args(schema)
            result = await client.call_tool(tool_name, args)
            latency_ms = (time.time() - t0) * 1000
            success = "error" not in result.lower() if isinstance(result, str) else True

            evaluator.record_tool_call(
                tool_name=tool_name,
                args=args,
                schema=schema,
                success=success,
                latency_ms=latency_ms,
                server=client.get_tool_server(tool_name) or "",
            )
            trajectory.append({
                "step": f"call_{tool_name}",
                "args": args,
                "result": result[:500] if isinstance(result, str) else str(result)[:500],
                "success": success,
                "latency_ms": latency_ms,
            })
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            evaluator.record_tool_call(
                tool_name=tool_name,
                args={},
                schema=schema,
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
            trajectory.append({
                "step": f"call_{tool_name}",
                "error": str(e),
                "latency_ms": latency_ms,
            })

    # Compute task accuracy (placeholder — real evaluation needs ground truth comparison)
    task_accuracy = 0.0  # To be filled by grading engine
    eval_result = evaluator.finish_scenario(task_accuracy)

    return {
        "id": scenario["task_id"],
        "text": scenario.get("input_question", ""),
        "classification_type": scenario["classification_type"],
        "trajectory": trajectory,
        "mcp_metrics": eval_result.to_dict()["mcp_metrics"],
    }


def _build_placeholder_args(schema: dict) -> dict:
    """Build minimal placeholder arguments from a JSON schema."""
    args = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for name, prop in properties.items():
        if name not in required:
            continue
        ptype = prop.get("type", "string")
        if ptype == "string":
            args[name] = prop.get("default", "test_value")
        elif ptype == "integer":
            args[name] = prop.get("default", 1)
        elif ptype == "number":
            args[name] = prop.get("default", 1.0)
        elif ptype == "boolean":
            args[name] = prop.get("default", True)
    return args


async def main(scenario_ids: list[str] | None = None, run_all: bool = False):
    """Main entry point for the PHMForge benchmark track."""
    from mcp_client import MCPClient

    scenarios = load_scenarios(None if run_all else scenario_ids)
    logger.info("Loaded %d scenarios", len(scenarios))

    client = MCPClient()
    _RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, scenario in enumerate(scenarios):
        logger.info(
            "[%d/%d] Running %s (%s)",
            i + 1, len(scenarios),
            scenario["task_id"],
            scenario["classification_type"],
        )
        try:
            result = await run_scenario(scenario, client)
            results.append(result)
        except Exception as e:
            logger.error("Failed scenario %s: %s", scenario["task_id"], e)
            results.append({
                "id": scenario["task_id"],
                "text": scenario.get("input_question", ""),
                "error": str(e),
            })

    # Save results
    output_file = _RESULT_DIR / "phmforge_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_file)

    # Print summary
    total = len(results)
    successful = sum(1 for r in results if "error" not in r)
    logger.info("Summary: %d/%d scenarios completed successfully", successful, total)

    if successful > 0:
        metrics = [r.get("mcp_metrics", {}) for r in results if "mcp_metrics" in r]
        if metrics:
            avg_discovery = sum(m.get("tool_discovery_latency_ms", 0) for m in metrics) / len(metrics)
            avg_success = sum(m.get("tool_success_rate", 0) for m in metrics) / len(metrics)
            logger.info(
                "MCP Metrics — Avg discovery: %.0fms, Avg tool success rate: %.1f%%",
                avg_discovery, avg_success * 100,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHMForge Benchmark Track")
    parser.add_argument("--scenario_ids", type=str, help="Comma-separated scenario IDs")
    parser.add_argument("--all", action="store_true", help="Run all 75 scenarios")
    args = parser.parse_args()

    ids = args.scenario_ids.split(",") if args.scenario_ids else None
    asyncio.run(main(scenario_ids=ids, run_all=args.all))
