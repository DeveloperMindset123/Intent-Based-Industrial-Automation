"""MCP Evaluation Metrics for PHMForge benchmark.

Measures MCP-specific quality dimensions beyond task accuracy:
1. Tool discovery latency — how fast can a client enumerate available tools
2. Tool call success rate — fraction of calls that return without error
3. Schema compliance — do tool args match declared input schemas
4. Routing accuracy — does the client route to the correct server
5. Context efficiency — how much context budget is consumed per scenario
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class MCPEvalResult:
    """Evaluation result for a single scenario run."""

    scenario_id: str
    task_accuracy: float
    tool_discovery_latency_ms: float
    tool_calls: int
    tool_successes: int
    tool_failures: int
    schema_violations: int
    routing_errors: int
    context_chars_used: int
    total_latency_ms: float
    tool_call_details: list[dict] = field(default_factory=list)

    @property
    def tool_success_rate(self) -> float:
        return self.tool_successes / self.tool_calls if self.tool_calls > 0 else 0.0

    @property
    def schema_compliance_rate(self) -> float:
        return 1.0 - (self.schema_violations / self.tool_calls) if self.tool_calls > 0 else 1.0

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "task_accuracy": self.task_accuracy,
            "mcp_metrics": {
                "tool_discovery_latency_ms": self.tool_discovery_latency_ms,
                "tool_calls": self.tool_calls,
                "tool_success_rate": self.tool_success_rate,
                "schema_compliance_rate": self.schema_compliance_rate,
                "routing_errors": self.routing_errors,
                "context_chars_used": self.context_chars_used,
                "total_latency_ms": self.total_latency_ms,
            },
            "tool_call_details": self.tool_call_details,
        }


def validate_tool_args(tool_name: str, args: dict, schema: dict) -> list[str]:
    """Validate tool arguments against the declared JSON schema.

    Returns a list of violation descriptions (empty if valid).
    """
    violations = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Check required fields
    for field_name in required:
        if field_name not in args:
            violations.append(f"Missing required field: {field_name}")

    # Check for unknown fields
    if properties:
        for key in args:
            if key not in properties:
                violations.append(f"Unknown field: {key}")

    # Basic type checking
    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for key, value in args.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type and expected_type in type_map:
                if not isinstance(value, type_map[expected_type]):
                    violations.append(
                        f"Type mismatch for {key}: expected {expected_type}, got {type(value).__name__}"
                    )

    return violations


class MCPEvaluator:
    """Evaluator that collects MCP-specific metrics during benchmark runs.

    Usage:
        evaluator = MCPEvaluator()
        evaluator.start_scenario("pdm_rul_001")

        # Record tool discovery
        evaluator.record_discovery(latency_ms=150.0)

        # Record each tool call
        evaluator.record_tool_call(
            tool_name="load_dataset",
            args={"dataset_name": "CMAPSS_FD001"},
            schema={...},
            success=True,
            latency_ms=50.0,
        )

        # Finish and get results
        result = evaluator.finish_scenario(task_accuracy=0.85)
    """

    def __init__(self) -> None:
        self._results: list[MCPEvalResult] = []
        self._current_scenario: str | None = None
        self._scenario_start: float = 0
        self._discovery_latency: float = 0
        self._call_details: list[dict] = []
        self._schema_violations: int = 0
        self._routing_errors: int = 0
        self._context_chars: int = 0

    @property
    def results(self) -> list[MCPEvalResult]:
        return list(self._results)

    def start_scenario(self, scenario_id: str) -> None:
        """Begin evaluation for a new scenario."""
        self._current_scenario = scenario_id
        self._scenario_start = time.time()
        self._discovery_latency = 0
        self._call_details = []
        self._schema_violations = 0
        self._routing_errors = 0
        self._context_chars = 0

    def record_discovery(self, latency_ms: float) -> None:
        """Record tool discovery latency."""
        self._discovery_latency = latency_ms

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        schema: dict,
        success: bool,
        latency_ms: float,
        server: str = "",
        error: str = "",
    ) -> None:
        """Record metrics for a single tool call."""
        # Validate schema compliance
        violations = validate_tool_args(tool_name, args, schema) if schema else []
        self._schema_violations += len(violations)

        self._call_details.append({
            "tool": tool_name,
            "server": server,
            "success": success,
            "latency_ms": latency_ms,
            "schema_violations": violations,
            "error": error,
        })

    def record_routing_error(self) -> None:
        """Record a tool routing error (tool sent to wrong server)."""
        self._routing_errors += 1

    def record_context_usage(self, chars: int) -> None:
        """Record context window characters consumed."""
        self._context_chars = chars

    def finish_scenario(self, task_accuracy: float) -> MCPEvalResult:
        """Finalize the current scenario and return results."""
        total_latency = (time.time() - self._scenario_start) * 1000
        successes = sum(1 for d in self._call_details if d["success"])
        failures = len(self._call_details) - successes

        result = MCPEvalResult(
            scenario_id=self._current_scenario or "unknown",
            task_accuracy=task_accuracy,
            tool_discovery_latency_ms=self._discovery_latency,
            tool_calls=len(self._call_details),
            tool_successes=successes,
            tool_failures=failures,
            schema_violations=self._schema_violations,
            routing_errors=self._routing_errors,
            context_chars_used=self._context_chars,
            total_latency_ms=total_latency,
            tool_call_details=self._call_details,
        )
        self._results.append(result)
        self._current_scenario = None
        return result

    def aggregate_results(self) -> dict[str, Any]:
        """Compute aggregate metrics across all evaluated scenarios."""
        if not self._results:
            return {"scenarios_evaluated": 0}

        n = len(self._results)
        return {
            "scenarios_evaluated": n,
            "avg_task_accuracy": sum(r.task_accuracy for r in self._results) / n,
            "avg_tool_discovery_latency_ms": sum(r.tool_discovery_latency_ms for r in self._results) / n,
            "avg_tool_success_rate": sum(r.tool_success_rate for r in self._results) / n,
            "avg_schema_compliance_rate": sum(r.schema_compliance_rate for r in self._results) / n,
            "total_tool_calls": sum(r.tool_calls for r in self._results),
            "total_routing_errors": sum(r.routing_errors for r in self._results),
            "avg_context_chars": sum(r.context_chars_used for r in self._results) / n,
            "avg_total_latency_ms": sum(r.total_latency_ms for r in self._results) / n,
        }

    def export_results(self) -> list[dict]:
        """Export all scenario results as JSON-serializable dicts."""
        return [r.to_dict() for r in self._results]
