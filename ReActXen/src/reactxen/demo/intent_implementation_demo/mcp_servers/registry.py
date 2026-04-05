"""MCP Registry / Gateway — tool cataloging, skill advertisement, and routing.

Provides a central registry that:
1. Catalogs tools from multiple MCP servers with metadata
2. Supports skill-based discovery (find tools by capability)
3. Routes tool calls to the correct server
4. Tracks tool call metrics for evaluation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    """Registry entry for a single MCP tool."""

    name: str
    server: str
    description: str
    parameters: list[dict]
    input_schema: dict
    category: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call."""

    tool_name: str
    server: str
    timestamp: float
    duration_ms: float
    success: bool
    error: str = ""


class MCPRegistry:
    """Central registry for MCP tool cataloging and routing.

    Maintains a catalog of all tools across servers, supports
    tag-based discovery, and collects call metrics for evaluation.
    """

    # Category assignments for PHMForge tools
    TOOL_CATEGORIES = {
        "load_dataset": ("Data Loading", ["data", "dataset", "cmapss", "cwru"]),
        "load_ground_truth": ("Data Loading", ["data", "ground-truth", "rul"]),
        "train_rul_model": ("Model Training", ["training", "rul", "ml"]),
        "predict_rul": ("Prediction", ["prediction", "rul", "inference"]),
        "train_fault_classifier": ("Model Training", ["training", "fault", "classification"]),
        "classify_faults": ("Prediction", ["prediction", "fault", "classification"]),
        "calculate_mae": ("Metrics", ["evaluation", "mae", "rul"]),
        "calculate_rmse": ("Metrics", ["evaluation", "rmse", "rul"]),
        "verify_ground_truth": ("Metrics", ["verification", "rul"]),
        "calculate_accuracy": ("Metrics", ["evaluation", "accuracy", "classification"]),
        "verify_classification": ("Metrics", ["verification", "classification"]),
        "analyze_engine_signals": ("Engine Health", ["analysis", "signals", "engine"]),
        "assess_component_health": ("Engine Health", ["health", "component", "engine"]),
        "diagnose_timing_issues": ("Engine Health", ["diagnosis", "timing", "fault"]),
        "detect_degradation_trend": ("Engine Health", ["trend", "degradation", "prognostics"]),
        "calculate_maintenance_cost": ("Cost-Benefit", ["cost", "maintenance", "preventive"]),
        "calculate_failure_cost": ("Cost-Benefit", ["cost", "failure", "unplanned"]),
        "optimize_maintenance_schedule": ("Cost-Benefit", ["optimization", "schedule", "maintenance"]),
        "assess_safety_risk": ("Safety/Policy", ["safety", "risk", "rpn"]),
        "check_compliance": ("Safety/Policy", ["compliance", "standards", "sil"]),
        "generate_safety_recommendations": ("Safety/Policy", ["safety", "recommendations"]),
        "web_search": ("Utility", ["search", "web", "information"]),
    }

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self._metrics: list[ToolCallMetrics] = []

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def server_names(self) -> list[str]:
        return sorted({t.server for t in self._tools.values()})

    @property
    def categories(self) -> list[str]:
        return sorted({t.category for t in self._tools.values() if t.category})

    def register_tool(self, tool: ToolEntry) -> None:
        """Register a single tool in the catalog."""
        # Auto-assign category and tags if known
        if tool.name in self.TOOL_CATEGORIES:
            cat, tags = self.TOOL_CATEGORIES[tool.name]
            tool.category = cat
            tool.tags = tags
        self._tools[tool.name] = tool
        _log.debug("Registered tool: %s (server: %s)", tool.name, tool.server)

    def register_from_discovery(self, server_name: str, tools: list[dict]) -> int:
        """Register tools discovered from an MCP server's list_tools response."""
        count = 0
        for t in tools:
            entry = ToolEntry(
                name=t["name"],
                server=server_name,
                description=t.get("description", ""),
                parameters=t.get("parameters", []),
                input_schema=t.get("input_schema", {}),
            )
            self.register_tool(entry)
            count += 1
        return count

    def get_tool(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def find_by_category(self, category: str) -> list[ToolEntry]:
        """Find all tools in a given category."""
        return [t for t in self._tools.values() if t.category == category]

    def find_by_tag(self, tag: str) -> list[ToolEntry]:
        """Find all tools with a given tag."""
        return [t for t in self._tools.values() if tag in t.tags]

    def find_by_server(self, server: str) -> list[ToolEntry]:
        """Find all tools on a given server."""
        return [t for t in self._tools.values() if t.server == server]

    def search(self, query: str) -> list[ToolEntry]:
        """Search tools by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for t in self._tools.values():
            if (
                query_lower in t.name.lower()
                or query_lower in t.description.lower()
                or any(query_lower in tag for tag in t.tags)
            ):
                results.append(t)
        return results

    def record_call(
        self,
        tool_name: str,
        server: str,
        duration_ms: float,
        success: bool,
        error: str = "",
    ) -> None:
        """Record metrics for a tool call."""
        self._metrics.append(
            ToolCallMetrics(
                tool_name=tool_name,
                server=server,
                timestamp=time.time(),
                duration_ms=duration_ms,
                success=success,
                error=error,
            )
        )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Compute aggregate metrics across all recorded tool calls."""
        if not self._metrics:
            return {"total_calls": 0}

        total = len(self._metrics)
        successes = sum(1 for m in self._metrics if m.success)
        durations = [m.duration_ms for m in self._metrics]

        by_tool: dict[str, dict] = {}
        for m in self._metrics:
            if m.tool_name not in by_tool:
                by_tool[m.tool_name] = {"calls": 0, "successes": 0, "total_ms": 0.0}
            by_tool[m.tool_name]["calls"] += 1
            if m.success:
                by_tool[m.tool_name]["successes"] += 1
            by_tool[m.tool_name]["total_ms"] += m.duration_ms

        return {
            "total_calls": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_latency_ms": sum(durations) / len(durations),
            "p95_latency_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "by_tool": {
                name: {
                    "calls": info["calls"],
                    "success_rate": info["successes"] / info["calls"],
                    "avg_latency_ms": info["total_ms"] / info["calls"],
                }
                for name, info in by_tool.items()
            },
        }

    def export_catalog(self) -> dict:
        """Export the full tool catalog as a JSON-serializable dict."""
        catalog = {}
        for name, entry in sorted(self._tools.items()):
            catalog[name] = {
                "server": entry.server,
                "description": entry.description,
                "category": entry.category,
                "tags": entry.tags,
                "parameters": entry.parameters,
            }
        return catalog

    def format_for_llm(self) -> str:
        """Format the tool catalog as a text block suitable for LLM system prompts."""
        lines = []
        for category in self.categories:
            tools = self.find_by_category(category)
            if not tools:
                continue
            lines.append(f"\n## {category}")
            for t in sorted(tools, key=lambda x: x.name):
                params = ", ".join(
                    f"{p['name']}: {p.get('type', 'any')}"
                    for p in t.parameters
                )
                lines.append(f"- **{t.name}**({params}): {t.description}")
        return "\n".join(lines)
