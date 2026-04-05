"""MCP Client for PHMForge — tool discovery, routing, and multi-server orchestration.

Connects to prognostics and maintenance MCP servers via stdio transport,
following the AssetOpsBench client pattern (mcp.client.stdio).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_SERVER_DIR = Path(__file__).parent
_PROJECT_ROOT = Path(__file__).parent.parent

# Server registry: name -> module path for stdio invocation
DEFAULT_SERVERS: dict[str, str] = {
    "prognostics": str(_SERVER_DIR / "prognostics_server.py"),
    "maintenance": str(_SERVER_DIR / "maintenance_server.py"),
}

# Tool-to-server routing (built dynamically via discover_tools, or use static fallback)
_STATIC_TOOL_ROUTING: dict[str, str] = {
    # Prognostics server (15 tools)
    "load_dataset": "prognostics",
    "load_ground_truth": "prognostics",
    "train_rul_model": "prognostics",
    "predict_rul": "prognostics",
    "train_fault_classifier": "prognostics",
    "classify_faults": "prognostics",
    "calculate_mae": "prognostics",
    "calculate_rmse": "prognostics",
    "verify_ground_truth": "prognostics",
    "calculate_accuracy": "prognostics",
    "verify_classification": "prognostics",
    "analyze_engine_signals": "prognostics",
    "assess_component_health": "prognostics",
    "diagnose_timing_issues": "prognostics",
    "detect_degradation_trend": "prognostics",
    # Maintenance server (7 tools)
    "calculate_maintenance_cost": "maintenance",
    "calculate_failure_cost": "maintenance",
    "optimize_maintenance_schedule": "maintenance",
    "assess_safety_risk": "maintenance",
    "check_compliance": "maintenance",
    "generate_safety_recommendations": "maintenance",
    "web_search": "maintenance",
}


def _make_stdio_params(server_path: str) -> "StdioServerParameters":
    """Build StdioServerParameters for a server script."""
    from mcp import StdioServerParameters

    return StdioServerParameters(
        command="python",
        args=[server_path],
        cwd=str(_PROJECT_ROOT),
    )


async def _connect_and_list(server_path: str) -> list[dict]:
    """Connect to an MCP server via stdio and list its tools with schemas."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parameters = [
                    {
                        "name": k,
                        "type": v.get("type", "any"),
                        "required": k in required,
                        "description": v.get("description", ""),
                    }
                    for k, v in props.items()
                ]
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": parameters,
                    "input_schema": schema,
                })
            return tools


async def _connect_and_call(server_path: str, tool_name: str, args: dict) -> str:
    """Connect to an MCP server via stdio and call a single tool."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return "\n".join(
                getattr(item, "text", str(item)) for item in result.content
            )


class MCPClient:
    """Multi-server MCP client with tool discovery and automatic routing.

    Usage:
        client = MCPClient()
        await client.discover_tools()  # optional: populates tool catalog
        result = await client.call_tool("load_dataset", {"dataset_name": "CMAPSS_FD001"})
    """

    def __init__(
        self,
        servers: dict[str, str] | None = None,
    ) -> None:
        self._servers = servers or DEFAULT_SERVERS
        self._tool_routing: dict[str, str] = dict(_STATIC_TOOL_ROUTING)
        self._tool_catalog: dict[str, dict] = {}
        self._discovered = False

    @property
    def servers(self) -> dict[str, str]:
        return dict(self._servers)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tool_routing.keys())

    @property
    def tool_catalog(self) -> dict[str, dict]:
        return dict(self._tool_catalog)

    async def discover_tools(self) -> dict[str, list[dict]]:
        """Query all registered servers and build the tool catalog.

        Returns a dict of server_name -> list of tool info dicts.
        """
        all_tools: dict[str, list[dict]] = {}
        for name, path in self._servers.items():
            try:
                tools = await _connect_and_list(path)
                all_tools[name] = tools
                for t in tools:
                    self._tool_routing[t["name"]] = name
                    self._tool_catalog[t["name"]] = {
                        "server": name,
                        "description": t["description"],
                        "parameters": t["parameters"],
                        "input_schema": t.get("input_schema", {}),
                    }
                _log.info("Discovered %d tools from %s", len(tools), name)
            except Exception as exc:
                _log.warning("Failed to discover tools from %s: %s", name, exc)
                all_tools[name] = []

        self._discovered = True
        return all_tools

    def get_tool_server(self, tool_name: str) -> str | None:
        """Return the server name that hosts a given tool."""
        return self._tool_routing.get(tool_name)

    def get_tool_schema(self, tool_name: str) -> dict:
        """Return the input schema for a given tool."""
        info = self._tool_catalog.get(tool_name, {})
        return info.get("input_schema", {})

    def format_tool_descriptions(self) -> str:
        """Format all discovered tools as a text block for LLM prompts."""
        lines = []
        for server_name in sorted(self._servers):
            tools = [
                (name, info)
                for name, info in self._tool_catalog.items()
                if info.get("server") == server_name
            ]
            if not tools:
                continue
            lines.append(f"\n[{server_name} server]")
            for name, info in sorted(tools):
                params = ", ".join(
                    f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                    for p in info.get("parameters", [])
                )
                lines.append(f"  - {name}({params}): {info['description']}")
        return "\n".join(lines)

    async def call_tool(self, tool_name: str, args: dict | None = None) -> str:
        """Route a tool call to the correct server and return the result."""
        server_name = self._tool_routing.get(tool_name)
        if server_name is None:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Available: {self.tool_names}"
            )
        server_path = self._servers.get(server_name)
        if server_path is None:
            raise ValueError(f"Server '{server_name}' not registered")

        _log.info("Calling %s on %s with args: %s", tool_name, server_name, args)
        return await _connect_and_call(server_path, tool_name, args or {})

    async def call_tools_sequential(
        self, calls: list[tuple[str, dict]]
    ) -> list[str]:
        """Execute a sequence of tool calls, returning results in order."""
        results = []
        for tool_name, args in calls:
            result = await self.call_tool(tool_name, args)
            results.append(result)
        return results

    async def health_check(self) -> dict[str, bool]:
        """Check if each server is reachable by listing its tools."""
        status = {}
        for name, path in self._servers.items():
            try:
                tools = await _connect_and_list(path)
                status[name] = len(tools) > 0
            except Exception:
                status[name] = False
        return status


# ---------------------------------------------------------------------------
# Convenience: run from CLI for quick testing
# ---------------------------------------------------------------------------

async def _demo():
    """Demo: discover tools and call a sample tool."""
    logging.basicConfig(level=logging.INFO)
    client = MCPClient()

    print("Discovering tools...")
    catalog = await client.discover_tools()
    for server, tools in catalog.items():
        print(f"\n{server}: {len(tools)} tools")
        for t in tools:
            print(f"  - {t['name']}: {t['description'][:80]}")

    print(f"\nTool descriptions for LLM:\n{client.format_tool_descriptions()}")

    print("\nHealth check:")
    health = await client.health_check()
    for name, ok in health.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(_demo())
