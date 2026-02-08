"""Intelligent Maintenance MCP Server — wraps Cost-Benefit and Safety/Policy tools."""

import asyncio
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent

# Ensure tools package is importable
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tools.analysis_tools import (
    CalculateMaintenanceCostTool,
    CalculateFailureCostTool,
    OptimizeMaintenanceScheduleTool,
    AssessSafetyRiskTool,
    CheckComplianceTool,
    GenerateSafetyRecommendationsTool,
)
from tools.web_search_tool import WebSearchTool

# Tool registry: name -> BaseTool instance
TOOL_REGISTRY = {
    # Cost-Benefit tools
    "calculate_maintenance_cost": CalculateMaintenanceCostTool(),
    "calculate_failure_cost": CalculateFailureCostTool(),
    "optimize_maintenance_schedule": OptimizeMaintenanceScheduleTool(),
    # Safety/Policy tools
    "assess_safety_risk": AssessSafetyRiskTool(),
    "check_compliance": CheckComplianceTool(),
    "generate_safety_recommendations": GenerateSafetyRecommendationsTool(),
    # Web search
    "web_search": WebSearchTool(),
}

server = Server("maintenance-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all maintenance tools."""
    tools = []
    for name, tool_instance in TOOL_REGISTRY.items():
        schema = tool_instance.args_schema.model_json_schema() if tool_instance.args_schema else {}
        tools.append(
            Tool(
                name=name,
                description=tool_instance.description,
                inputSchema=schema,
            )
        )
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a maintenance tool by name."""
    if name not in TOOL_REGISTRY:
        return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    tool_instance = TOOL_REGISTRY[name]
    try:
        result = tool_instance._run(**arguments)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {e}")]


async def main():
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
