"""Prognostics MCP Server — wraps RUL, Fault Classification, and Engine Health tools."""

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

from tools.data_tools import LoadDatasetTool, LoadGroundTruthTool
from tools.model_tools import (
    TrainRULModelTool,
    PredictRULTool,
    TrainFaultClassifierTool,
    ClassifyFaultsTool,
)
from tools.metric_tools import (
    CalculateMAETool,
    CalculateRMSETool,
    VerifyGroundTruthTool,
    CalculateAccuracyTool,
    VerifyClassificationTool,
)
from tools.analysis_tools import (
    AnalyzeEngineSignalsTool,
    AssessComponentHealthTool,
    DiagnoseTimingIssuesTool,
    DetectDegradationTrendTool,
)

# Tool registry: name -> BaseTool instance
TOOL_REGISTRY = {
    # Data tools
    "load_dataset": LoadDatasetTool(),
    "load_ground_truth": LoadGroundTruthTool(),
    # RUL tools
    "train_rul_model": TrainRULModelTool(),
    "predict_rul": PredictRULTool(),
    # Fault tools
    "train_fault_classifier": TrainFaultClassifierTool(),
    "classify_faults": ClassifyFaultsTool(),
    # Metric tools
    "calculate_mae": CalculateMAETool(),
    "calculate_rmse": CalculateRMSETool(),
    "verify_ground_truth": VerifyGroundTruthTool(),
    "calculate_accuracy": CalculateAccuracyTool(),
    "verify_classification": VerifyClassificationTool(),
    # Engine Health tools
    "analyze_engine_signals": AnalyzeEngineSignalsTool(),
    "assess_component_health": AssessComponentHealthTool(),
    "diagnose_timing_issues": DiagnoseTimingIssuesTool(),
    "detect_degradation_trend": DetectDegradationTrendTool(),
}

server = Server("prognostics-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all prognostics tools."""
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
    """Execute a prognostics tool by name."""
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
