"""Single agent with access to ALL tools from both MCP servers."""

import sys
from pathlib import Path

# Add ReActXen/src to path for reactxen imports
# From: .../intent_implementation_demo/single_agent_implementation/agent.py
# To:   .../ReActXen/src/
_reactxen_src = Path(__file__).parent.parent.parent.parent.parent
if str(_reactxen_src) not in sys.path:
    sys.path.insert(0, str(_reactxen_src))

# Add project root for tools imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent


class SingleAgent:
    """Single agent with access to ALL tools from both MCP servers."""

    def __init__(self, model_id: int = 8, model_source: str = "watsonx"):
        self.model_id = model_id
        self.model_source = model_source

    def _get_all_tools(self) -> list:
        """Get all tools as a flat list (no routing)."""
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
            CalculateMaintenanceCostTool,
            CalculateFailureCostTool,
            OptimizeMaintenanceScheduleTool,
            AssessSafetyRiskTool,
            CheckComplianceTool,
            GenerateSafetyRecommendationsTool,
        )
        from tools.web_search_tool import WebSearchTool

        return [
            # Data tools
            LoadDatasetTool(),
            LoadGroundTruthTool(),
            # RUL tools
            TrainRULModelTool(),
            PredictRULTool(),
            # Fault tools
            TrainFaultClassifierTool(),
            ClassifyFaultsTool(),
            # Metric tools
            CalculateMAETool(),
            CalculateRMSETool(),
            VerifyGroundTruthTool(),
            CalculateAccuracyTool(),
            VerifyClassificationTool(),
            # Engine Health tools
            AnalyzeEngineSignalsTool(),
            AssessComponentHealthTool(),
            DiagnoseTimingIssuesTool(),
            DetectDegradationTrendTool(),
            # Cost-Benefit tools
            CalculateMaintenanceCostTool(),
            CalculateFailureCostTool(),
            OptimizeMaintenanceScheduleTool(),
            # Safety tools
            AssessSafetyRiskTool(),
            CheckComplianceTool(),
            GenerateSafetyRecommendationsTool(),
            # Web search
            WebSearchTool(),
        ]

    def run(self, scenario: dict) -> dict:
        """Execute a scenario using the single agent with all tools."""
        tools = self._get_all_tools()
        question = scenario.get("input_question", "") or scenario.get("fuzzy_description", "")
        question += (
            "\n\nIMPORTANT: When calling tools, use this format:\n"
            "- Action: tool_name (just the name, no brackets or parameters)\n"
            '- Action Input: JSON object like {"param1": "value1", "param2": "value2"}\n'
            "DO NOT use formats like tool_name[param1, param2] or tool_name('param1', 'param2')"
        )

        agent = create_reactxen_agent(
            question=question,
            key=str(scenario.get("ground_truth", {})),
            tools=tools,
            react_llm_model_id=self.model_id,
        )
        agent.run()
        return {
            "task_id": scenario.get("task_id", "unknown"),
            "result": agent.answer,
            "dataset": scenario.get("dataset", ""),
            "classification_type": scenario.get("classification_type", ""),
            "agent_type": "single_agent",
        }
