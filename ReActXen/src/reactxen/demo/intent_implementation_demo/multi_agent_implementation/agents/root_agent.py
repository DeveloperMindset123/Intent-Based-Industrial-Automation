"""Root agent that orchestrates sub-agents based on scenario type (5 categories)."""

import sys
from pathlib import Path

# Add ReActXen/src to path for reactxen imports
_reactxen_src = Path(__file__).parent.parent.parent.parent.parent.parent
if str(_reactxen_src) not in sys.path:
    sys.path.insert(0, str(_reactxen_src))

# Add project root for tools/agents imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from multi_agent_implementation.agents.rul_agent import RULAgent
from multi_agent_implementation.agents.fault_agent import FaultAgent
from multi_agent_implementation.agents.health_agent import HealthAgent
from multi_agent_implementation.agents.cost_agent import CostAgent
from multi_agent_implementation.agents.safety_agent import SafetyAgent


class RootAgent:
    """Root agent that routes scenarios to appropriate sub-agents (5 categories)."""

    def __init__(
        self, scenario: dict, model_id: int = 8, model_source: str = "watsonx"
    ):
        self.scenario = scenario
        self.model_id = model_id
        self.model_source = model_source
        self.classification_type = scenario.get("classification_type", "Unknown")

    def run(self) -> dict:
        """Route scenario to appropriate sub-agent and execute."""
        ctype = self.classification_type

        if "RUL" in ctype:
            agent = RULAgent(self.scenario, self.model_id, self.model_source)
            return agent.run()
        elif "Fault" in ctype:
            agent = FaultAgent(self.scenario, self.model_id, self.model_source)
            return agent.run()
        elif "Engine Health" in ctype:
            agent = HealthAgent(self.scenario, self.model_id, self.model_source)
            return agent.run()
        elif "Cost" in ctype:
            agent = CostAgent(self.scenario, self.model_id, self.model_source)
            return agent.run()
        elif "Safety" in ctype:
            agent = SafetyAgent(self.scenario, self.model_id, self.model_source)
            return agent.run()
        else:
            # Generic fallback with all tools
            tools = self._get_all_tools()
            question = self.scenario.get("input_question", "") or self.scenario.get("fuzzy_description", "")
            agent = create_reactxen_agent(
                question=question,
                key=str(self.scenario.get("ground_truth", {})),
                tools=tools,
                react_llm_model_id=self.model_id,
            )
            agent.run()
            return {
                "task_id": self.scenario.get("task_id", "unknown"),
                "result": agent.answer,
                "dataset": self.scenario.get("dataset", ""),
                "classification_type": self.classification_type,
                "agent_type": "multi_agent",
            }

    def _get_all_tools(self):
        """Get all available tools for fallback routing."""
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
            LoadDatasetTool(),
            LoadGroundTruthTool(),
            TrainRULModelTool(),
            PredictRULTool(),
            TrainFaultClassifierTool(),
            ClassifyFaultsTool(),
            CalculateMAETool(),
            CalculateRMSETool(),
            VerifyGroundTruthTool(),
            CalculateAccuracyTool(),
            VerifyClassificationTool(),
            AnalyzeEngineSignalsTool(),
            AssessComponentHealthTool(),
            DiagnoseTimingIssuesTool(),
            DetectDegradationTrendTool(),
            CalculateMaintenanceCostTool(),
            CalculateFailureCostTool(),
            OptimizeMaintenanceScheduleTool(),
            AssessSafetyRiskTool(),
            CheckComplianceTool(),
            GenerateSafetyRecommendationsTool(),
            WebSearchTool(),
        ]
