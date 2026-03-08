"""Root agent that orchestrates sub-agents based on scenario type."""

import sys
from pathlib import Path

# Add ReActXen/src to path so reactxen can be imported
# From: ReActXen/src/reactxen/demo/.../agents/root_agent.py
# To:   ReActXen/src/ (go up 5 levels)
_reactxen_src = Path(__file__).parent.parent.parent.parent.parent.parent

# print(f"current reactxen source: {_reactxen_src}")
if str(_reactxen_src) not in sys.path:
    sys.path.insert(0, str(_reactxen_src))

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from agents.rul_agent import RULAgent
from agents.fault_agent import FaultAgent


class RootAgent:
    """Root agent that routes scenarios to appropriate sub-agents."""

    def __init__(
        self, scenario: dict, model_id: int = 8, model_source: str = "watsonx"
    ):
        self.scenario = scenario
        self.model_id = model_id
        self.model_source = model_source

        # self.classification_type could be either "RUL prediction" or "fault classification", reference my_scenarios.json to see the breakdown
        self.classification_type = scenario.get("classification_type", "Unknown")

    def run(self) -> dict:
        """Route scenario to appropriate sub-agent and execute."""
        if "RUL" in self.classification_type:
            agent = RULAgent(self.scenario, self.model_id, self.model_source)
            return agent.run()
        elif "Fault" in self.classification_type:
            agent = FaultAgent(self.scenario, self.model_id, self.model_source)
            return agent.run()
        else:
            tools = self._get_tools()
            agent = create_reactxen_agent(
                question=self.scenario.get("fuzzy_description", ""),
                key=str(self.scenario.get("ground_truth", {})),
                tools=tools,
                react_llm_model_id=self.model_id,
            )
            agent.run()
            return {"result": agent.answer, "scenario": self.scenario["task_id"]}

    def _get_tools(self):
        """Get all available tools."""
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
            WebSearchTool(),
        ]
