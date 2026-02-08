"""Fault Classification Agent for fault detection and classification tasks."""

import sys
from pathlib import Path

# Add ReActXen/src to path for reactxen imports
_reactxen_src = Path(__file__).parent.parent.parent.parent.parent.parent
if str(_reactxen_src) not in sys.path:
    sys.path.insert(0, str(_reactxen_src))

# Add project root for tools imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent


class FaultAgent:
    """Agent specialized for fault classification tasks."""

    def __init__(
        self, scenario: dict, model_id: int = 8, model_source: str = "watsonx"
    ):
        self.scenario = scenario
        self.model_id = model_id
        self.model_source = model_source

    def run(self) -> dict:
        """Execute fault classification scenario."""
        tools = self._get_fault_tools()
        question = self.scenario.get("input_question", "") or self.scenario.get("fuzzy_description", "")
        question += (
            "\n\nIMPORTANT: When calling tools, use this format:\n"
            "- Action: tool_name (just the name, no brackets or parameters)\n"
            '- Action Input: JSON object like {"param1": "value1", "param2": "value2"}\n'
            "DO NOT use formats like tool_name[param1, param2] or tool_name('param1', 'param2')"
        )

        agent = create_reactxen_agent(
            question=question,
            key=str(self.scenario.get("ground_truth", {})),
            tools=tools,
            react_llm_model_id=self.model_id,
        )
        agent.run()
        return {
            "task_id": self.scenario["task_id"],
            "result": agent.answer,
            "dataset": self.scenario.get("dataset", ""),
            "classification_type": "Fault Classification",
            "agent_type": "multi_agent",
        }

    def _get_fault_tools(self):
        """Get fault classification tools."""
        from tools.data_tools import LoadDatasetTool
        from tools.model_tools import TrainFaultClassifierTool, ClassifyFaultsTool
        from tools.metric_tools import CalculateAccuracyTool, VerifyClassificationTool
        from tools.web_search_tool import WebSearchTool

        return [
            LoadDatasetTool(),
            TrainFaultClassifierTool(),
            ClassifyFaultsTool(),
            CalculateAccuracyTool(),
            VerifyClassificationTool(),
            WebSearchTool(),
        ]
