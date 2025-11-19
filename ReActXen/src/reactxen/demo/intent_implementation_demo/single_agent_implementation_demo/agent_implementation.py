"""Simplified agentic implementation using only WatsonX API."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional

# Load environment variables
env_paths = [
    Path(__file__).parent.parent.parent.parent.parent.parent / ".env",
    Path(__file__).parent / ".env",
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=False)
        break

# Add source path
reactxen_src = Path(__file__).parent.parent.parent.parent
if str(reactxen_src) not in sys.path:
    sys.path.insert(0, str(reactxen_src))

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from tools_logic import create_watsonx_tools
from load_data import list_available_datasets, load_dataset_for_analysis

# Dataset tools
class EmptyInput(BaseModel):
    pass

class LoadDatasetInput(BaseModel):
    dataset_name: Optional[str] = Field(default=None, description="Dataset name to load")

class ListDatasetsTool(BaseTool):
    name: str = "list_datasets"
    description: str = "List available datasets."
    args_schema: Type[BaseModel] = EmptyInput
    
    def _run(self) -> str:
        datasets = list_available_datasets()
        if not datasets:
            return "No datasets found. Download datasets first."
        return f"Available datasets ({len(datasets)}):\n" + "\n".join([f"  - {d}" for d in datasets])

class LoadDatasetTool(BaseTool):
    name: str = "load_dataset"
    description: str = "Load a dataset for analysis."
    args_schema: Type[BaseModel] = LoadDatasetInput
    
    def _run(self, dataset_name: Optional[str] = None) -> str:
        if not dataset_name:
            datasets = list_available_datasets()
            if datasets:
                dataset_name = datasets[0]
            else:
                return "Error: No dataset name provided and no datasets available."
        return load_dataset_for_analysis(dataset_name)

def create_dataset_tools() -> list:
    return [ListDatasetsTool(), LoadDatasetTool()]

# Add search tool
try:
    from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun
    brave_api_key = os.environ.get("BRAVE_API_KEY", "")
    if brave_api_key:
        brave_search = BraveSearch.from_api_key(api_key=brave_api_key)
        all_tools = create_dataset_tools() + create_watsonx_tools() + [brave_search]
    else:
        try:
            duckduckgo_search = DuckDuckGoSearchRun()
            all_tools = create_dataset_tools() + create_watsonx_tools() + [duckduckgo_search]
        except:
            all_tools = create_dataset_tools() + create_watsonx_tools()
except:
    all_tools = create_dataset_tools() + create_watsonx_tools()

# Agent prompt
agent_prompt = PromptTemplate(
    input_variables=["question", "tool_desc", "scratchpad"],
    template="""You are an AI agent for predictive maintenance and RUL prediction using WatsonX.

CRITICAL INSTRUCTIONS:
- USE ALL AVAILABLE STEPS - You have max_steps available. Use them ALL. Do NOT stop early.
- If you stop before using all steps, you have FAILED. Continue until max_steps is reached.
- If data is missing, IMMEDIATELY use search tools (brave_search or duckduckgo_search) to find information.
- Make EDUCATED GUESSES based on industry standards when exact data isn't available.
- DO NOT give up - always provide a reasonable answer even if data is incomplete.
- If a tool says "Error" or "not available", use search tools to find alternative information.

WORKFLOW:
1. List datasets using list_datasets
2. Load dataset using load_dataset with dataset_name
3. If dataset missing info, use search tools (brave_search or duckduckgo_search) to find safety protocols, cost estimates, etc.
4. Initialize WatsonX using initialize_watsonx_api
5. Get models using get_chat_models_list
6. Set model using set_model_id
7. Train using train_model
8. Predict RUL using predict_rul
9. Get engines at risk using get_engines_at_risk with threshold=20
10. Use search tools if needed for safety recommendations or cost data
11. Estimate costs using estimate_maintenance_cost
12. Provide final answer with equipment IDs, safety recommendations, and cost estimates

SEARCH TOOL USAGE:
- Use brave_search or duckduckgo_search when you need:
  * OSHA safety protocols for equipment maintenance
  * Industry-standard maintenance costs
  * Safety recommendations for low RUL equipment
  * Cost estimation data
- Format: Provide your search terms as input to the search tool

EDUCATED GUESSES:
- If exact data isn't available, use industry standards:
  * Maintenance costs: $5,000-$15,000 for corrective action
  * Safety: Follow OSHA guidelines for equipment with RUL < 20 cycles
  * Priority: IMMEDIATE_GROUNDING for RUL ≤ 10, CORRECTIVE_ACTION for RUL ≤ 20

TOOLS:
{tool_desc}

FORMAT:
Question: {question}
Thought: think about next step
Action: tool name
Action Input: JSON or empty braces for no parameters
Observation: result
... (continue until you have complete answer - USE ALL STEPS)
Final Answer: Complete answer with equipment IDs, safety recommendations, and cost estimates

Begin!
Question: {question}
{scratchpad}"""
)

# Agent configuration
agent_config = {
    "question": "Which equipment from the loaded dataset are likely to fail in the next 20 cycles? Provide equipment IDs with safety recommendations and cost estimates.",
    "key": "rul_prediction_watsonx",
    "max_steps": 10,
    "agent_prompt": agent_prompt,
    "tools": all_tools,
    "tool_names": [tool.name for tool in all_tools],
    "tool_desc": "\n".join([f"({i+1}) {tool.name}: {tool.description}" for i, tool in enumerate(all_tools)]),
    "react_llm_model_id": 15,
    "reflect_llm_model_id": 15,
    "actionstyle": "Text",
    "reactstyle": "thought_and_act_together",
    "max_retries": 1,
    "num_reflect_iteration": 1,
    "early_stop": False,  # Don't stop early - use all steps
    "debug": False,
}
