"""
Predictive Maintenance Agent - Handles RUL prediction and risk assessment.
"""
from typing import List, Any
from shared_utils import get_dataset_tools, get_search_tools
from ml_framework_tools import create_ml_framework_tools
from sub_agent_base import create_sub_agent


def create_predictive_maintenance_agent(react_llm_model_id: int = 15) -> Any:
    """Create a Predictive Maintenance sub-agent for RUL prediction."""
    tools = []
    tools.extend(get_dataset_tools())
    
    # Add RUL prediction tools
    try:
        from tools_logic import PredictRULTool, GetEnginesAtRiskTool
        tools.extend([PredictRULTool(), GetEnginesAtRiskTool()])
    except ImportError:
        try:
            from tools_logic import create_watsonx_tools
            watsonx_tools = create_watsonx_tools()
            for tool in watsonx_tools:
                if tool.name in ["predict_rul", "get_engines_at_risk"]:
                    tools.append(tool)
        except:
            pass
    
    tools.extend(create_ml_framework_tools())
    tools.extend(get_search_tools())
    
    role = """Predictive Maintenance Agent specialized in RUL prediction and risk assessment.
- Predict Remaining Useful Life (RUL) for equipment
- Identify equipment at risk of failure
- Assess maintenance urgency
- Recommend maintenance schedules"""
    
    workflow = """1. If dataset not loaded, use load_dataset tool with dataset_name parameter
2. Use get_engines_at_risk tool with threshold=20 to identify equipment at risk (RUL <= 20 cycles)
3. Use predict_rul tool to get RUL predictions for equipment
4. Assess risk levels based on RUL values
5. Provide maintenance recommendations for equipment at risk"""
    
    return create_sub_agent(
        question="Identify equipment at risk and predict RUL",
        key="predictive_maintenance_agent",
        role=role,
        workflow=workflow,
        tools=tools,
        max_steps=12,
        react_llm_model_id=react_llm_model_id
    )

