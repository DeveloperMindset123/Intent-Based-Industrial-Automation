"""
Cost-Benefit Analysis Agent - Handles cost estimation and ROI analysis.
"""
from typing import List, Any
from shared_utils import get_search_tools
from sub_agent_base import create_sub_agent


def create_cost_benefit_agent(react_llm_model_id: int = 15) -> Any:
    """Create a Cost-Benefit Analysis sub-agent."""
    tools = []
    
    # Add cost estimation tools
    try:
        from tools_logic import CostEstimationTool, CostBenefitAnalysisTool
        tools.extend([CostEstimationTool(), CostBenefitAnalysisTool()])
    except ImportError:
        try:
            from tools_logic import CostEstimationTool
            tools.append(CostEstimationTool())
            try:
                from tools_logic import CostBenefitAnalysisTool
                tools.append(CostBenefitAnalysisTool())
            except ImportError:
                pass
        except ImportError:
            pass
    
    tools.extend(get_search_tools())
    
    role = """Cost-Benefit Analysis Agent specialized in maintenance cost estimation and ROI analysis.
- Estimate maintenance costs for equipment
- Perform cost-benefit analysis
- Calculate ROI for maintenance actions
- Provide budget recommendations"""
    
    workflow = """1. Get list of equipment requiring maintenance (from previous agent results or use estimate_maintenance_cost with engine_id)
2. For each equipment at risk, use estimate_maintenance_cost tool with:
   - engine_id: integer ID of the equipment
   - maintenance_type: "PREVENTIVE" or "CORRECTIVE" (defaults to CORRECTIVE if not provided)
   - estimated_hours: number of hours for maintenance
3. Calculate total budget requirements by summing all costs
4. Provide cost optimization recommendations and budget allocation"""
    
    return create_sub_agent(
        question="Estimate maintenance costs and perform cost-benefit analysis",
        key="cost_benefit_agent",
        role=role,
        workflow=workflow,
        tools=tools,
        max_steps=10,
        react_llm_model_id=react_llm_model_id
    )

