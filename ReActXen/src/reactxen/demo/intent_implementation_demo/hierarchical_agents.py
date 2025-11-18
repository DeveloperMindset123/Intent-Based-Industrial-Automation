"""
Hierarchical Agent System - Factory for creating all sub-agents as tools.
"""
from typing import List
from langchain_core.tools import BaseTool
from agent_tool_wrapper import AgentTool

from data_scientist_agent import create_data_scientist_agent
from predictive_maintenance_agent import create_predictive_maintenance_agent
from cost_benefit_agent import create_cost_benefit_agent
from safety_policy_agent import create_safety_policy_agent


def create_sub_agent_tools(react_llm_model_id: int = 15) -> List[BaseTool]:
    """Create all sub-agents and wrap them as tools for the root agent."""
    sub_agents = []
    
    # Create sub-agents
    data_scientist_agent = create_data_scientist_agent(react_llm_model_id)
    predictive_maintenance_agent = create_predictive_maintenance_agent(react_llm_model_id)
    cost_benefit_agent = create_cost_benefit_agent(react_llm_model_id)
    safety_policy_agent = create_safety_policy_agent(react_llm_model_id)
    
    # Wrap as tools
    sub_agents.append(AgentTool(
        agent=data_scientist_agent,
        name="data_scientist_agent",
        description="""Data Scientist Agent: Handles data loading, model training (scikit-learn, PyTorch, TensorFlow, HuggingFace, WatsonX), 
        model evaluation, and model selection. Use this agent for ML model development tasks."""
    ))
    
    sub_agents.append(AgentTool(
        agent=predictive_maintenance_agent,
        name="predictive_maintenance_agent",
        description="""Predictive Maintenance Agent: Handles RUL prediction, risk assessment, equipment failure prediction, 
        and maintenance scheduling. Use this agent to identify equipment at risk."""
    ))
    
    sub_agents.append(AgentTool(
        agent=cost_benefit_agent,
        name="cost_benefit_agent",
        description="""Cost-Benefit Analysis Agent: Handles maintenance cost estimation, ROI analysis, cost-benefit comparisons, 
        and budget planning. Use this agent for financial analysis."""
    ))
    
    sub_agents.append(AgentTool(
        agent=safety_policy_agent,
        name="safety_policy_agent",
        description="""Safety and Policy Agent: Handles OSHA compliance, safety protocols, regulatory requirements, 
        and safety recommendations. Use this agent for safety and compliance tasks."""
    ))
    
    return sub_agents
