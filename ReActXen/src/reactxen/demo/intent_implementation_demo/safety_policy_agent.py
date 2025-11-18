"""
Safety/Policy Agent - Handles safety protocols and compliance.
"""
from typing import List, Any
from shared_utils import get_search_tools
from sub_agent_base import create_sub_agent


def create_safety_policy_agent(react_llm_model_id: int = 15) -> Any:
    """Create a Safety/Policy sub-agent."""
    tools = get_search_tools()
    
    role = """Safety and Policy Agent specialized in industrial safety protocols and compliance.
- Provide OSHA safety protocols
- Recommend safety procedures for equipment maintenance
- Ensure regulatory compliance
- Identify safety risks"""
    
    workflow = """1. Use brave_search tool (or available search tool) to search for OSHA safety protocols for equipment maintenance
   - Action Input: JSON with query key containing search query
2. Analyze safety requirements for equipment with low RUL
3. Provide specific safety recommendations for each equipment at risk
4. Ensure compliance with OSHA regulations and industry standards"""
    
    return create_sub_agent(
        question="Provide safety protocols and recommendations",
        key="safety_policy_agent",
        role=role,
        workflow=workflow,
        tools=tools,
        max_steps=8,
        react_llm_model_id=react_llm_model_id
    )

