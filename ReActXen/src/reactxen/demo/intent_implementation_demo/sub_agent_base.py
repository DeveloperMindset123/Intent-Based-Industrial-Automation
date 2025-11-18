"""
Sub-Agent Base - Common utilities for creating sub-agents.
"""
from typing import List, Any
from langchain_core.prompts import PromptTemplate
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from shared_utils import get_tool_descriptions, create_agent_prompt_template


def create_sub_agent(
    question: str,
    key: str,
    role: str,
    workflow: str,
    tools: List,
    max_steps: int = 12,
    react_llm_model_id: int = 15,
    **kwargs
) -> Any:
    """Create a sub-agent with standardized configuration."""
    tool_names, tool_desc = get_tool_descriptions(tools)
    
    agent_prompt = PromptTemplate(
        input_variables=["question", "tool_desc", "scratchpad"],
        template=create_agent_prompt_template(role, workflow, tool_desc)
    )
    
    agent_config = {
        "question": question,
        "key": key,
        "max_steps": max_steps,
        "agent_prompt": agent_prompt,
        "tools": tools,
        "tool_names": tool_names,
        "tool_desc": tool_desc,
        "react_llm_model_id": react_llm_model_id,
        "reflect_llm_model_id": react_llm_model_id,
        "actionstyle": "Text",
        "reactstyle": "thought_and_act_together",
        "max_retries": 2,  # Increased retries for sub-agents
        "num_reflect_iteration": 2,  # Increased reflection iterations
        "early_stop": False,
        "debug": False,
    }
    
    agent_config.update(kwargs)
    return create_reactxen_agent(**agent_config)

