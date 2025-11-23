"""
Action Normalizer - Handles action name normalization for ReactAgent.
Fixes issues like "Tool: tool_name" -> "tool_name"
"""
import re
from typing import Optional


def normalize_action_name(action_name: str, valid_tool_names: list) -> str:
    """
    Normalize action name to match tool names.
    
    Fixes common issues:
    - "Tool: quick_data_summary" -> "quick_data_summary"
    - "tool: quick_data_summary" -> "quick_data_summary"
    - "  quick_data_summary  " -> "quick_data_summary"
    
    Args:
        action_name: The action name from LLM output
        valid_tool_names: List of valid tool names
        
    Returns:
        Normalized action name, or original if no match found
    """
    if not action_name or not isinstance(action_name, str):
        return action_name or ""
    
    # Strip whitespace
    action_name = action_name.strip()
    
    # Remove "Tool:" prefix (case-insensitive)
    action_name = re.sub(r'^Tool\s*:\s*', '', action_name, flags=re.IGNORECASE)
    action_name = action_name.strip()
    
    # Remove "tool:" prefix (lowercase)
    action_name = re.sub(r'^tool\s*:\s*', '', action_name)
    action_name = action_name.strip()
    
    # Try exact match first (case-insensitive)
    for tool_name in valid_tool_names:
        if tool_name.lower() == action_name.lower():
            return tool_name
    
    # Try partial match (contains)
    for tool_name in valid_tool_names:
        if tool_name.lower() in action_name.lower() or action_name.lower() in tool_name.lower():
            return tool_name
    
    # Return normalized (but unmatched) name
    return action_name


def is_valid_action(action_name: str, valid_tool_names: list) -> bool:
    """
    Check if action name matches any valid tool name.
    
    Args:
        action_name: The action name to check
        valid_tool_names: List of valid tool names
        
    Returns:
        True if action matches a tool name, False otherwise
    """
    normalized = normalize_action_name(action_name, valid_tool_names)
    return normalized.lower() in [name.lower() for name in valid_tool_names]

