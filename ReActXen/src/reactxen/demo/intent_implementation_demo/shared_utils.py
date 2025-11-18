"""
Shared utilities for hierarchical agent system.
Common functions and classes reused across modules.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
def load_env_vars():
    """Load environment variables from common locations."""
    env_paths = [
        Path(__file__).parent.parent.parent.parent.parent.parent / ".env",
        Path(__file__).parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return
    load_dotenv(override=False)

# Add source path
def setup_paths():
    """Setup Python paths for imports."""
    reactxen_src = Path(__file__).parent.parent.parent.parent
    if str(reactxen_src) not in sys.path:
        sys.path.insert(0, str(reactxen_src))

# Initialize on import
load_env_vars()
setup_paths()


def get_dataset_tools():
    """Get dataset tools with fallback."""
    try:
        # Try importing from our new dataset_tools module first
        from dataset_tools import create_dataset_tools
        return create_dataset_tools()
    except (ImportError, AttributeError):
        try:
            # Fallback to agent_implementation_hf
            from agent_implementation_hf import create_dataset_tools
            return create_dataset_tools()
        except (ImportError, AttributeError):
            return []


def get_search_tools():
    """Get search tools with fallback."""
    tools = []
    try:
        from agent_implementation_hf import SearchToolWrapper
        if SearchToolWrapper:
            tools.append(SearchToolWrapper())
    except ImportError:
        pass
    
    try:
        from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun
        brave_api_key = os.environ.get("BRAVE_API_KEY", "")
        if brave_api_key:
            tools.append(BraveSearch.from_api_key(api_key=brave_api_key))
        else:
            tools.append(DuckDuckGoSearchRun())
    except:
        pass
    
    return tools


def create_agent_prompt_template(role: str, workflow: str, tool_desc: str) -> str:
    """Create a standardized agent prompt template."""
    return f"""You are a {role}.

YOUR ROLE:
{role}

WORKFLOW:
{workflow}

TOOLS:
{tool_desc}

IMPORTANT:
- Use the exact tool names from the TOOLS list above
- Action Input must be valid JSON format
- For tools with no parameters, use empty JSON object
- For tools with parameters, use JSON with parameter names as keys

FORMAT:
Question: {{question}}
Thought: think about next step
Action: tool name (must match exactly from TOOLS list)
Action Input: JSON format (empty JSON object for no parameters, or JSON with parameter names as keys)
Observation: result
... (continue until complete)
Thought: I have completed the task
Action: Finish
Action Input: Your final answer summarizing all results

Begin!
Question: {{question}}
{{scratchpad}}"""


def get_tool_descriptions(tools: List) -> tuple:
    """Generate tool names and descriptions."""
    tool_names = [tool.name for tool in tools]
    tool_desc_parts = []
    for i, tool in enumerate(tools):
        try:
            if tool.args_schema is not None:
                if hasattr(tool.args_schema, 'model_json_schema'):
                    schema = tool.args_schema.model_json_schema()
                else:
                    schema = tool.args_schema.schema()
                props = schema.get('properties', {})
                params = list(props.keys()) if props else []
                tool_desc_parts.append(f"({i+1}) {tool.name}[{', '.join(params)}]: {tool.description}")
            else:
                tool_desc_parts.append(f"({i+1}) {tool.name}[]: {tool.description}")
        except Exception:
            tool_desc_parts.append(f"({i+1}) {tool.name}[]: {tool.description}")
    
    tool_desc = "\n".join(tool_desc_parts)
    return tool_names, tool_desc

