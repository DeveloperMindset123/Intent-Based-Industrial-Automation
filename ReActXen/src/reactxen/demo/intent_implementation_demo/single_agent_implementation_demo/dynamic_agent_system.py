"""
Dynamic Agent System - Allows agents to create sub-agents and tools dynamically.
"""
import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent_tool_wrapper import AgentTool
from sub_agent_base import create_sub_agent


class CodeExecutionTool(BaseTool):
    """Tool for executing Python code dynamically."""
    
    name: str = "execute_python_code"
    description: str = """Execute Python code dynamically. Use this to:
    - Perform calculations
    - Process data
    - Create helper functions
    - Generate dynamic tools
    
    Input: JSON with 'code' key containing Python code as string.
    Returns: Execution result or error message.
    """
    
    class CodeInput(BaseModel):
        code: str = Field(description="Python code to execute")
        return_result: bool = Field(default=True, description="Whether to return the result")
    
    args_schema: type[BaseModel] = CodeInput
    
    def _run(self, code: str, return_result: bool = True) -> str:
        """Execute Python code safely."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': __builtins__,
                'json': json,
                'os': os,
                'sys': sys,
                'Path': Path,
            }
            
            # Execute code
            exec_result = exec(code, safe_globals)
            
            if return_result and 'result' in safe_globals:
                return str(safe_globals['result'])
            elif return_result:
                return "Code executed successfully (no result variable set)"
            else:
                return "Code executed successfully"
                
        except Exception as e:
            return f"Error executing code: {str(e)}"


class CreateSubAgentTool(BaseTool):
    """Tool for creating sub-agents dynamically."""
    
    name: str = "create_sub_agent"
    description: str = """Create a new sub-agent with specified role and tools.
    
    Input: JSON with:
    - agent_name: Name of the agent
    - role: Role description
    - workflow: Workflow steps
    - tools_description: Description of tools needed
    - model_id: Optional model ID (defaults to parent's model ID)
    
    Returns: Agent tool that can be used by the parent agent.
    """
    
    class CreateAgentInput(BaseModel):
        agent_name: str = Field(description="Name of the sub-agent")
        role: str = Field(description="Role description for the agent")
        workflow: str = Field(description="Workflow steps the agent should follow")
        tools_description: str = Field(description="Description of tools the agent needs")
        model_id: Optional[int] = Field(default=None, description="Model ID for the agent")
    
    args_schema: type[BaseModel] = CreateAgentInput
    
    def __init__(self, parent_model_id: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.parent_model_id = parent_model_id
    
    def _run(
        self,
        agent_name: str,
        role: str,
        workflow: str,
        tools_description: str,
        model_id: Optional[int] = None
    ) -> str:
        """Create a sub-agent dynamically."""
        try:
            # Use parent model ID if not specified
            if model_id is None:
                model_id = self.parent_model_id
            
            # Generate tools based on description
            tools = self._generate_tools_from_description(tools_description)
            
            # Create the agent
            agent = create_sub_agent(
                question=f"Execute tasks as {role}",
                key=agent_name.lower().replace(" ", "_"),
                role=role,
                workflow=workflow,
                tools=tools,
                max_steps=15,
                react_llm_model_id=model_id
            )
            
            # Wrap as tool
            agent_tool = AgentTool(
                agent=agent,
                name=agent_name.lower().replace(" ", "_"),
                description=f"{role}: {workflow}"
            )
            
            # Store in registry (simplified - in production, use proper registry)
            if not hasattr(self, '_agent_registry'):
                self._agent_registry = {}
            self._agent_registry[agent_name] = agent_tool
            
            return f"✅ Created sub-agent '{agent_name}' with {len(tools)} tools. Use '{agent_name.lower().replace(' ', '_')}' to delegate tasks."
            
        except Exception as e:
            return f"Error creating sub-agent: {str(e)}"
    
    def _generate_tools_from_description(self, description: str) -> List[BaseTool]:
        """Generate tools based on description (simplified - can be enhanced with LLM)."""
        tools = []
        
        # Import common tools
        try:
            from shared_utils import get_dataset_tools, get_search_tools, create_watsonx_tools
            from ml_framework_tools import create_ml_framework_tools
            
            # Add dataset tools if mentioned
            if any(keyword in description.lower() for keyword in ['dataset', 'data', 'load']):
                tools.extend(get_dataset_tools())
            
            # Add search tools if mentioned
            if any(keyword in description.lower() for keyword in ['search', 'lookup', 'find']):
                tools.extend(get_search_tools())
            
            # Add WatsonX tools if mentioned
            if any(keyword in description.lower() for keyword in ['watsonx', 'model', 'train']):
                tools.extend(create_watsonx_tools())
            
            # Add ML framework tools if mentioned
            if any(keyword in description.lower() for keyword in ['ml', 'machine learning', 'sklearn', 'pytorch', 'tensorflow']):
                tools.extend(create_ml_framework_tools())
                
        except ImportError as e:
            pass
        
        return tools


class CreateDynamicToolTool(BaseTool):
    """Tool for creating custom tools dynamically."""
    
    name: str = "create_dynamic_tool"
    description: str = """Create a custom tool dynamically based on Python code.
    
    Input: JSON with:
    - tool_name: Name of the tool
    - tool_description: Description of what the tool does
    - code: Python code that implements the tool's _run method
    - parameters: List of parameter names and types
    
    Returns: Confirmation that tool was created.
    """
    
    class CreateToolInput(BaseModel):
        tool_name: str = Field(description="Name of the tool")
        tool_description: str = Field(description="Description of the tool")
        code: str = Field(description="Python code implementing the tool")
        parameters: Optional[str] = Field(default="{}", description="JSON string of parameters")
    
    args_schema: type[BaseModel] = CreateToolInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dynamic_tools = {}
    
    def _run(
        self,
        tool_name: str,
        tool_description: str,
        code: str,
        parameters: str = "{}"
    ) -> str:
        """Create a dynamic tool."""
        try:
            # Parse parameters
            params = json.loads(parameters) if parameters else {}
            
            # Create tool class dynamically
            tool_code = f"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

class {tool_name}Tool(BaseTool):
    name: str = "{tool_name}"
    description: str = "{tool_description}"
    
    class ToolInput(BaseModel):
"""
            # Add parameters to input schema
            for param_name, param_type in params.items():
                tool_code += f'        {param_name}: {param_type} = Field(description="Parameter {param_name}")\n'
            
            tool_code += f"""
    args_schema = ToolInput
    
    def _run(self, {', '.join(params.keys()) if params else ''}):
{self._indent_code(code)}
"""
            
            # Execute tool creation
            exec_globals = {
                'BaseTool': BaseTool,
                'BaseModel': BaseModel,
                'Field': Field,
                'Optional': Optional,
            }
            exec(tool_code, exec_globals)
            
            # Store tool
            tool_class = exec_globals[f'{tool_name}Tool']
            self._dynamic_tools[tool_name] = tool_class()
            
            return f"✅ Created tool '{tool_name}'. It is now available for use."
            
        except Exception as e:
            return f"Error creating tool: {str(e)}"
    
    def _indent_code(self, code: str, indent: int = 8) -> str:
        """Indent code for proper formatting."""
        lines = code.split('\n')
        return '\n'.join(' ' * indent + line for line in lines)
    
    def get_dynamic_tools(self) -> List[BaseTool]:
        """Get all dynamically created tools."""
        return list(self._dynamic_tools.values())


def create_dynamic_agent_tools(parent_model_id: int = 15) -> List[BaseTool]:
    """Create tools for dynamic agent system."""
    return [
        CodeExecutionTool(),
        CreateSubAgentTool(parent_model_id=parent_model_id),
        CreateDynamicToolTool(),
    ]

