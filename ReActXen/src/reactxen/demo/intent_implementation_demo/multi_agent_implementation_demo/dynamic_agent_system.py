"""
Dynamic Agent System - Allows agents to create sub-agents and tools dynamically.
Agents can write and execute their own code to reduce manual coding.
"""
import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with fallback for different directory structures
try:
    from agent_tool_wrapper import AgentTool
    from sub_agent_base import create_sub_agent
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent_tool_wrapper import AgentTool
    from sub_agent_base import create_sub_agent

# Ensure we're using the local sub_agent_base
import sys
from pathlib import Path
local_sub_agent_base = Path(__file__).parent / "sub_agent_base.py"
if local_sub_agent_base.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub_agent_base_local", local_sub_agent_base)
    sub_agent_base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub_agent_base_module)
    create_sub_agent = sub_agent_base_module.create_sub_agent


class CodeExecutionTool(BaseTool):
    """Tool for executing Python code dynamically."""
    
    name: str = "execute_python_code"
    description: str = """Execute Python code dynamically. Use this to:
    - Perform calculations
    - Process data
    - Create helper functions
    - Generate dynamic tools
    - Write and execute any Python code needed
    
    CRITICAL INPUT FORMAT:
    Action Input must be a JSON object with code key:
    - code: string (your Python code as a string)
    - return_result: boolean (optional, defaults to true)
    
    Example: JSON with code key containing your Python code
    The code parameter must be a string containing your Python code
    
    Returns: Execution result or error message.
    """
    
    class CodeInput(BaseModel):
        code: str = Field(description="Python code to execute (REQUIRED - must be a string)")
        return_result: bool = Field(default=True, description="Whether to return the result (optional, defaults to True)")
    
    args_schema: type[BaseModel] = CodeInput
    
    def _run(self, code: str = None, return_result: bool = True, **kwargs) -> str:
        """Execute Python code safely with improved error handling."""
        # Handle case where code might be passed in kwargs or as None
        if code is None:
            # Try to get from kwargs
            if 'code' in kwargs:
                code = kwargs['code']
            else:
                return "ERROR: Missing required 'code' parameter. Action Input must be JSON like: {\"code\": \"your Python code here\"}"
        
        # Ensure code is a string
        if not isinstance(code, str):
            return f"ERROR: 'code' must be a string, but received {type(code).__name__}. Action Input format: {{\"code\": \"your Python code as string\"}}"
        
        try:
            safe_globals = {
                '__builtins__': __builtins__,
                'json': json,
                'os': os,
                'sys': sys,
                'Path': Path,
                'List': List,
                'Dict': Dict,
                'Any': Any,
            }
            
            # Execute code
            exec(code, safe_globals)
            
            # Check for result variable
            if return_result and 'result' in safe_globals:
                return str(safe_globals['result'])
            elif return_result:
                return "Code executed successfully (no result variable set). Tip: Set 'result = your_value' to return a value."
            else:
                return "Code executed successfully"
                
        except SyntaxError as e:
            return f"Syntax Error in code: {str(e)}\n\nPlease check your Python syntax and try again."
        except NameError as e:
            return f"Name Error: {str(e)}\n\nMake sure all variables and functions are defined before use."
        except Exception as e:
            return f"Error executing code: {type(e).__name__}: {str(e)}\n\nPlease review your code and fix the error."


class CreateSubAgentTool(BaseTool):
    """Tool for creating sub-agents dynamically."""
    
    name: str = "create_sub_agent"
    description: str = """Create a new sub-agent with specified role and tools.
    
    CRITICAL INPUT FORMAT:
    Action Input must be a JSON object with these keys:
    - agent_name: string (name of the agent)
    - role: string (role description)
    - workflow: string (workflow steps)
    - tools_description: string (description of tools needed)
    - model_id: integer or null (optional, defaults to parent's model ID)
    
    Example JSON structure:
    agent_name: "data_loader"
    role: "Data Loader Agent"
    workflow: "Load and preprocess datasets"
    tools_description: "Data loading and preprocessing tools"
    model_id: null
    
    The agent will automatically generate appropriate tools based on the description.
    Returns: Agent tool that can be used by the parent agent.
    """
    
    # Add parent_model_id as a Pydantic field so ReActXen can set it
    parent_model_id: int = 15
    
    class CreateAgentInput(BaseModel):
        agent_name: str = Field(description="Name of the sub-agent")
        role: str = Field(description="Role description for the agent")
        workflow: str = Field(description="Workflow steps the agent should follow")
        tools_description: str = Field(description="Description of tools the agent needs")
        model_id: Optional[int] = Field(default=None, description="Model ID for the agent")
    
    args_schema: type[BaseModel] = CreateAgentInput
    
    def __init__(self, parent_model_id: int = 15, root_agent=None, **kwargs):
        super().__init__(**kwargs)
        # Set parent_model_id as a Pydantic field (will be set by framework later)
        self.parent_model_id = parent_model_id
        # Use object.__setattr__ for non-Pydantic attributes
        object.__setattr__(self, 'root_agent', root_agent)
        object.__setattr__(self, '_agent_registry', {})
    
    def _run(
        self,
        agent_name: str,
        role: str,
        workflow: str,
        tools_description: str,
        model_id: Optional[int] = None
    ) -> str:
        """Create a sub-agent dynamically and make it available."""
        try:
            if model_id is None:
                model_id = self.parent_model_id
            
            tools = self._generate_tools_from_description(tools_description)
            
            # Update workflow to mention dynamic tool creation capabilities
            enhanced_workflow = f"""{workflow}

IMPORTANT: You have the ability to CREATE your own tools dynamically:
- Use create_dynamic_tool to create custom tools with your own logic
- Use execute_python_code to write and execute Python code
- Create tools as needed to accomplish your tasks - don't rely only on pre-provided tools"""
            
            agent = create_sub_agent(
                question=f"Execute tasks as {role}",
                key=agent_name.lower().replace(" ", "_"),
                role=role,
                workflow=enhanced_workflow,
                tools=tools,
                max_steps=15,
                react_llm_model_id=model_id
            )
            
            agent_tool = AgentTool(
                agent=agent,
                name=agent_name.lower().replace(" ", "_"),
                description=f"{role}: {workflow}"
            )
            
            self._agent_registry[agent_name] = agent_tool
            
            # Try to add to root agent's tools if available
            if self.root_agent and hasattr(self.root_agent, 'cbm_tools'):
                self.root_agent.cbm_tools.append(agent_tool)
                if hasattr(self.root_agent, 'tool_names'):
                    self.root_agent.tool_names.append(agent_tool.name)
            
            return f"✅ Created sub-agent '{agent_name}' (tool name: '{agent_name.lower().replace(' ', '_')}') with {len(tools)} tools. You can now use this agent by calling it with Action: {agent_name.lower().replace(' ', '_')} and Action Input: JSON with 'query' key containing your question."
            
        except Exception as e:
            return f"Error creating sub-agent: {str(e)}"
    
    def _generate_tools_from_description(self, description: str) -> List[BaseTool]:
        """Generate tools based on description. ALWAYS includes dynamic tool creation capabilities."""
        tools = []
        
        # CRITICAL: All sub-agents get dynamic tool creation capabilities
        # This allows them to create their own tools and execute code during execution
        # Note: We create new instances to avoid sharing state between agents
        tools.append(CodeExecutionTool())
        # Create a new instance of CreateDynamicToolTool for this sub-agent
        # This allows each sub-agent to have its own tool registry
        sub_agent_tool_creator = CreateDynamicToolTool()
        tools.append(sub_agent_tool_creator)
        
        try:
            from shared.shared_utils import get_dataset_tools, get_search_tools
            from tools_logic import create_watsonx_tools
            
            desc_lower = description.lower()
            
            # Only provide base tools - agents can create ML framework tools dynamically
            if any(k in desc_lower for k in ['dataset', 'data', 'load']):
                tools.extend(get_dataset_tools())
            
            if any(k in desc_lower for k in ['search', 'lookup', 'find']):
                tools.extend(get_search_tools())
            
            if any(k in desc_lower for k in ['watsonx', 'model', 'train']):
                tools.extend(create_watsonx_tools())
            
            # Note: ML framework tools (sklearn, pytorch, tensorflow) are NOT pre-provided
            # Agents should create these dynamically using create_dynamic_tool or execute_python_code
            # This reduces code and allows agents to write exactly what they need
                
        except ImportError:
            pass
        
        return tools
    
    def get_created_agents(self) -> Dict[str, Any]:
        """Get all created agents."""
        return self._agent_registry


class CreateDynamicToolTool(BaseTool):
    """Tool for creating custom tools dynamically."""
    
    name: str = "create_dynamic_tool"
    description: str = """Create a custom tool dynamically based on Python code.
    
    Input: JSON with:
    - tool_name: Name of the tool
    - tool_description: Description of what the tool does
    - code: Python code that implements the tool's _run method
    - parameters: Optional JSON string of parameters
    
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
            params = json.loads(parameters) if parameters else {}
            
            tool_code = f"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

class {tool_name}Tool(BaseTool):
    name: str = "{tool_name}"
    description: str = "{tool_description}"
    
    class ToolInput(BaseModel):
"""
            for param_name, param_type in params.items():
                tool_code += f'        {param_name}: {param_type} = Field(description="Parameter {param_name}")\n'
            
            tool_code += f"""
    args_schema = ToolInput
    
    def _run(self, {', '.join(params.keys()) if params else ''}):
{self._indent_code(code)}
"""
            
            exec_globals = {
                'BaseTool': BaseTool,
                'BaseModel': BaseModel,
                'Field': Field,
                'Optional': Optional,
            }
            exec(tool_code, exec_globals)
            
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


def create_dynamic_agent_tools(parent_model_id: int = 15, root_agent=None) -> List[BaseTool]:
    """Create tools for dynamic agent system."""
    return [
        CodeExecutionTool(),
        CreateSubAgentTool(parent_model_id=parent_model_id, root_agent=root_agent),
        CreateDynamicToolTool(),
    ]
