"""
Dynamic Agent System - Allows agents to create sub-agents and tools dynamically.
Agents can write and execute their own code to reduce manual coding.
Refactored to use modular tools.
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

# Import code execution tool from refactored module
try:
    from tools.code_execution import CodeExecutionTool
except ImportError:
    # Fallback to local import if tools module not available
    CodeExecutionTool = None

# Import memory system for automatic recording
try:
    from memory_system import get_memory_system
    _memory_available = True
except ImportError:
    _memory_available = False
    def get_memory_system():
        return None

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
    
    def run(self, tool_input: Any = None, **kwargs) -> str:
        """Override run method to handle input parsing manually before Pydantic validation."""
        # CRITICAL FIX: ReactAgent checks for "=" before checking for JSON, so it may incorrectly
        # parse JSON strings containing "axis=1" etc. as key-value pairs. We need to handle this.
        
        # FIRST: Try to get the original action_input from kwargs (this is the raw string from LLM)
        original_action_input = kwargs.get('action_input', None)
        if original_action_input and isinstance(original_action_input, str):
            # If it looks like JSON, try to parse it directly
            if original_action_input.strip().startswith('{') and original_action_input.strip().endswith('}'):
                try:
                    parsed = json.loads(original_action_input)
                    if 'code' in parsed:
                        # Success! Use this parsed version
                        tool_input = parsed
                except json.JSONDecodeError:
                    pass  # Will try other methods below
        
        # Handle case where tool_input is a string (JSON string)
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError as e:
                error_msg = f"ERROR: Invalid JSON string. Action Input must be valid JSON like: {{\"code\": \"your Python code here\"}}"
                # Record parsing error in memory
                if _memory_available:
                    memory = get_memory_system()
                    if memory:
                        memory.record_mistake(
                            action="execute_python_code",
                            action_input=tool_input[:200] if tool_input else "",
                            error=f"JSON parsing error: {str(e)}",
                            context="JSON parsing failed in execute_python_code input"
                        )
                return error_msg
        
        # Handle case where tool_input is None or empty dict
        if not tool_input or (isinstance(tool_input, dict) and len(tool_input) == 0):
            # Try to get from kwargs
            if 'code' in kwargs:
                tool_input = {'code': kwargs['code'], 'return_result': kwargs.get('return_result', True)}
            elif 'action_input' in kwargs:
                action_input = kwargs['action_input']
                if isinstance(action_input, str):
                    # If it looks like JSON, try to parse it
                    if action_input.strip().startswith('{') and action_input.strip().endswith('}'):
                    try:
                        tool_input = json.loads(action_input)
                        except json.JSONDecodeError as e:
                            error_msg = f"ERROR: Could not parse JSON from action_input. Action Input must be valid JSON like: {{\"code\": \"your Python code here\"}}"
                            if _memory_available:
                                memory = get_memory_system()
                                if memory:
                                    memory.record_mistake(
                                        action="execute_python_code",
                                        action_input=action_input[:200],
                                        error=f"JSON decode error: {str(e)}",
                                        context="JSON parsing failed in execute_python_code"
                                    )
                            return error_msg
                    else:
                        # Not JSON format - might be just the code string
                        tool_input = {'code': action_input, 'return_result': True}
                else:
                    tool_input = action_input
        
        # If tool_input is still empty or missing code, provide helpful error
        if not tool_input or not isinstance(tool_input, dict):
            error_msg = "ERROR: Missing or invalid input. Action Input must be a JSON object with 'code' key containing your Python code as a string."
            # Record parsing error in memory
            if _memory_available:
                memory = get_memory_system()
                if memory:
                    memory.record_mistake(
                        action="execute_python_code",
                        action_input=str(tool_input)[:200] if tool_input else "",
                        error="Missing or invalid input - not a dict",
                        context="Input validation failed in execute_python_code"
                    )
            return error_msg
        
        # CRITICAL FIX: Handle case where ReactAgent incorrectly parsed JSON as key-value pairs
        # When ReactAgent sees "=" in the JSON string, it extracts it as key-value pairs instead of
        # parsing the full JSON. We need to detect this and reconstruct from kwargs or action_input.
        if 'code' not in tool_input:
            # Check if we have wrong keys (like 'axis' which indicates incorrect parsing)
            received_keys = list(tool_input.keys())
            
            # Try to get the original action_input from kwargs (this is the raw string from LLM)
            original_action_input = kwargs.get('action_input', None)
            if original_action_input and isinstance(original_action_input, str):
                # If it looks like JSON, try to parse it directly
                if original_action_input.strip().startswith('{') and original_action_input.strip().endswith('}'):
                    try:
                        # Try to parse as JSON directly
                        tool_input = json.loads(original_action_input)
                        if 'code' in tool_input:
                            # Success! Use this instead
                            pass
                        else:
                            error_msg = f"ERROR: Missing 'code' key in input. Received keys: {received_keys}. Action Input format: {{\"code\": \"your Python code as string\"}}"
                            if _memory_available:
                                memory = get_memory_system()
                                if memory:
                                    memory.record_mistake(
                                        action="execute_python_code",
                                        action_input=str(tool_input)[:200],
                                        error=f"Missing 'code' key. Received keys: {received_keys}",
                                        context="JSON parsing issue - missing code key"
                                    )
                            return error_msg
                    except json.JSONDecodeError as e:
                        error_msg = f"ERROR: Could not parse JSON from action_input. Action Input must be valid JSON like: {{\"code\": \"your Python code here\"}}"
                        if _memory_available:
                            memory = get_memory_system()
                            if memory:
                                memory.record_mistake(
                                    action="execute_python_code",
                                    action_input=str(original_action_input)[:200],
                                    error=f"JSON decode error: {str(e)}",
                                    context="JSON parsing failed in execute_python_code"
                                )
                        return error_msg
                else:
                    # Not JSON format - ReactAgent may have incorrectly parsed it
                    error_msg = f"ERROR: Missing 'code' key in input. Received keys: {received_keys}. Action Input format: {{\"code\": \"your Python code as string\"}}. This may indicate that the ReactAgent incorrectly parsed JSON containing '=' characters. Try using learning_analyze tool to understand the issue."
                    if _memory_available:
                        memory = get_memory_system()
                        if memory:
                            # Check if this is a repeated mistake
                            should_avoid, avoid_reason = memory.should_avoid_action("execute_python_code", str(tool_input))
                            if should_avoid:
                                error_msg += f"\n\n⚠️ MEMORY WARNING: {avoid_reason}\n💡 SUGGESTION: Use learning_analyze tool to get recommendations: Action: learning_analyze, Action Input: {{\"action\": \"execute_python_code\", \"error\": \"Missing code key\"}}"
                            
                            memory.record_mistake(
                                action="execute_python_code",
                                action_input=str(tool_input)[:200],
                                error=f"Missing 'code' key. Received keys: {received_keys}. Possible ReactAgent parsing bug",
                                context="JSON parsing issue - ReactAgent may have incorrectly parsed JSON"
                            )
                    return error_msg
            else:
                error_msg = f"ERROR: Missing 'code' key in input. Received keys: {received_keys}. Action Input format: {{\"code\": \"your Python code as string\"}}"
                if _memory_available:
                    memory = get_memory_system()
                    if memory:
                        # Check if this is a repeated mistake
                        should_avoid, avoid_reason = memory.should_avoid_action("execute_python_code", str(tool_input))
                        if should_avoid:
                            error_msg += f"\n\n⚠️ MEMORY WARNING: {avoid_reason}\n💡 SUGGESTION: Use learning_analyze tool: Action: learning_analyze, Action Input: {{\"action\": \"execute_python_code\", \"error\": \"Missing code key\"}}"
                        
                        memory.record_mistake(
                            action="execute_python_code",
                            action_input=str(tool_input)[:200] if tool_input else "",
                            error=f"Missing 'code' key. Received keys: {received_keys}",
                            context="Input validation failed - missing code key"
                        )
                return error_msg
        
        # Extract code and return_result from tool_input
        code = tool_input['code']
        return_result = tool_input.get('return_result', True)
        
        # Call _run directly with parsed values to bypass Pydantic validation issues
        return self._run(code=code, return_result=return_result, tool_input=tool_input, **kwargs)
    
    def _run(self, code: str = None, return_result: bool = True, tool_input: dict = None, **kwargs) -> str:
        """Execute Python code safely with improved error handling."""
        # Handle tool_input parameter (used by ReActXen framework)
        if tool_input is not None:
            if isinstance(tool_input, dict):
                if 'code' in tool_input:
                    code = tool_input['code']
                    if 'return_result' in tool_input:
                        return_result = tool_input['return_result']
                else:
                    # Try to parse if it's a JSON string
                    if len(tool_input) == 0:
                        return "ERROR: Empty input received. Action Input must be a JSON object with 'code' key containing your Python code as a string."
                    return f"ERROR: Missing 'code' key in input. Received keys: {list(tool_input.keys())}. Action Input must be JSON like: {{\"code\": \"your Python code here\"}}"
        
        # Handle case where code might be passed in kwargs or as None
        if code is None:
            # Try to get from kwargs
            if 'code' in kwargs:
                code = kwargs['code']
            # Try to parse from action_input if it's a dict
            elif 'action_input' in kwargs:
                action_input = kwargs['action_input']
                if isinstance(action_input, str):
                    try:
                        action_input = json.loads(action_input)
                    except json.JSONDecodeError:
                        return f"ERROR: Could not parse JSON. Action Input must be valid JSON like: {{\"code\": \"your Python code here\"}}"
                if isinstance(action_input, dict) and 'code' in action_input:
                    code = action_input['code']
                    if 'return_result' in action_input:
                        return_result = action_input['return_result']
                elif isinstance(action_input, dict):
                    return f"ERROR: Missing 'code' key. Received keys: {list(action_input.keys())}. Action Input format: {{\"code\": \"your Python code as string\"}}"
            else:
                return "ERROR: Missing required 'code' parameter. Action Input must be a JSON object with 'code' key containing your Python code as a string."
        
        # Handle case where code might be a dict (malformed input)
        if isinstance(code, dict):
            if 'code' in code:
                code = code['code']
            else:
                return f"ERROR: Input is a dict but missing 'code' key. Received keys: {list(code.keys())}. Action Input format: {{\"code\": \"your Python code as string\"}}"
        
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
                result = str(safe_globals['result'])
                # Record success in memory
                if _memory_available:
                    memory = get_memory_system()
                    if memory:
                        memory.record_solution(
                            action="execute_python_code",
                            action_input=code[:200] if code else "",
                            result=result[:200],
                            context="Python code executed successfully"
                        )
                return result
            elif return_result:
                result = "Code executed successfully (no result variable set). Tip: Set 'result = your_value' to return a value."
                # Record success in memory
                if _memory_available:
                    memory = get_memory_system()
                    if memory:
                        memory.record_solution(
                            action="execute_python_code",
                            action_input=code[:200] if code else "",
                            result="Code executed successfully",
                            context="Python code executed successfully"
                        )
                return result
            else:
                result = "Code executed successfully"
                # Record success in memory
                if _memory_available:
                    memory = get_memory_system()
                    if memory:
                        memory.record_solution(
                            action="execute_python_code",
                            action_input=code[:200] if code else "",
                            result="Code executed successfully",
                            context="Python code executed successfully"
                        )
                return result
                
        except SyntaxError as e:
            error_msg = f"Syntax Error in code: {str(e)}\n\nPlease check your Python syntax and try again."
            # Record mistake in memory
            if _memory_available:
                memory = get_memory_system()
                if memory:
                    memory.record_mistake(
                        action="execute_python_code",
                        action_input=code[:200] if code else "",
                        error=str(e),
                        context="Syntax error in Python code execution"
                    )
            return error_msg
        except NameError as e:
            error_msg = f"Name Error: {str(e)}\n\nMake sure all variables and functions are defined before use."
            # Record mistake in memory
            if _memory_available:
                memory = get_memory_system()
                if memory:
                    memory.record_mistake(
                        action="execute_python_code",
                        action_input=code[:200] if code else "",
                        error=str(e),
                        context="Name error in Python code execution"
                    )
            return error_msg
        except Exception as e:
            error_msg = f"Error executing code: {type(e).__name__}: {str(e)}\n\nPlease review your code and fix the error."
            # Record mistake in memory
            if _memory_available:
                memory = get_memory_system()
                if memory:
                    memory.record_mistake(
                        action="execute_python_code",
                        action_input=code[:200] if code else "",
                        error=str(e),
                        context="Error in Python code execution"
                    )
            return error_msg


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
    
    def run(self, tool_input: Any = None, **kwargs) -> str:
        """Override run method to handle ReactAgent parsing issues."""
        import json
        
        # Handle case where ReactAgent incorrectly parsed JSON as key-value pairs
        if tool_input is None:
            tool_input = {}
        elif not isinstance(tool_input, dict):
            tool_input = {}
        
        # Check if ReactAgent incorrectly parsed JSON (missing required fields)
        if not all(k in tool_input for k in ['tool_name', 'tool_description', 'code']):
            # Try to get the original action_input from kwargs
            if 'action_input' in kwargs:
                action_input = kwargs['action_input']
                if isinstance(action_input, str) and action_input.strip().startswith('{'):
                    try:
                        # Try to parse as JSON directly
                        tool_input = json.loads(action_input)
                    except json.JSONDecodeError:
                        pass
            
            # If still missing required fields, return helpful error
            if not all(k in tool_input for k in ['tool_name', 'tool_description', 'code']):
                received_keys = list(tool_input.keys())
                error_msg = f"ERROR: Missing required keys in input. Required: tool_name, tool_description, code. Received keys: {received_keys}. Action Input must be valid JSON like: {{\"tool_name\": \"name\", \"tool_description\": \"desc\", \"code\": \"your code here\", \"parameters\": null}}"
                # Record mistake in memory
                if _memory_available:
                    memory = get_memory_system()
                    if memory:
                        memory.record_mistake(
                            action="create_dynamic_tool",
                            action_input=str(tool_input)[:200] if tool_input else "",
                            error="Missing required keys",
                            context="JSON parsing error in create_dynamic_tool"
                        )
                return error_msg
        
        # Call _run with parsed values
        return self._run(
            tool_name=tool_input.get('tool_name', ''),
            tool_description=tool_input.get('tool_description', ''),
            code=tool_input.get('code', ''),
            parameters=tool_input.get('parameters', '{}')
        )
    
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
            
            result = f"✅ Created tool '{tool_name}'. It is now available for use."
            # Record success in memory
            if _memory_available:
                memory = get_memory_system()
                if memory:
                    memory.record_solution(
                        action="create_dynamic_tool",
                        action_input=f"tool_name={tool_name}",
                        result=result,
                        context="Dynamic tool created successfully"
                    )
            return result
            
        except Exception as e:
            error_msg = f"Error creating tool: {str(e)}"
            # Record mistake in memory
            if _memory_available:
                memory = get_memory_system()
                if memory:
                    memory.record_mistake(
                        action="create_dynamic_tool",
                        action_input=f"tool_name={tool_name}",
                        error=str(e),
                        context="Error creating dynamic tool"
                    )
            return error_msg
    
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
