"""
Agent Tool Wrapper - Wraps ReActXen agents as tools for hierarchical agent structures.
This allows agents to be used as tools by other agents, enabling multi-level agent hierarchies.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Type, Optional, Any
import json


class AgentToolInput(BaseModel):
    """Input schema for agent tool wrapper."""
    query: str = Field(description="The question or task to delegate to the sub-agent")
    context: Optional[str] = Field(default=None, description="Additional context for the sub-agent")


class AgentTool(BaseTool):
    """
    Wraps a ReActXen agent as a tool, allowing it to be used by other agents.
    This enables hierarchical agent structures where agents can delegate tasks to sub-agents.
    """
    
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = AgentToolInput
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    def __init__(
        self,
        agent: Any,  # ReactReflectAgent or similar
        name: str,
        description: str,
        **kwargs
    ):
        """
        Initialize an AgentTool wrapper.
        
        Args:
            agent: The ReActXen agent instance to wrap
            name: Tool name (should be unique)
            description: Description of what this agent tool does
        """
        super().__init__(name=name, description=description, **kwargs)
        # Set agent as instance attribute (not a Pydantic field)
        object.__setattr__(self, 'agent', agent)
    
    def run(self, tool_input: Any = None, **kwargs) -> str:
        """
        Override run method to properly handle JSON input parsing.
        This ensures the query parameter is correctly extracted from the tool_input.
        Handles ReactAgent parsing bug where JSON containing "=" is incorrectly parsed.
        """
        try:
            # Handle different input formats
            if tool_input is None:
                tool_input = kwargs
            
            # CRITICAL FIX: Handle case where ReactAgent incorrectly parsed JSON as key-value pairs
            # When ReactAgent sees "=" in the JSON string, it extracts it as {"key": "value"} instead
            # of parsing the full JSON. We need to detect this and reconstruct from kwargs or action_input.
            if isinstance(tool_input, dict) and 'query' not in tool_input and len(tool_input) == 0:
                # Empty dict - ReactAgent probably failed to parse JSON
                # Try to get the original action_input from kwargs
                if 'action_input' in kwargs:
                    action_input = kwargs['action_input']
                    if isinstance(action_input, str) and action_input.strip().startswith('{'):
                        try:
                            # Try to parse as JSON directly
                            tool_input = json.loads(action_input)
                        except json.JSONDecodeError:
                            # If still can't parse, treat as query string
                            return self._run(query=action_input)
            
            # If tool_input is a string (JSON), parse it
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as query string
                    return self._run(query=tool_input)
            
            # If tool_input is a dict, extract query and context
            if isinstance(tool_input, dict):
                query = tool_input.get('query', '')
                context = tool_input.get('context', None)
                
                # If query is missing but tool_input has content, try to extract it
                if not query and tool_input:
                    # Try to find query in various formats
                    for key in ['query', 'question', 'text', 'input']:
                        if key in tool_input:
                            query = tool_input[key]
                            break
                    
                    # If still no query but we have action_input in kwargs, try that
                    if not query and 'action_input' in kwargs:
                        action_input = kwargs['action_input']
                        if isinstance(action_input, str):
                            if action_input.strip().startswith('{'):
                                try:
                                    parsed = json.loads(action_input)
                                    query = parsed.get('query', action_input)
                                except:
                                    query = action_input
                            else:
                                query = action_input
                    
                    # If still no query, use the whole dict as a string representation
                    if not query:
                        query = str(tool_input)
                
                # Ensure query is not empty
                if not query or query == '{}':
                    return "ERROR: Missing 'query' key in input. Action Input must be JSON like: {\"query\": \"your question here\"}"
                
                return self._run(query=query, context=context)
            
            # Fallback: treat as query string
            return self._run(query=str(tool_input))
            
        except Exception as e:
            return f"Error parsing tool input: {str(e)}. Expected JSON with 'query' key or a query string."
    
    def _run(self, query: str, context: Optional[str] = None) -> str:
        """
        Execute the wrapped agent with the given query.
        
        Args:
            query: The question or task for the sub-agent
            context: Additional context (optional, must be a string)
        
        Returns:
            The agent's response as a string
        """
        try:
            # Convert context to string if it's not already
            if context and not isinstance(context, str):
                import json
                context = json.dumps(context) if isinstance(context, dict) else str(context)
            
            # Update the agent's question
            if hasattr(self.agent, 'question'):
                if context:
                    self.agent.question = f"{query}\n\nContext: {context}"
                else:
                    self.agent.question = query
            
            # Run the agent
            if hasattr(self.agent, 'run'):
                self.agent.run()
                
                # Extract the answer
                if hasattr(self.agent, 'answer'):
                    return self.agent.answer
                elif hasattr(self.agent, 'export_benchmark_metric'):
                    metrics = self.agent.export_benchmark_metric()
                    if metrics and 'final_answer' in metrics:
                        return metrics['final_answer']
                    elif metrics and 'status' in metrics:
                        return f"Agent completed with status: {metrics['status']}"
                
                return "Agent execution completed, but no answer was extracted."
            else:
                return f"Error: Agent does not have a 'run' method."
                
        except Exception as e:
            import traceback
            error_details = str(e)
            # Provide helpful error message
            return f"Error executing sub-agent: {error_details}. Please try with a simpler query or use direct tools instead."
    
    def _arun(self, query: str, context: Optional[str] = None) -> str:
        """Async version (not implemented yet)."""
        return self._run(query, context)

