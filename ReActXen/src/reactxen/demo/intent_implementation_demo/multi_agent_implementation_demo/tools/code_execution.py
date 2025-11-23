"""
Code Execution Tool - Handles dynamic Python code execution.
Refactored from dynamic_agent_system.py to keep files under 200 lines.
"""
import os
import sys
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Thread-local storage for raw action input (workaround for ReactAgent's JSON parsing)
_thread_local = threading.local()

# Import JSON parser handler
try:
    from tools.json_parser import parse_json_input, normalize_json_input, extract_code_from_input
except ImportError:
    # Fallback if import fails
    def parse_json_input(input_data, tool_name="unknown"):
        if isinstance(input_data, dict):
            return input_data if input_data else None
        if isinstance(input_data, str):
            try:
                return json.loads(input_data)
            except json.JSONDecodeError:
                return None
        return None
    
    def normalize_json_input(input_data, expected_keys=None):
        parsed = parse_json_input(input_data)
        if parsed is None:
            raise ValueError(f"Failed to parse input")
        if expected_keys:
            missing = [k for k in expected_keys if k not in parsed]
            if missing:
                raise ValueError(f"Missing keys: {missing}")
        return parsed
    
    def extract_code_from_input(input_data):
        parsed = parse_json_input(input_data)
        if parsed and 'code' in parsed:
            return str(parsed['code'])
        return str(input_data) if input_data else ""

# Import memory system
try:
    from memory_system import get_memory_system
    _memory_available = True
except ImportError:
    _memory_available = False
    def get_memory_system():
        return None


class CodeExecutionTool(BaseTool):
    """Tool for executing Python code dynamically."""
    
    name: str = "execute_python_code"
    description: str = """Execute Python code dynamically. Use this to:
    - Perform calculations
    - Process data
    - Create helper functions
    - Generate dynamic tools
    - Write and execute any Python code needed
    
    CRITICAL INPUT FORMAT (MUST FOLLOW THIS EXACTLY):
    Action Input MUST be a SINGLE-LINE JSON object. Use escaped newlines (\\n) NOT literal newlines.
    
    Format: {"code": "your code here", "return_result": true}
    
    IMPORTANT RULES:
    1. Use escaped newlines (\\n) in code strings, NOT literal newlines
    2. Keep JSON on one line to avoid parsing issues
    3. Escape quotes inside code strings: {"code": "print(\\\"hello\\\")"}
    
    CORRECT EXAMPLE:
    Action Input: {"code": "import pandas as pd\\n\\ndata = pd.read_csv('file.csv')\\nprint(data.head())", "return_result": true}
    
    WRONG (will fail):
    Action Input: {
      "code": "import pandas as pd
      data = pd.read_csv('file.csv')",
      "return_result": true
    }
    
    Returns: Execution result or error message.
    """
    
    class CodeInput(BaseModel):
        code: str = Field(description="Python code to execute (REQUIRED)")
        return_result: bool = Field(default=True, description="Whether to return result")
    
    args_schema: type[BaseModel] = CodeInput
    
    def _invoke(self, input: Any, config: Any = None) -> Any:
        """LangChain invoke method - wraps run() to handle different input formats."""
        # LangChain may call _invoke with different input formats
        if isinstance(input, dict) and 'code' in input:
            return self.run(tool_input=input)
        elif isinstance(input, str):
            return self.run(tool_input=input)
        else:
            return self.run(tool_input=input)
    
    def run(self, tool_input: Any = None, **kwargs) -> str:
        """
        Override run method to handle input parsing using robust JSON parser.
        
        ReactAgent may pass:
        1. A parsed dict (from successful JSON parsing)
        2. An empty dict {} (when ReactAgent's JSON parsing produces empty dict)
        3. A raw JSON string (as fallback)
        4. Nothing (None)
        
        CRITICAL: When ReactAgent's JSON parsing fails or produces empty dict,
        we need to handle it gracefully and extract code from whatever input we have.
        """
        # CRITICAL FIX: ReactAgent calls tool.run(tool_input=dictionary) after parsing JSON
        # If JSON parsing succeeds but produces empty dict {}, tool_input will be {}
        # In this case, ReactAgent's JSON parsing likely failed silently or the JSON was malformed
        # We need to handle this case by trying to extract code from any available source
        
        # Priority 0: Check thread-local storage for raw action input (workaround)
        raw_action_input = getattr(_thread_local, 'raw_action_input', None)
        if raw_action_input:
            # Clear it after use
            _thread_local.raw_action_input = None
            try:
                parsed = parse_json_input(raw_action_input, "execute_python_code")
                if parsed and isinstance(parsed, dict) and 'code' in parsed and parsed.get('code'):
                    parsed_input = parsed
                else:
                    parsed_input = None
            except Exception:
                parsed_input = None
        else:
            parsed_input = None
        
        # Priority 1: If tool_input is a string (alternative format), try to parse it as code
        if not parsed_input and isinstance(tool_input, str) and tool_input.strip():
            # Alternative format: code passed as a simple string
            # This is a workaround for ReactAgent's JSON parsing issues
            if tool_input.strip().startswith('{'):
                # It's JSON, try to parse it
                try:
                    parsed = parse_json_input(tool_input, "execute_python_code")
                    if parsed and isinstance(parsed, dict) and 'code' in parsed and parsed.get('code'):
                        parsed_input = parsed
                except Exception:
                    pass
            else:
                # It's plain code, wrap it
                parsed_input = {
                    'code': tool_input,
                    'return_result': True
                }
        
        # Priority 2: If tool_input is already a valid dict with 'code', use it directly
        if not parsed_input and isinstance(tool_input, dict):
            if 'code' in tool_input and tool_input.get('code'):
                # Valid dict with code - use it directly
                parsed_input = tool_input
            elif len(tool_input) == 0:
                # Empty dict {} - ReactAgent's JSON parsing succeeded but produced empty dict
                # CRITICAL: ReactAgent's extract_action_and_input (line 115 in utils.py) replaces newlines with spaces
                # This breaks JSON when the code string contains literal newlines
                # SOLUTION: Try to recover from agent's json_log or scratchpad
                if not parsed_input:
                    # Try to access agent's json_log to get raw_llm_action_output
                    try:
                        agent = getattr(self, 'agent_ref', None)
                        if agent and hasattr(agent, 'json_log') and agent.json_log:
                            try:
                                # Get the most recent log entry
                                latest_log = agent.json_log[-1]
                                # Try multiple keys for raw LLM output
                                raw_llm_output = (
                                    latest_log.get('raw_llm_action_output', '') or
                                    latest_log.get('raw_llm_output', '') or
                                    latest_log.get('action', '') or
                                    ''
                                )
                                
                                # Also try scratchpad as fallback
                                scratchpad = getattr(agent, 'scratchpad', '')
                                
                                # Search in both raw_llm_output and scratchpad
                                for source_text in [raw_llm_output, scratchpad]:
                                    if not source_text:
                                        continue
                                    
                                    # Extract action input from LLM output
                                    # Format: "Action: execute_python_code\nAction Input: {...}"
                                    import re
                                    # Try multiple patterns - be more aggressive with matching
                                    # CRITICAL: ReactAgent replaces newlines with spaces, so we need to match across spaces
                                    patterns = [
                                        # Pattern 1: Standard format with greedy matching (non-greedy to avoid matching too much)
                                        r'Action\s+Input\s*\d*\s*:\s*(\{[^}]*"code"[^}]*\})',
                                        # Pattern 2: Match even if newlines were replaced with spaces (more flexible)
                                        r'Action\s+Input\s*\d*\s*:\s*(\{[^}]*"code"[^}]*"return_result"[^}]*\})',
                                        # Pattern 3: Match with spaces where newlines were (ReactAgent's corruption)
                                        r'Action\s+Input\s*\d*\s*:\s*(\{[^}]*"code"\s*:\s*"[^"]*"[^}]*\})',
                                        # Pattern 4: Direct code extraction (more flexible)
                                        r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                        # Pattern 5: Match code even if JSON is broken by newline replacement
                                        r'"code"\s*:\s*"([^"]+)"',
                                    ]
                                    
                                    for pattern in patterns:
                                        action_input_match = re.search(
                                            pattern,
                                            source_text,
                                            re.DOTALL | re.IGNORECASE
                                        )
                                        if action_input_match:
                                            raw_json_str = action_input_match.group(1)
                                            
                                            # If pattern matched code directly, wrap it
                                            if pattern.startswith(r'"code"'):
                                                code_value = raw_json_str
                                                # Unescape if needed
                                                code_value = code_value.replace('\\"', '"').replace('\\n', '\n')
                                                import json
                                                raw_json_str = f'{{"code": {json.dumps(code_value)}, "return_result": true}}'
                                            
                                            # Try to fix JSON if newlines were replaced with spaces
                                            # Look for patterns like "code": "import pandas  data = ..." where spaces should be newlines
                                            if '  ' in raw_json_str and '"code"' in raw_json_str:
                                                # Try to reconstruct: find the code value and fix it
                                                code_match = re.search(r'"code"\s*:\s*"([^"]+)"', raw_json_str)
                                                if code_match:
                                                    code_value = code_match.group(1)
                                                    # Heuristic: if there are multiple spaces in a row, they might be newlines
                                                    # But this is risky - only do it if we can't parse otherwise
                                                    try:
                                                        test_parsed = json.loads(raw_json_str)
                                                        if test_parsed and 'code' in test_parsed:
                                                            raw_json_str = json.dumps(test_parsed)  # Re-serialize to fix formatting
                                                    except:
                                                        pass
                                            
                                            parsed = parse_json_input(raw_json_str, "execute_python_code")
                                            if parsed and isinstance(parsed, dict) and 'code' in parsed and parsed.get('code'):
                                                parsed_input = parsed
                                                break
                                    
                                    if parsed_input:
                                        break
                            except Exception as e:
                                # Debug: print error if needed
                                pass
                    except Exception:
                        pass  # Silently fail if agent ref access fails
                    
                    # If still no input, try thread-local storage
                    if not parsed_input:
                        raw_action_input = getattr(_thread_local, 'raw_action_input', None)
                        if raw_action_input:
                            try:
                                parsed = parse_json_input(raw_action_input, "execute_python_code")
                                if parsed and isinstance(parsed, dict) and 'code' in parsed and parsed.get('code'):
                                    parsed_input = parsed
                                    _thread_local.raw_action_input = None
                            except Exception:
                                pass
                    
                    # If still no input, try to get any string input from kwargs
                    if not parsed_input:
                        for key in ['argument', 'action_input', 'input', 'raw_input', 'raw_string']:
                            raw_string = kwargs.get(key)
                            if raw_string and isinstance(raw_string, str) and raw_string.strip():
                                try:
                                    parsed = parse_json_input(raw_string, "execute_python_code")
                                    if parsed and isinstance(parsed, dict) and 'code' in parsed and parsed.get('code'):
                                        parsed_input = parsed
                                        break
                                except Exception:
                                    continue
            else:
                # Dict with keys but no 'code' - might be malformed or different structure
                if not parsed_input:
                    parsed_input = None
        elif not parsed_input:
            # tool_input is not a dict - might be a string or other type
            parsed_input = None
        
        # Priority 2: If not a valid dict yet, try to parse from all sources
        if not parsed_input or not isinstance(parsed_input, dict) or 'code' not in parsed_input or not parsed_input.get('code'):
            # Get all possible input sources
            input_sources = [
                tool_input,
                kwargs.get('argument'),
                kwargs.get('action_input'),
                kwargs.get('input'),
                str(kwargs) if kwargs else None
            ]
            
            # Try to parse using the robust JSON parser
            for source in input_sources:
                if source is None or source == {}:
                    continue
                    
                # If source is already a valid dict, use it
                if isinstance(source, dict) and 'code' in source and source.get('code'):
                    parsed_input = source
                    break
                
                # Try parsing as JSON string
                try:
                    if isinstance(source, str) and source.strip():
                        # Use robust JSON parser
                        parsed = parse_json_input(source, "execute_python_code")
                        if parsed and isinstance(parsed, dict) and 'code' in parsed and parsed.get('code'):
                            parsed_input = parsed
                            break
                except Exception:
                    continue
            
            # If still no valid input, try extract_code_from_input as fallback
            if not parsed_input or not isinstance(parsed_input, dict) or 'code' not in parsed_input or not parsed_input.get('code'):
                for source in input_sources:
                    if not source or source == {}:
                        continue
                    code_str = extract_code_from_input(source)
                    if code_str and code_str.strip():
                        parsed_input = {
                            'code': code_str,
                            'return_result': kwargs.get('return_result', True)
                        }
                        break
        
        # Validate parsed input - final check
        if not parsed_input or not isinstance(parsed_input, dict) or 'code' not in parsed_input or not parsed_input.get('code'):
            # Last resort error handling with helpful guidance
            received_keys = list(parsed_input.keys()) if isinstance(parsed_input, dict) and parsed_input else []
            
            # Special message for empty dict case
            if isinstance(tool_input, dict) and len(tool_input) == 0:
                error_msg = """ERROR: Empty JSON input received. ReactAgent's JSON parsing produced an empty dict {}.
                
CRITICAL FIX REQUIRED: The Action Input JSON format is being parsed incorrectly by ReactAgent.

SOLUTION: Use a single-line JSON format without literal newlines inside the code string. ReactAgent replaces newlines with spaces, which can break JSON parsing.

CORRECT FORMAT:
Action Input: {"code": "import pandas as pd\\n\\ndata = pd.read_csv('file.csv')", "return_result": true}

IMPORTANT:
1. Use escaped newlines (\\n) instead of literal newlines in the code string
2. Keep the JSON on a single line or minimize newlines
3. Ensure the JSON is properly quoted and escaped

If this error persists, try breaking the code into smaller execute_python_code calls."""
            else:
                error_msg = f"ERROR: Missing or invalid input. Could not parse JSON or extract code. Input must be a JSON object with 'code' key. Received keys: {received_keys if received_keys else 'none'}. Input type: {type(tool_input).__name__}, Input value: {str(tool_input)[:200] if tool_input else 'empty'}, kwargs keys: {list(kwargs.keys()) if kwargs else 'none'}"
            
            if _memory_available:
                memory = get_memory_system()
                if memory:
                    should_avoid, avoid_reason = memory.should_avoid_action("execute_python_code", str(tool_input or kwargs))
                    if should_avoid:
                        error_msg += f"\n\n⚠️ MEMORY WARNING: {avoid_reason}\n💡 CRITICAL: Switch to execute_code_simple tool instead. Do NOT retry execute_python_code."
                    memory.record_mistake(
                        action="execute_python_code",
                        action_input=str(tool_input)[:200] if tool_input else "",
                        error=f"Missing 'code' key. Received keys: {received_keys}",
                        context="Input validation failed - missing code key"
                    )
            
            # Always suggest execute_code_simple as alternative
            error_msg += "\n\n💡 SOLUTION: Use execute_code_simple tool instead. It handles JSON parsing issues better."
            return error_msg
        
        # Extract and execute - use parsed_input, not tool_input
        code = parsed_input['code']
        return_result = parsed_input.get('return_result', True)
        return self._run(code=code, return_result=return_result, **kwargs)
    
    def _run(self, code: str = None, return_result: bool = True, **kwargs) -> str:
        """Execute Python code safely."""
        if code is None:
            return "ERROR: Missing required 'code' parameter."
        
        if not isinstance(code, str):
            return f"ERROR: 'code' must be a string, but received {type(code).__name__}."
        
        try:
            # Import shared namespace for variable persistence
            try:
                from shared_execution_namespace import get_shared_namespace
                shared_namespace = get_shared_namespace()
            except ImportError:
                # Fallback if shared namespace not available
                shared_namespace = {}
            
            # Start with shared namespace (allows variable persistence)
            safe_globals = dict(shared_namespace)  # Copy to avoid modifying original
            safe_globals.update({
                'json': json,
                'os': os,
                'sys': sys,
                'Path': Path,
            })
            
            # Also import from shared.load_data if available
            try:
                from shared.load_data import train_data, test_data, ground_truth
                safe_globals['train_data'] = train_data
                safe_globals['test_data'] = test_data
                safe_globals['ground_truth'] = ground_truth
            except ImportError:
                pass
            
            exec(code, safe_globals)
            
            # Update shared namespace with new/modified variables
            try:
                from shared_execution_namespace import update_shared_namespace
                # Only update non-builtin variables
                updates = {k: v for k, v in safe_globals.items() 
                          if not k.startswith('__') and k not in ['json', 'os', 'sys', 'Path']}
                update_shared_namespace(updates)
            except ImportError:
                pass
            
            if return_result and 'result' in safe_globals:
                result = str(safe_globals['result'])
                self._record_success(code, result)
                return result
            elif return_result:
                result = "Code executed successfully (no result variable set)."
                self._record_success(code, result)
                return result
            else:
                self._record_success(code, "Code executed successfully")
                return "Code executed successfully"
                
        except SyntaxError as e:
            return self._record_error("Syntax error", str(e), code)
        except NameError as e:
            return self._record_error("Name error", str(e), code)
        except Exception as e:
            return self._record_error(f"Error: {type(e).__name__}", str(e), code)
    
    def _record_error(self, error_type: str, error_msg: str, action_input: Any) -> str:
        """Record error in memory and return error message."""
        error = f"ERROR: {error_type}: {error_msg}"
        if _memory_available:
            memory = get_memory_system()
            if memory:
                memory.record_mistake(
                    action="execute_python_code",
                    action_input=str(action_input)[:200] if action_input else "",
                    error=error_msg,
                    context=f"{error_type} in code execution"
                )
        return error
    
    def set_agent_ref(self, agent):
        """Set agent reference for accessing scratchpad/json_log."""
        object.__setattr__(self, 'agent_ref', agent)
    
    def _record_success(self, code: str, result: str):
        """Record success in memory."""
        if _memory_available:
            memory = get_memory_system()
            if memory:
                memory.record_solution(
                    action="execute_python_code",
                    action_input=code[:200] if code else "",
                    result=result[:200],
                    context="Python code executed successfully"
                )
