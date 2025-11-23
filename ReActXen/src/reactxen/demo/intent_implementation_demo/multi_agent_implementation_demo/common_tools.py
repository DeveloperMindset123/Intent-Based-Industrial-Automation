"""
Pre-defined Common Tools
Tools that are commonly used and don't need to be created dynamically.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.shared_utils import setup_paths
setup_paths()


class EmptyInput(BaseModel):
    """Empty input for tools with no parameters."""
    pass


class SearchInput(BaseModel):
    """Input for search tools."""
    query: str = Field(description="Search query string")


class CheckDataLoadedTool(BaseTool):
    """Check if datasets are already loaded in memory."""
    
    name: str = "check_data_loaded"
    description: str = """Check if datasets are already loaded and available.
    Use this BEFORE attempting to load datasets to avoid redundant operations.
    Returns list of currently loaded datasets."""
    
    def _run(self) -> str:
        """Check loaded datasets."""
        try:
            from shared.load_data import AVAILABLE_DATASETS, DATA_DIR
            
            loaded = []
            for dataset_name in AVAILABLE_DATASETS:
                dataset_path = DATA_DIR / dataset_name
                if dataset_path.exists():
                    loaded.append(dataset_name)
            
            if loaded:
                return f"Datasets already loaded: {', '.join(loaded)}"
            else:
                return "No datasets currently loaded. Use list_datasets to see available datasets."
        except Exception as e:
            return f"Error checking loaded data: {str(e)}"


class QuickDataSummaryTool(BaseTool):
    """Get quick summary of loaded data without full loading."""
    
    name: str = "quick_data_summary"
    description: str = """Get a quick summary of available data without loading full datasets.
    Returns basic info: dataset names, sizes, and locations."""
    
    def _run(self) -> str:
        """Get data summary."""
        try:
            from shared.load_data import list_available_datasets, DATA_DIR
            
            datasets = list_available_datasets()
            summary = ["Available Datasets:"]
            
            for ds_name in datasets:
                ds_path = DATA_DIR / ds_name
                if ds_path.exists():
                    # Get size
                    size_mb = sum(f.stat().st_size for f in ds_path.rglob('*') if f.is_file()) / (1024*1024)
                    summary.append(f"  - {ds_name}: {size_mb:.1f} MB")
            
            return "\n".join(summary)
        except Exception as e:
            return f"Error getting summary: {str(e)}"


class ValidateToolInputInput(BaseModel):
    """Input for validate_tool_input."""
    tool_name: str = Field(description="Name of the tool to validate input for")
    action_input: str = Field(description="The action input string to validate")


class ValidateToolInputTool(BaseTool):
    """Validate tool input format before calling tools."""
    
    name: str = "validate_tool_input"
    description: str = """Validate that tool input is in correct JSON format.
    Use this to check tool inputs before execution to avoid parsing errors."""
    args_schema: type[BaseModel] = ValidateToolInputInput
    
    def _run(self, tool_name: str, action_input: str) -> str:
        """Validate tool input."""
        import json
        
        try:
            # Try to parse as JSON
            parsed = json.loads(action_input)
            
            # Check if it's a dict
            if not isinstance(parsed, dict):
                return f"ERROR: Tool input must be a JSON object, got {type(parsed).__name__}"
            
            return f"✅ Valid JSON input for {tool_name}: {list(parsed.keys())}"
        except json.JSONDecodeError as e:
            return f"ERROR: Invalid JSON format: {str(e)}\nExpected format: JSON object with parameter keys and values"
        except Exception as e:
            return f"ERROR: Validation failed: {str(e)}"


class SimpleCodeInput(BaseModel):
    """Input for simple code execution tool."""
    code: str = Field(description="Python code to execute (as a plain string, no JSON needed)")


class SimpleCodeExecutionTool(BaseTool):
    """Simple code execution tool that accepts code as a direct string parameter.
    
    This is an alternative to execute_python_code that bypasses JSON parsing issues.
    Use this when execute_python_code fails due to JSON parsing errors.
    """
    
    name: str = "execute_code_simple"
    description: str = """Execute Python code by passing code as a direct string parameter.
    
    This tool is simpler than execute_python_code - it accepts code as a plain string parameter,
    avoiding JSON parsing issues. Use this when execute_python_code fails.
    
    Action Input: {"code": "your Python code here"}
    
    Example:
    Action Input: {"code": "import pandas as pd\\nprint('Hello')"}
    
    The code parameter accepts multi-line code with escaped newlines (\\n).
    """
    args_schema: type[BaseModel] = SimpleCodeInput
    
    def run(self, tool_input: Any = None, **kwargs) -> str:
        """Override run to handle ReactAgent's empty dict issue."""
        import json
        import re
        
        code = None
        
        # Handle empty dict from ReactAgent's failed JSON parsing
        if isinstance(tool_input, dict):
            if 'code' in tool_input and tool_input.get('code'):
                code = tool_input['code']
            elif len(tool_input) == 0:
                # Empty dict - try to recover from agent's scratchpad/json_log FIRST (most reliable)
                agent = getattr(self, 'agent_ref', None)
                if agent:
                    # Try json_log first (contains raw LLM output before ReactAgent processes it)
                    json_log = getattr(agent, 'json_log', [])
                    if json_log:
                        # Check the most recent entry first (most likely to contain current action)
                        for entry in reversed(json_log[-10:]):  # Check last 10 entries
                            # Try raw_llm_action_output first (most reliable - before ReactAgent processing)
                            raw_output = entry.get('raw_llm_action_output', '')
                            # Check if this entry is for execute_code_simple (check action field too)
                            action_field = entry.get('action', '')
                            is_execute_code_simple = (
                                'execute_code_simple' in raw_output or 
                                'execute_code_simple' in action_field or
                                entry.get('action', '').strip() == 'execute_code_simple'
                            )
                            
                            if raw_output and isinstance(raw_output, str) and is_execute_code_simple:
                                # Look for the JSON in the raw output
                                # Pattern: Action Input N: {"code": "..."}
                                # CRITICAL: ReactAgent may have replaced newlines with spaces, so we need flexible matching
                                patterns = [
                                    # Pattern 1: Standard format with escaped newlines (most specific)
                                    r'Action\s+Input\s+\d+:\s*\{[^}]*"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                    # Pattern 2: Match even if newlines were replaced with spaces (more flexible)
                                    r'Action\s+Input\s+\d+:\s*\{[^}]*"code"\s*:\s*"([^"]+)"',
                                    # Pattern 3: Match JSON object with code key (handles multi-line corruption)
                                    r'\{[^}]*"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                    # Pattern 4: Direct code extraction (greedy, handles escaped quotes)
                                    r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                    # Pattern 5: Match with spaces where newlines were (ReactAgent's corruption)
                                    r'"code"\s*:\s*"([^"]+)"',
                                    # Pattern 6: Match code even if JSON structure is broken (last resort)
                                    r'code["\']?\s*:\s*["\']([^"\']+)["\']',
                                ]
                                for pattern in patterns:
                                    match = re.search(pattern, raw_output, re.DOTALL)
                                    if match:
                                        code = match.group(1)
                                        # Handle escaped sequences
                                        code = code.replace('\\n', '\n').replace('\\"', '"').replace('\\r', '\r')
                                        # Unescape if needed
                                        code = code.replace('\\\\', '\\')
                                        # If code contains multiple spaces in a row, they might be replaced newlines
                                        # But be careful - only do this if it looks like code (contains keywords)
                                        if 'import' in code or 'from' in code or 'def' in code:
                                            # Don't replace all spaces, but this is a heuristic
                                            pass
                                        if code.strip():
                                            break
                                if code and code.strip():
                                    break
                            
                            # Try action_input from json_log (might have partial data)
                            action_input = entry.get('action_input', '')
                            if action_input and isinstance(action_input, str) and '"code"' in action_input:
                                # Try to extract code from action_input
                                match = re.search(r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', action_input, re.DOTALL)
                                if match:
                                    code = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\r', '\r')
                                    code = code.replace('\\\\', '\\')
                                    if code.strip():
                                        break
                            
                            # Try action field (might contain the full output)
                            action_field = entry.get('action', '')
                            if action_field and isinstance(action_field, str) and 'execute_code_simple' in action_field:
                                # The action field might contain the full LLM output
                                match = re.search(r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', action_field, re.DOTALL)
                                if match:
                                    code = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\r', '\r')
                                    code = code.replace('\\\\', '\\')
                                    if code.strip():
                                        break
                    
                    # Try scratchpad if json_log didn't work
                    if not code:
                        scratchpad = getattr(agent, 'scratchpad', '')
                        if scratchpad and 'execute_code_simple' in scratchpad:
                            # Look for Action Input with code in scratchpad
                            # CRITICAL: ReactAgent replaces newlines with spaces, so we need flexible matching
                            patterns = [
                                # Pattern 1: Standard format
                                r'Action\s+Input\s+\d+:\s*\{[^}]*"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                # Pattern 2: Match with spaces (ReactAgent corruption)
                                r'Action\s+Input\s+\d+:\s*\{[^}]*"code"\s*:\s*"([^"]+)"',
                                # Pattern 3: Direct code extraction
                                r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                # Pattern 4: Greedy match (last resort)
                                r'"code"\s*:\s*"([^"]+)"',
                            ]
                            for pattern in patterns:
                                match = re.search(pattern, scratchpad, re.DOTALL)
                                if match:
                                    code = match.group(1)
                                    # Handle escaped sequences
                                    code = code.replace('\\n', '\n').replace('\\"', '"').replace('\\r', '\r')
                                    code = code.replace('\\\\', '\\')
                                    # If code looks valid (contains Python keywords), use it
                                    if code.strip() and ('import' in code or 'from' in code or 'def' in code or 'print' in code or '=' in code):
                                        break
                                    else:
                                        code = None  # Reset if doesn't look like code
                            # If still no code, try to find the most recent Action Input for execute_code_simple
                            if not code:
                                # Find the last occurrence of "Action Input" before "execute_code_simple"
                                action_input_matches = list(re.finditer(r'Action\s+Input\s+\d+:\s*(\{[^}]*\})', scratchpad, re.DOTALL))
                                if action_input_matches:
                                    # Get the last match
                                    last_match = action_input_matches[-1]
                                    json_str = last_match.group(1)
                                    # Try to parse it
                                    try:
                                        parsed = json.loads(json_str)
                                        if isinstance(parsed, dict) and 'code' in parsed:
                                            code = parsed['code']
                                    except:
                                        # Try to extract code directly
                                        code_match = re.search(r'"code"\s*:\s*"([^"]+)"', json_str)
                                        if code_match:
                                            code = code_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                
                # Fallback: try to recover from kwargs
                if not code:
                    for key in ['argument', 'action_input', 'input', 'raw_input']:
                        raw_string = kwargs.get(key)
                        if raw_string and isinstance(raw_string, str) and raw_string.strip():
                            try:
                                parsed = json.loads(raw_string)
                                if isinstance(parsed, dict) and 'code' in parsed:
                                    code = parsed['code']
                                    break
                            except:
                                # Try to extract code directly from string
                                if '"code"' in raw_string:
                                    match = re.search(r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_string, re.DOTALL)
                                    if match:
                                        code = match.group(1).replace('\\n', '\n').replace('\\"', '"')
                                        break
                
                if code is None:
                    return "ERROR: Empty JSON input received. ReactAgent's JSON parsing produced an empty dict {}. Recovery from scratchpad/json_log failed. Please use single-line JSON format: {\"code\": \"your code here\"}"
            else:
                # Non-empty dict but missing 'code' key - try to recover from agent's scratchpad/json_log
                agent = getattr(self, 'agent_ref', None)
                if agent:
                    # Try to extract from scratchpad
                    scratchpad = getattr(agent, 'scratchpad', '')
                    if scratchpad:
                        # Look for Action Input with code
                        patterns = [
                            r'Action Input\s+\d+:\s*\{[^}]*"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                            r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                            r'Action Input\s+\d+:\s*"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, scratchpad, re.DOTALL)
                            if match:
                                code = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\r', '\r')
                                break
                    
                    # Try json_log if scratchpad didn't work
                    if not code:
                        json_log = getattr(agent, 'json_log', [])
                        if json_log:
                            for entry in reversed(json_log[-5:]):  # Check last 5 entries
                                raw_output = entry.get('raw_llm_action_output', '')
                                if raw_output:
                                    match = re.search(r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_output, re.DOTALL)
                                    if match:
                                        code = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\r', '\r')
                                        break
                
                if code is None:
                    return f"ERROR: Missing 'code' key. Received keys: {list(tool_input.keys())}. ReactAgent's JSON parsing may have corrupted the input. Try using execute_python_code instead."
        elif isinstance(tool_input, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(tool_input)
                if isinstance(parsed, dict) and 'code' in parsed:
                    code = parsed['code']
                else:
                    return f"ERROR: Invalid JSON format. Expected {{\"code\": \"...\"}}, got {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}"
            except:
                # Treat as plain code
                code = tool_input
        else:
            return f"ERROR: Invalid input type. Expected dict or str, got {type(tool_input).__name__}"
        
        if code is None:
            return "ERROR: Could not extract code from input."
        
        return self._run(code)
    
    def set_agent_ref(self, agent):
        """Set agent reference for accessing scratchpad/json_log."""
        object.__setattr__(self, 'agent_ref', agent)
    
    def _run(self, code: str) -> str:
        """Execute Python code."""
        import os
        import sys
        import json
        from pathlib import Path
        
        if not code or not code.strip():
            return "ERROR: Code parameter is required and cannot be empty."
        
        try:
            # Unescape newlines if needed
            code = code.replace('\\n', '\n')
            
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
            
            if 'result' in safe_globals:
                result = str(safe_globals['result'])
                return result
            else:
                return "Code executed successfully (no result variable set)."
                
        except SyntaxError as e:
            return f"ERROR: Syntax error: {str(e)}"
        except NameError as e:
            return f"ERROR: Name error: {str(e)}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {str(e)}"


def create_common_tools() -> list[BaseTool]:
    """Create list of common pre-defined tools."""
    return [
        CheckDataLoadedTool(),
        QuickDataSummaryTool(),
        ValidateToolInputTool(),
        SimpleCodeExecutionTool(),
    ]

