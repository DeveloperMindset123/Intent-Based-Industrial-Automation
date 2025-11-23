"""
JSON Parser Handler - Robust JSON parsing for agent tool inputs.
Handles multi-line JSON, malformed JSON, and various input formats.
"""
import json
import re
from typing import Any, Dict, Optional


def parse_json_input(input_data: Any, tool_name: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Robust JSON parser that handles various input formats.
    
    Args:
        input_data: The input data (string, dict, or other)
        tool_name: Name of the tool for error reporting
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    # If already a dict, return it
    if isinstance(input_data, dict):
        return input_data
    
    # If None or empty, return None
    if not input_data:
        return None
    
    # Convert to string if not already
    if not isinstance(input_data, str):
        input_data = str(input_data)
    
    # Strip whitespace
    input_data = input_data.strip()
    
    # If empty after stripping, return None
    if not input_data:
        return None
    
    # Strategy 1: Try direct JSON parsing
    if input_data.startswith('{') and input_data.endswith('}'):
        try:
            parsed = json.loads(input_data)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Strategy 1b: Try JSON parsing with newlines already replaced with spaces (ReactAgent does this)
    # ReactAgent replaces \n with spaces in SINGLE_LINE_TOOL_CALL mode
    # So we need to handle JSON strings that have had newlines normalized
    if input_data.startswith('{') and input_data.endswith('}'):
        try:
            # Try to fix common issues: unescaped newlines in string values
            # Replace spaces that might be newlines in code strings
            fixed_input = input_data
            # Try parsing as-is first
            try:
                parsed = json.loads(fixed_input)
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # If that fails, try to reconstruct proper JSON by handling \\n sequences
            # This handles cases where ReactAgent replaced \n with spaces in the JSON string itself
            fixed_input = re.sub(r'"code"\s*:\s*"([^"]*)"', lambda m: f'"code": {json.dumps(m.group(1))}', input_data)
            try:
                parsed = json.loads(fixed_input)
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except (json.JSONDecodeError, AttributeError):
                pass
        except Exception:
            pass
    
    # Strategy 2: Try to fix common issues and parse
    # Remove trailing commas before closing braces/brackets
    fixed_input = re.sub(r',(\s*[}\]])', r'\1', input_data)
    if fixed_input != input_data:
        try:
            return json.loads(fixed_input)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Extract JSON from multi-line strings (handles ReactAgent's newline replacement)
    # ReactAgent replaces newlines with spaces in SINGLE_LINE_TOOL_CALL mode
    # So we need to handle JSON strings where newlines inside code strings are replaced with spaces
    # Look for JSON-like patterns in the string
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, input_data, re.DOTALL)
    if matches:
        # Try the longest match (most likely to be complete JSON)
        matches.sort(key=len, reverse=True)
        for match in matches:
            try:
                # Try parsing as-is first
                parsed = json.loads(match)
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except json.JSONDecodeError:
                # If parsing fails, try to fix common issues
                # Handle case where ReactAgent replaced newlines in code strings with spaces
                # We can't perfectly reconstruct, but we can try to fix escape sequences
                try:
                    # Try to fix escaped newlines that might have been converted to spaces
                    fixed_match = match.replace('\\n', '\n').replace('\\r', '\r')
                    # But this might break things - better to try extracting the code value directly
                    code_match = re.search(r'"code"\s*:\s*"([^"]*)"', match, re.DOTALL)
                    if code_match:
                        code_value = code_match.group(1)
                        # Try to reconstruct JSON with the code value
                        fixed_json = match.replace(f'"code": "{code_value}"', f'"code": {json.dumps(code_value)}')
                        try:
                            parsed = json.loads(fixed_json)
                            if isinstance(parsed, dict) and parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    continue
    
    # Strategy 4: Try to extract key-value pairs from formatted text (CRITICAL for ReactAgent)
    # Handle cases like: {"code": "...", "return_result": true}
    # Even if there are newlines inside the values (ReactAgent replaces \n with spaces)
    # Try to match code string even if it has broken quotes due to newline replacement
    code_patterns = [
        # Pattern 1: Standard JSON string with escaped sequences
        r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        # Pattern 2: Code string that might be broken by ReactAgent's newline replacement
        # ReactAgent replaces \n with spaces, so we need to match across spaces
        r'"code"\s*:\s*"([^"]+)"',
        # Pattern 3: Code might be outside quotes if JSON is very broken
        r'"code"\s*:\s*"([^"]*?)"(?:\s*[,}])',
    ]
    
    for pattern in code_patterns:
        code_match = re.search(pattern, input_data, re.DOTALL)
        if code_match:
            code_value = code_match.group(1)
            # Try to restore newlines (if they were replaced with spaces)
            # This is a best-effort reconstruction
            code_value = code_value.replace('\\n', '\n').replace('\\r', '\r')
            # If code contains patterns like "import pandas  # comment  data = ..."
            # where spaces were newlines, we can't perfectly restore, but we can use it as-is
            result = {"code": code_value}
            
            # Try to find return_result
            return_result_match = re.search(r'"return_result"\s*:\s*(true|false)', input_data, re.IGNORECASE)
            if return_result_match:
                result["return_result"] = return_result_match.group(1).lower() == 'true'
            else:
                result["return_result"] = True  # Default
            
            if code_value.strip():  # Only return if we found actual code
                return result
    
    # Strategy 4b: Try to extract code from very broken JSON (last resort)
    # Look for code patterns even if JSON structure is broken
    code_content_match = re.search(r'"code"\s*:\s*"([^"]+)"', input_data)
    if code_content_match:
        code_value = code_content_match.group(1)
        # Basic cleanup
        code_value = code_value.replace('\\"', '"').replace('\\n', '\n').replace('  ', '\n')
        if code_value.strip():
            return {"code": code_value, "return_result": True}
    
    # Strategy 5: Try to find code in triple-quoted strings or code blocks
    # Handle Python code blocks
    code_block_patterns = [
        r'```python\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'"""\s*(.*?)\s*"""',
        r"'''\s*(.*?)\s*'''",
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, input_data, re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            return {"code": code_content, "return_result": True}
    
    # Strategy 6: If the input looks like Python code (not JSON), wrap it
    # Check if it contains Python keywords or syntax
    python_keywords = ['import ', 'def ', 'class ', 'if ', 'for ', 'while ', 'return ', 'print(']
    if any(keyword in input_data for keyword in python_keywords):
        return {"code": input_data, "return_result": True}
    
    # Strategy 7: Try parsing as a single-line JSON with escaped newlines
    try:
        # Replace actual newlines with \n in the string
        escaped_input = input_data.replace('\n', '\\n').replace('\r', '\\r')
        return json.loads(escaped_input)
    except json.JSONDecodeError:
        pass
    
    # Strategy 8: Last resort - if it's a single string that might be code
    # Check if it's likely code (contains common Python patterns)
    if len(input_data) > 50 and ('=' in input_data or '(' in input_data or '[' in input_data):
        return {"code": input_data, "return_result": True}
    
    # If all strategies fail, return None
    return None


def normalize_json_input(input_data: Any, expected_keys: Optional[list] = None) -> Dict[str, Any]:
    """
    Parse and normalize JSON input, ensuring required keys are present.
    
    Args:
        input_data: The input data to parse
        expected_keys: List of expected keys (e.g., ['code', 'return_result'])
        
    Returns:
        Normalized dictionary
        
    Raises:
        ValueError: If parsing fails or required keys are missing
    """
    parsed = parse_json_input(input_data)
    
    if parsed is None:
        raise ValueError(f"Failed to parse input: {input_data}")
    
    # Ensure expected keys exist
    if expected_keys:
        missing_keys = [key for key in expected_keys if key not in parsed]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}. Received: {list(parsed.keys())}")
    
    return parsed


def extract_code_from_input(input_data: Any) -> str:
    """
    Extract Python code from various input formats.
    
    Args:
        input_data: Input data (string, dict, etc.)
        
    Returns:
        Extracted Python code as string
    """
    parsed = parse_json_input(input_data)
    
    if parsed and 'code' in parsed:
        return str(parsed['code'])
    
    # If parsing failed, try to extract code directly
    if isinstance(input_data, str):
        # Remove JSON wrapper if present
        code_match = re.search(r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', input_data, re.DOTALL)
        if code_match:
            return code_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        
        # Try code block patterns
        code_block_patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]
        for pattern in code_block_patterns:
            match = re.search(pattern, input_data, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If it looks like code, return as-is
        if any(keyword in input_data for keyword in ['import ', 'def ', 'class ', 'return ']):
            return input_data
    
    # Fallback: return string representation
    return str(input_data) if input_data else ""

