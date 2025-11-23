"""
Shared execution namespace for persisting variables across code execution calls.
This allows variables to persist between execute_python_code and execute_code_simple calls.
"""
import sys
from typing import Dict, Any

# Global shared namespace - persists across all code execution calls
_shared_namespace: Dict[str, Any] = {
    '__builtins__': __builtins__,
}

def get_shared_namespace() -> Dict[str, Any]:
    """Get the shared execution namespace."""
    return _shared_namespace

def reset_shared_namespace():
    """Reset the shared namespace (useful for testing)."""
    global _shared_namespace
    _shared_namespace = {
        '__builtins__': __builtins__,
    }

def update_shared_namespace(updates: Dict[str, Any]):
    """Update the shared namespace with new variables."""
    _shared_namespace.update(updates)

def get_variable(name: str, default: Any = None) -> Any:
    """Get a variable from the shared namespace."""
    return _shared_namespace.get(name, default)

def set_variable(name: str, value: Any):
    """Set a variable in the shared namespace."""
    _shared_namespace[name] = value

