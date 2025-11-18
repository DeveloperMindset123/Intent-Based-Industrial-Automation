"""
ML Framework Tools - Factory module that combines all ML framework tools.
"""
from typing import List
from langchain_core.tools import BaseTool

from sklearn_tools import create_sklearn_tools
from pytorch_tools import create_pytorch_tools
from tensorflow_tools import create_tensorflow_tools
from ml_prediction_tools import create_ml_prediction_tools


def create_ml_framework_tools() -> List[BaseTool]:
    """Create all ML framework tools."""
    tools = []
    tools.extend(create_sklearn_tools())
    tools.extend(create_pytorch_tools())
    tools.extend(create_tensorflow_tools())
    tools.extend(create_ml_prediction_tools())
    return tools

