"""
Data Scientist Agent - Handles ML model training and evaluation.
"""
from typing import List, Any
from shared_utils import get_dataset_tools
from tools_logic import create_watsonx_tools
from ml_framework_tools import create_ml_framework_tools
from sub_agent_base import create_sub_agent


def create_data_scientist_agent(react_llm_model_id: int = 15) -> Any:
    """Create a Data Scientist sub-agent for ML model development."""
    tools = []
    tools.extend(get_dataset_tools())
    tools.extend(create_ml_framework_tools())
    tools.extend(create_watsonx_tools())
    
    # Add HuggingFace tools if available
    try:
        from tools_logic import create_huggingface_tools
        hf_tools = create_huggingface_tools()
        if hf_tools:
            tools.extend(hf_tools)
    except (ImportError, AttributeError):
        pass
    
    role = """Data Scientist Agent specialized in machine learning model development.
- Load and preprocess datasets
- Train models using multiple frameworks (scikit-learn, PyTorch, TensorFlow, HuggingFace, WatsonX)
- Evaluate model performance
- Select optimal models based on metrics"""
    
    workflow = """1. Load dataset using load_dataset tool with dataset_name parameter (e.g., "CWRU" or "Azure")
2. Initialize WatsonX API using initialize_watsonx_api tool if not already initialized
3. Train models using appropriate tools:
   - For scikit-learn: use train_sklearn_model
   - For PyTorch: use train_pytorch_model
   - For TensorFlow: use train_tensorflow_model
   - For WatsonX: use train_model (requires dataset to be loaded first)
4. Make predictions using predict_with_ml_model or predict_rul (for WatsonX)
5. Evaluate models using evaluate_ml_model
6. Compare models and select the best one"""
    
    return create_sub_agent(
        question="Train and evaluate ML models for RUL prediction",
        key="data_scientist_agent",
        role=role,
        workflow=workflow,
        tools=tools,
        max_steps=15,
        react_llm_model_id=react_llm_model_id
    )

