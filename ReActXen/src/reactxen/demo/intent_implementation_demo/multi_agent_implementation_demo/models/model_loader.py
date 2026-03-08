from transformers import AutoModel, AutoModelForSequenceClassification
import os
import sys
from pathlib import Path

# Add ReActXen/src to path so reactxen can be imported
# From: ReActXen/src/reactxen/demo/.../models/model_loader.py
# To:   ReActXen/src/ (go up 6 levels)
_reactxen_src = Path(__file__).parent.parent.parent.parent.parent.parent
if str(_reactxen_src) not in sys.path:
    sys.path.insert(0, str(_reactxen_src))

def get_model(model_id: str, model_source: str = "watsonx", task: str = None):
    """Dynamically load models from HuggingFace or WatsonX catalogs."""
    if model_source == 'huggingface':
        try:
            if task == "classification":
                return AutoModelForSequenceClassification.from_pretrained(model_id)
            return AutoModel.from_pretrained(model_id)
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace model {model_id}: {e}")
    elif model_source == 'watsonx':
        # Lazy import to avoid loading at module import time
        from reactxen.utils.watsonx_llm import get_llm
        from reactxen.utils.model_inference import modelset
        if isinstance(model_id, int) and 0 <= model_id < len(modelset):
            return get_llm(modelset[model_id])
        elif model_id in modelset:
            return get_llm(model_id)
        else:
            return get_llm(model_id)
    else:
        raise ValueError(f"Invalid model source: {model_source}")

def list_available_models(source: str = "watsonx", task: str = None):
    """List available models from HuggingFace or WatsonX catalogs."""
    if source == 'watsonx':
        from reactxen.utils.model_inference import modelset
        return {i: model for i, model in enumerate(modelset)}
    elif source == 'huggingface':
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            if task == "rul":
                models = api.list_models(search="time-series", task="regression")
            elif task == "classification":
                models = api.list_models(search="time-series", task="classification")
            else:
                models = api.list_models(search="time-series")
            return {m.id: m.id for m in list(models)[:20]}
        except ImportError:
            return {"error": "huggingface_hub not installed"}
    return {}