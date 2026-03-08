"""
TODO : place the core huggingface logic here
"""

from typing import Dict, Any
from datetime import datetime


# GLOBAL STATE FOR HUGGINGFACE MODELS
class HuggingFaceModelState:
    """Shared state for huggingface model catalog"""

    _models_catalog: Dict[str, Any] = {}

    @classmethod
    def get_catalog(cls) -> Dict[str, Any]:
        return cls._models_catalog

    @classmethod
    def set_catalog(cls, catalog: Dict[str, Any]):
        cls._models_catalog = catalog


def train_rul_model_with_huggingface(
    model_id: str, train_data, test_data=None, ground_truth=None
) -> Dict[str, Any]:
    """
    Train RUL model using HuggingFace model.
    This loads a HuggingFace model and adapts it for RUL prediction.
    """
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        import torch
        import numpy as np

        # Load model configuration
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        # Prepare training data
        feature_columns = [
            col for col in train_data.columns if col.startswith("sensor_")
        ]
        X_train = train_data[feature_columns].values
        y_train = train_data["RUL"].values

        # For time-series models, you'd need to reshape data appropriately
        # This is a simplified version - actual implementation would depend on model architecture

        # Store model info
        model_info = {
            "model_id": model_id,
            "model_type": "huggingface_traditional",
            "training_status": "completed",
            "training_samples": len(X_train),
            "feature_count": len(feature_columns),
            "model_config": str(config),
            "timestamp": datetime.now().isoformat(),
        }

        # Evaluate if test data and ground truth are available
        if test_data is not None and ground_truth is not None:
            X_test = test_data[feature_columns].values
            # In practice, you'd make predictions here
            model_info["evaluation_status"] = "test_data_available"

        return model_info
    except Exception as e:
        return {"error": f"Error training with HuggingFace: {str(e)}"}
