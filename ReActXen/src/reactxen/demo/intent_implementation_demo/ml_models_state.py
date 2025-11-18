"""
ML Models State - Global state management for trained models.
"""
from typing import Dict, Any

# Global state for trained models
trained_models: Dict[str, Any] = {}
model_metadata: Dict[str, Any] = {}


def get_trained_model(model_id: str) -> Any:
    """Get a trained model by ID."""
    return trained_models.get(model_id)


def store_model(model_id: str, model: Any, metadata: Dict[str, Any]):
    """Store a trained model and its metadata."""
    trained_models[model_id] = model
    model_metadata[model_id] = metadata


def list_all_models() -> Dict[str, Any]:
    """List all trained models."""
    return model_metadata.copy()

