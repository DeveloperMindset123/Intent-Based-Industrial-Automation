"""
ML Prediction Tools - Common tools for prediction and evaluation across all ML frameworks.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List
import numpy as np
import pandas as pd

from ml_models_state import get_trained_model, list_all_models
from ml_data_prep import prepare_prediction_data


class EmptyInput(BaseModel):
    pass


class PredictWithModelInput(BaseModel):
    model_id: str = Field(description="ID of the trained model")
    data_source: Optional[str] = Field(default="test", description="Data source: 'test' or 'train'")


class EvaluateModelInput(BaseModel):
    model_id: str = Field(description="ID of the trained model")
    metrics: Optional[str] = Field(default=None, description="Comma-separated list of metrics to compute")


class PredictWithMLModelTool(BaseTool):
    """Make predictions using a trained ML model."""
    
    name: str = "predict_with_ml_model"
    description: str = """Make predictions using a trained model (scikit-learn, PyTorch, or TensorFlow).
    Requires model_id from training step."""
    args_schema: Type[BaseModel] = PredictWithModelInput
    
    def _run(self, model_id: str, data_source: str = "test") -> str:
        try:
            model_info = get_trained_model(model_id)
            if not model_info:
                return f"Error: Model {model_id} not found. Train a model first."
            
            X = prepare_prediction_data(data_source)
            if X is None:
                return "Error: No data available for prediction."
            
            model = model_info['model']
            model_type = model_info['type']
            
            # Make predictions based on framework
            if model_type == 'sklearn':
                scaler = model_info.get('scaler')
                if scaler:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X.values
                predictions = model.predict(X_scaled)
            elif model_type == 'pytorch':
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X.values)
                    predictions = model(X_tensor).numpy().flatten()
            elif model_type == 'tensorflow':
                predictions = model.predict(X, verbose=0).flatten()
            else:
                return f"Error: Unknown model type {model_type}"
            
            # Store predictions globally
            import tools_logic as tl
            tl.predictions = predictions
            
            return f"✅ Predictions generated for {len(predictions)} samples using {model_id}. Range: {predictions.min():.2f}-{predictions.max():.2f}"
            
        except Exception as e:
            return f"Error making predictions: {str(e)}"


class EvaluateMLModelTool(BaseTool):
    """Evaluate a trained ML model using various metrics."""
    
    name: str = "evaluate_ml_model"
    description: str = """Evaluate a trained model using metrics like MAE, RMSE, R2 score.
    Requires model_id and optionally ground truth data."""
    args_schema: Type[BaseModel] = EvaluateModelInput
    
    def _run(self, model_id: str, metrics: Optional[str] = None) -> str:
        try:
            if not get_trained_model(model_id):
                return f"Error: Model {model_id} not found."
            
            import tools_logic as tl
            if tl.predictions is None:
                return "Error: No predictions available. Run predict_with_ml_model first."
            
            predictions = tl.predictions
            ground_truth = tl.ground_truth
            
            if ground_truth is None:
                return "Error: No ground truth available for evaluation."
            
            # Get ground truth values
            if isinstance(ground_truth, pd.DataFrame):
                y_true = ground_truth['RUL'].values if 'RUL' in ground_truth.columns else ground_truth.iloc[:, 0].values
            else:
                y_true = ground_truth
            
            # Align lengths
            min_len = min(len(predictions), len(y_true))
            predictions = predictions[:min_len]
            y_true = y_true[:min_len]
            
            # Compute metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            r2 = r2_score(y_true, predictions)
            
            result = f"✅ Model Evaluation Results for {model_id}:\n"
            result += f"   MAE: {mae:.4f}\n"
            result += f"   RMSE: {rmse:.4f}\n"
            result += f"   R² Score: {r2:.4f}\n"
            
            return result
            
        except Exception as e:
            return f"Error evaluating model: {str(e)}"


class ListTrainedModelsTool(BaseTool):
    """List all trained models and their metadata."""
    
    name: str = "list_trained_models"
    description: str = "List all trained models with their IDs and metadata."
    args_schema: Type[BaseModel] = EmptyInput
    
    def _run(self) -> str:
        metadata = list_all_models()
        if not metadata:
            return "No trained models found. Train a model first."
        
        result = f"📊 Trained Models ({len(metadata)}):\n\n"
        for model_id, meta in metadata.items():
            result += f"  • {model_id}\n"
            result += f"    Framework: {meta.get('framework', 'unknown')}\n"
            result += f"    Type: {meta.get('model_type', meta.get('architecture', 'unknown'))}\n"
            result += f"    Training Samples: {meta.get('training_samples', 'N/A')}\n"
            if 'epochs' in meta:
                result += f"    Epochs: {meta['epochs']}\n"
            result += "\n"
        
        return result


def create_ml_prediction_tools() -> List[BaseTool]:
    """Create ML prediction and evaluation tools."""
    return [PredictWithMLModelTool(), EvaluateMLModelTool(), ListTrainedModelsTool()]

