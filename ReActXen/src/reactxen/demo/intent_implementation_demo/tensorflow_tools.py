"""
TensorFlow ML Tools - Tools for training and using TensorFlow/Keras models.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List

from ml_models_state import store_model, trained_models
from ml_data_prep import prepare_training_data


class TrainTensorFlowModelInput(BaseModel):
    model_type: str = Field(description="Model type: 'sequential', 'functional', 'lstm', 'cnn'")
    layers: Optional[str] = Field(default=None, description="JSON array of layer configurations")
    optimizer: str = Field(default="adam", description="Optimizer: 'adam', 'sgd', 'rmsprop'")
    loss: str = Field(default="mse", description="Loss function")
    epochs: int = Field(default=10, description="Number of training epochs")


class TrainTensorFlowModelTool(BaseTool):
    """Train a TensorFlow/Keras model for RUL prediction."""
    
    name: str = "train_tensorflow_model"
    description: str = """Train a TensorFlow/Keras model for RUL prediction.
    Returns a model_id that can be used for prediction."""
    args_schema: Type[BaseModel] = TrainTensorFlowModelInput
    
    def _run(
        self,
        model_type: str,
        layers: Optional[str] = None,
        optimizer: str = "adam",
        loss: str = "mse",
        epochs: int = 10
    ) -> str:
        try:
            X, y = prepare_training_data()
            if X is None or y is None:
                return "Error: Load dataset first using load_dataset tool."
            
            # Import TensorFlow
            try:
                import tensorflow as tf
                from tensorflow import keras
            except ImportError:
                return "Error: TensorFlow not installed. Install with: pip install tensorflow"
            
            # Build model
            model_type_lower = model_type.lower()
            if model_type_lower == 'sequential':
                model = keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dense(1)
                ])
            elif model_type_lower == 'lstm':
                model = keras.Sequential([
                    keras.layers.LSTM(50, input_shape=(X.shape[1], 1)),
                    keras.layers.Dense(1)
                ])
            else:
                # Default sequential
                model = keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dense(1)
                ])
            
            # Compile
            opt_map = {'adam': 'adam', 'sgd': 'sgd', 'rmsprop': 'rmsprop'}
            model.compile(optimizer=opt_map.get(optimizer.lower(), 'adam'), loss=loss)
            
            # Train
            model.fit(X, y, epochs=epochs, verbose=0)
            
            # Store model
            model_id = f"tensorflow_{model_type_lower}_{len(trained_models)}"
            store_model(model_id, {
                'model': model,
                'type': 'tensorflow',
                'model_type': model_type_lower
            }, {
                'framework': 'tensorflow',
                'model_type': model_type_lower,
                'epochs': epochs,
                'optimizer': optimizer,
                'loss': loss,
                'training_samples': len(X)
            })
            
            return f"✅ Trained TensorFlow {model_type_lower} model. Model ID: {model_id}"
            
        except Exception as e:
            return f"Error training TensorFlow model: {str(e)}"


def create_tensorflow_tools() -> List[BaseTool]:
    """Create TensorFlow tools."""
    return [TrainTensorFlowModelTool()]

