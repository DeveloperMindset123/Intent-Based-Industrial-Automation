"""
PyTorch ML Tools - Tools for training and using PyTorch models.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List
import json

from ml_models_state import store_model, trained_models
from ml_data_prep import prepare_training_data


class TrainPyTorchModelInput(BaseModel):
    model_architecture: str = Field(description="Model architecture: 'mlp', 'lstm', 'cnn', 'transformer'")
    input_size: Optional[int] = Field(default=None, description="Input feature size")
    hidden_sizes: Optional[str] = Field(default=None, description="JSON array of hidden layer sizes")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    epochs: int = Field(default=10, description="Number of training epochs")


class TrainPyTorchModelTool(BaseTool):
    """Train a PyTorch model for RUL prediction."""
    
    name: str = "train_pytorch_model"
    description: str = """Train a PyTorch model (MLP, LSTM, CNN, Transformer) for RUL prediction.
    Returns a model_id that can be used for prediction."""
    args_schema: Type[BaseModel] = TrainPyTorchModelInput
    
    def _run(
        self,
        model_architecture: str,
        input_size: Optional[int] = None,
        hidden_sizes: Optional[str] = None,
        learning_rate: float = 0.001,
        epochs: int = 10
    ) -> str:
        try:
            X, y = prepare_training_data()
            if X is None or y is None:
                return "Error: Load dataset first using load_dataset tool."
            
            # Import PyTorch
            try:
                import torch
                import torch.nn as nn
                import torch.optim as optim
            except ImportError:
                return "Error: PyTorch not installed. Install with: pip install torch"
            
            # Determine input size
            if input_size is None:
                input_size = X.shape[1]
            
            # Parse hidden sizes
            if hidden_sizes:
                try:
                    hidden_layers = json.loads(hidden_sizes)
                except:
                    hidden_layers = [64, 32]
            else:
                hidden_layers = [64, 32]
            
            # Create model architecture
            arch = model_architecture.lower()
            if arch == 'mlp':
                layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
                for i in range(len(hidden_layers) - 1):
                    layers.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU()])
                layers.append(nn.Linear(hidden_layers[-1], 1))
                model = nn.Sequential(*layers)
            else:
                # Default MLP
                layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
                for i in range(len(hidden_layers) - 1):
                    layers.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU()])
                layers.append(nn.Linear(hidden_layers[-1], 1))
                model = nn.Sequential(*layers)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X.values)
            y_tensor = torch.FloatTensor(y).reshape(-1, 1)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            # Store model
            model_id = f"pytorch_{arch}_{len(trained_models)}"
            store_model(model_id, {
                'model': model,
                'type': 'pytorch',
                'architecture': arch,
                'input_size': input_size
            }, {
                'framework': 'pytorch',
                'architecture': arch,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'training_samples': len(X)
            })
            
            return f"✅ Trained PyTorch {arch} model. Model ID: {model_id}"
            
        except Exception as e:
            return f"Error training PyTorch model: {str(e)}"


def create_pytorch_tools() -> List[BaseTool]:
    """Create PyTorch tools."""
    return [TrainPyTorchModelTool()]

