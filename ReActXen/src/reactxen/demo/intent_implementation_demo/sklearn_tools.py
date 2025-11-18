"""
Scikit-learn ML Tools - Tools for training and using scikit-learn models.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List
import json
import numpy as np

from ml_models_state import store_model
from ml_data_prep import prepare_training_data


class TrainSklearnModelInput(BaseModel):
    model_type: str = Field(description="Model type: 'random_forest', 'linear_regression', 'svr', 'gradient_boosting'")
    task_type: str = Field(default="regression", description="Task type: 'regression' or 'classification'")
    hyperparameters: Optional[str] = Field(default=None, description="JSON string of hyperparameters")


class TrainSklearnModelTool(BaseTool):
    """Train a scikit-learn model for RUL prediction."""
    
    name: str = "train_sklearn_model"
    description: str = """Train a scikit-learn model (Random Forest, Linear Regression, SVR, Gradient Boosting).
    Returns a model_id that can be used for prediction."""
    args_schema: Type[BaseModel] = TrainSklearnModelInput
    
    def _run(
        self,
        model_type: str,
        task_type: str = "regression",
        hyperparameters: Optional[str] = None
    ) -> str:
        try:
            X, y = prepare_training_data()
            if X is None or y is None:
                return "Error: Load dataset first using load_dataset tool."
            
            # Import scikit-learn
            try:
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import LinearRegression
                from sklearn.svm import SVR
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                return "Error: scikit-learn not installed. Install with: pip install scikit-learn"
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Parse hyperparameters
            params = {}
            if hyperparameters:
                try:
                    params = json.loads(hyperparameters)
                except:
                    pass
            
            # Train model based on type
            model_type_lower = model_type.lower()
            if model_type_lower == 'random_forest':
                model = RandomForestRegressor(n_estimators=params.get('n_estimators', 100), random_state=42)
            elif model_type_lower == 'linear_regression':
                model = LinearRegression()
            elif model_type_lower == 'svr':
                model = SVR(kernel=params.get('kernel', 'rbf'), C=params.get('C', 1.0))
            elif model_type_lower == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=params.get('n_estimators', 100), random_state=42)
            else:
                return f"Error: Unknown model type '{model_type}'. Supported: random_forest, linear_regression, svr, gradient_boosting"
            
            # Train
            model.fit(X_scaled, y)
            
            # Store model
            from ml_models_state import trained_models
            model_id = f"sklearn_{model_type}_{len(trained_models)}"
            store_model(model_id, {
                'model': model,
                'scaler': scaler,
                'type': 'sklearn',
                'model_type': model_type
            }, {
                'framework': 'scikit-learn',
                'model_type': model_type,
                'task_type': task_type,
                'training_samples': len(X)
            })
            
            return f"✅ Trained scikit-learn {model_type} model. Model ID: {model_id}"
            
        except Exception as e:
            return f"Error training scikit-learn model: {str(e)}"


def create_sklearn_tools() -> List[BaseTool]:
    """Create scikit-learn tools."""
    return [TrainSklearnModelTool()]

