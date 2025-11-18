"""
ML Data Preparation - Common data preparation utilities for ML frameworks.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def prepare_training_data() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Prepare training data from global state.
    Returns (X, y) tuple.
    """
    try:
        import tools_logic as tl
        if tl.train_data is None or (hasattr(tl.train_data, 'empty') and tl.train_data.empty):
            return None, None
        
        data = tl.train_data.copy()
        
        # Assume RUL is the target if it exists, otherwise use last column
        if 'RUL' in data.columns:
            y = data['RUL'].values
            X = data.drop(['RUL'], axis=1).select_dtypes(include=[np.number])
        else:
            y = data.iloc[:, -1].values
            X = data.iloc[:, :-1].select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y
    except Exception:
        return None, None


def prepare_prediction_data(data_source: str = "test") -> Optional[pd.DataFrame]:
    """Prepare data for prediction."""
    try:
        import tools_logic as tl
        data = tl.test_data if data_source == "test" and tl.test_data is not None else tl.train_data
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            return None
        
        # Prepare features
        if 'RUL' in data.columns:
            X = data.drop(['RUL'], axis=1).select_dtypes(include=[np.number])
        else:
            X = data.select_dtypes(include=[np.number])
        
        X = X.fillna(X.mean())
        return X
    except Exception:
        return None

