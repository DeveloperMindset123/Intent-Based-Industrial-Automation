"""Simplified data loading for RUL prediction."""
import pandas as pd
from pathlib import Path
from typing import Optional
from datasets import load_dataset, DatasetDict
import json

DATA_DIR = Path(__file__).parent / "downloaded_datasets"
DATA_DIR.mkdir(exist_ok=True)

# Global data storage
train_data = None
test_data = None
ground_truth = None

def load_saved_dataset(dataset_name: str, split: Optional[str] = None, format: str = "pandas"):
    """Load a saved dataset from local storage."""
    dataset_path = DATA_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found. Download it first.")
    
    if format == "pandas":
        if split:
            csv_path = dataset_path / split / f"{split}.csv"
            if csv_path.exists():
                return pd.read_csv(csv_path)
        # Try to find any CSV
        for csv_file in dataset_path.rglob("*.csv"):
            return pd.read_csv(csv_file)
    else:
        dataset_path_full = dataset_path / "dataset"
        if dataset_path_full.exists():
            dataset = DatasetDict.load_from_disk(str(dataset_path_full))
            if split and isinstance(dataset, DatasetDict):
                return dataset[split]
            return dataset
    raise ValueError(f"Could not load dataset {dataset_name}")

def list_available_datasets():
    """List all available datasets."""
    if not DATA_DIR.exists():
        return []
    datasets = []
    for item in DATA_DIR.iterdir():
        if item.is_dir() and (item / "metadata.json").exists():
                datasets.append(item.name)
    return sorted(datasets)

def get_dataset_info(dataset_name: str):
    """Get dataset metadata."""
    metadata_file = DATA_DIR / dataset_name / "metadata.json"
    if not metadata_file.exists():
        return {"error": f"Dataset {dataset_name} not found"}
    with open(metadata_file, 'r') as f:
        return json.load(f)

def load_dataset_for_analysis(dataset_name: str):
    """Load dataset and automatically split if no test data exists."""
    global train_data, test_data, ground_truth
    
    try:
        info = get_dataset_info(dataset_name)
        if 'error' in info:
            return f"Error: {info['error']}"
        
        # Load train and test splits if available
        if 'splits' in info and info['splits']:
            if 'train' in info['splits']:
                train_data = load_saved_dataset(dataset_name, split='train', format='pandas')
            if 'test' in info['splits']:
                test_data = load_saved_dataset(dataset_name, split='test', format='pandas')
                if 'RUL' in test_data.columns:
                    ground_truth = test_data[['RUL']].copy()
        else:
            # No splits - load full dataset
            train_data = load_saved_dataset(dataset_name, format='pandas')
        
        # Auto-split if no test data exists
        if test_data is None or (hasattr(test_data, 'empty') and test_data.empty):
            if train_data is not None and len(train_data) > 100:
                split_idx = int(len(train_data) * 0.8)
                test_data = train_data.iloc[split_idx:].copy()
                train_data = train_data.iloc[:split_idx].copy()
                if 'RUL' in test_data.columns:
                    ground_truth = test_data[['RUL']].copy()
        
        # Set in tools_logic module
        import tools_logic as tl
        tl.train_data = train_data
        tl.test_data = test_data
        tl.ground_truth = ground_truth
        
        return f"✅ Loaded {dataset_name}: Train={len(train_data) if train_data is not None else 0}, Test={len(test_data) if test_data is not None else 0}"
    except Exception as e:
        return f"Error: {str(e)}"
