"""Enhanced data loading for RUL prediction with HuggingFace integration."""
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from datasets import load_dataset, DatasetDict
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=False)

# Data directories
DATA_DIR = Path(__file__).parent / "downloaded_datasets"
DATA_DIR.mkdir(exist_ok=True)

# Project data directory (where CMAPSSData is stored)
PROJECT_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"
PROJECT_DATA_DIR.mkdir(exist_ok=True)

# Available HuggingFace datasets
AVAILABLE_DATASETS = [
    "submission096/XJTU",
    "submission096/MAFAULDA",
    "submission096/Padeborn",
    "submission096/IMS",
    "submission096/UoC",
    "submission096/RotorBrokenBar",
    "submission096/MFPT",
    "submission096/HUST",
    "submission096/FEMTO",
    "submission096/Mendeley",
    "submission096/ElectricMotorVibrations",
    "submission096/CWRU",
    "submission096/Azure",
    "submission096/PlanetaryPdM",
]

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

def download_and_save_dataset(hf_dataset_path: str, force_download: bool = False) -> Dict:
    """Download a HuggingFace dataset and save it locally."""
    try:
        dataset_name = hf_dataset_path.split("/")[-1]
        dataset_dir = DATA_DIR / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists() and not force_download:
            return {"status": "already_exists", "dataset_name": dataset_name}
        
        # Set HuggingFace token from environment
        hf_token = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
        
        # Load dataset from HuggingFace
        if hf_token:
            dataset = load_dataset(hf_dataset_path, token=hf_token)
        else:
            dataset = load_dataset(hf_dataset_path)
        
        # Save dataset
        dataset_path = dataset_dir / "dataset"
        if isinstance(dataset, DatasetDict):
            dataset.save_to_disk(str(dataset_path))
            splits = list(dataset.keys())
        else:
            # Single split dataset
            dataset.save_to_disk(str(dataset_path))
            splits = ["train"]
        
        # Convert to pandas and save CSVs
        total_rows = 0
        if isinstance(dataset, DatasetDict):
            for split_name in splits:
                split_data = dataset[split_name]
                df = split_data.to_pandas()
                total_rows += len(df)
                split_dir = dataset_dir / split_name
                split_dir.mkdir(exist_ok=True)
                df.to_csv(split_dir / f"{split_name}.csv", index=False)
                
                # Save to project data directory as well
                project_split_dir = PROJECT_DATA_DIR / dataset_name / split_name
                project_split_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(project_split_dir / f"{split_name}.csv", index=False)
        else:
            df = dataset.to_pandas()
            total_rows = len(df)
            df.to_csv(dataset_dir / "data.csv", index=False)
            splits = ["train"]
            
            # Save to project data directory as well
            project_data_dir = PROJECT_DATA_DIR / dataset_name
            project_data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(project_data_dir / "data.csv", index=False)
        
        # Calculate total size
        try:
            total_size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        except:
            total_size = 0
        
        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "hf_path": hf_dataset_path,
            "splits": splits,
            "num_rows": total_rows,
            "total_size_mb": round(total_size, 2),
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {"status": "success", "dataset_name": dataset_name, "metadata": metadata}
        
    except Exception as e:
        return {"status": "error", "error": str(e), "dataset_path": hf_dataset_path}


def list_available_datasets() -> List[str]:
    """List all available datasets (both local and HuggingFace)."""
    datasets = []
    
    # Check local downloaded datasets
    if DATA_DIR.exists():
        for item in DATA_DIR.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                datasets.append(item.name)
    
    return sorted(datasets)


def list_huggingface_datasets() -> List[str]:
    """List all available HuggingFace dataset paths."""
    return AVAILABLE_DATASETS


def download_all_datasets(force_download: bool = False) -> Dict:
    """Download all available HuggingFace datasets."""
    results = {}
    for hf_path in AVAILABLE_DATASETS:
        dataset_name = hf_path.split("/")[-1]
        results[dataset_name] = download_and_save_dataset(hf_path, force_download)
    return results

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
