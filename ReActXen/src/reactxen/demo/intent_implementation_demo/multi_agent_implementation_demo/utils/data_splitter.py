"""Data splitting utility for 60/20/20 train/val/test splits."""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "PDMBench_Data_Directory" / "submission096"

def split_dataset(dataset_name: str, random_state: int = 42) -> dict:
    """Split dataset into 60/20/20 train/val/test splits."""
    csv_path = DATA_DIR / f"{dataset_name}_train.csv"
    if not csv_path.exists():
        return {"error": f"Dataset {dataset_name} not found"}
    
    df = pd.read_csv(csv_path)
    
    # First split: 60% train, 40% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=random_state
    )
    
    # Second split: 20% val, 20% test from temp
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state
    )
    
    # Save splits
    train_path = DATA_DIR / f"{dataset_name}_train_split.csv"
    val_path = DATA_DIR / f"{dataset_name}_val_split.csv"
    test_path = DATA_DIR / f"{dataset_name}_test_split.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path)
    }
