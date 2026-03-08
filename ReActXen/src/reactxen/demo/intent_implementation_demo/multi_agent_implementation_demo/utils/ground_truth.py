"""Ground truth loading and validation utility."""
from pathlib import Path
import pandas as pd
import json

def load_rul_ground_truth(dataset_name: str, data_dir: Path = None) -> list:
    """Load RUL ground truth values from file."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "PDMBench_Data_Directory" / "submission096"
    
    # Check for RUL file in submission096
    rul_file = data_dir / f"RUL_{dataset_name}.txt"
    
    # For CMAPSS, also check for RUL_FD001.txt format
    if not rul_file.exists() and "CMAPSS" in dataset_name:
        fd_num = dataset_name.split("_")[-1] if "_" in dataset_name else dataset_name.replace("CMAPSS", "")
        # Try RUL_FD001.txt format
        rul_file = data_dir / f"RUL_{fd_num}.txt"
        # If still not found, try alternative path
        if not rul_file.exists():
            cmapss_dir = data_dir.parent.parent.parent.parent / "data" / "CMAPSSData"
            if cmapss_dir.exists():
                rul_file = cmapss_dir / f"RUL_{fd_num}.txt"
    
    if rul_file.exists():
        with open(rul_file, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    return []

def load_fault_labels(dataset_name: str, data_dir: Path = None) -> list:
    """Load fault classification labels from CSV."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "PDMBench_Data_Directory" / "submission096"
    
    csv_path = data_dir / f"{dataset_name}_train.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "label" in df.columns:
            return df["label"].tolist()
    return []

def validate_predictions(predictions: list, ground_truth: list, threshold: float = 15.0) -> dict:
    """Validate predictions against ground truth."""
    if len(predictions) != len(ground_truth):
        return {"error": "Mismatched lengths", "pred_len": len(predictions), "gt_len": len(ground_truth)}
    
    matches = sum(1 for p, g in zip(predictions, ground_truth) if abs(p - g) <= threshold)
    return {
        "total": len(predictions),
        "matches": matches,
        "accuracy": matches / len(predictions) if predictions else 0.0,
        "threshold": threshold
    }