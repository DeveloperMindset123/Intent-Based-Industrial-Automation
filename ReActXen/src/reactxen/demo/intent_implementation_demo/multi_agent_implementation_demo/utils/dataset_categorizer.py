"""Dataset categorization utility for RUL vs Fault Classification tasks. This has been defined in hardcoded fashion."""

import os
import pandas as pd
from pathlib import Path

# RUL datasets from PDMBench/PHM-Bench papers
RUL_DATASETS = [
    "CMAPSS_FD001",
    "FEMTO",
    "IMS",
    "HUST",
    "XJTU",
    "MFPT",
    "Mendeley",
    "Azure",
]

# Fault classification datasets
FAULT_DATASETS = [
    "CWRU",
    "Paderborn",
    "Padeborn",
    "ElectricMotorVibrations",
    "RotorBrokenBar",
    "PlanetaryPdM",
    "GearboxUoC",
    "UoC",
]

# datasets responsible for both RUL and Fault Classification
BOTH_DATASETS = ["HUST", "MFPT"]


def categorize_dataset(dataset_name: str, data_dir: str = None) -> str:
    """Categorize dataset as RUL or Fault based on papers, with structure fallback."""
    # Normalize dataset name
    name_upper = dataset_name.upper()

    # Check paper-based categorization first
    if any(ds in name_upper for ds in RUL_DATASETS):
        if any(ds in name_upper for ds in BOTH_DATASETS):
            return "both"  # Can do both, scenario determines
        return "rul"

    if any(ds in name_upper for ds in FAULT_DATASETS):
        return "fault"

    # Fallback: check structure
    if data_dir:
        data_path = Path(data_dir)
    else:
        data_path = (
            Path(__file__).parent.parent / "PDMBench_Data_Directory" / "submission096"
        )

    file_path = data_path / f"{dataset_name}_train.csv"
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, nrows=10)
            if "label" in df.columns:
                return "fault"
        except:
            pass
        # Check for RUL ground truth file
        rul_file = data_path / f"RUL_{dataset_name}.txt"
        if not rul_file.exists() and "CMAPSS" in name_upper:
            # Try RUL_FD001.txt format
            fd_num = (
                dataset_name.split("_")[-1]
                if "_" in dataset_name
                else dataset_name.replace("CMAPSS", "")
            )
            rul_file = data_path / f"RUL_{fd_num}.txt"
        if rul_file.exists() or "RUL" in str(data_path):
            return "rul"

    return "unknown"


def get_task_type(dataset_name: str, classification_type: str = None) -> str:
    """Get task type for a dataset given scenario classification."""
    if classification_type:
        if "RUL" in classification_type:
            return "rul"
        if "Fault" in classification_type:
            return "fault"

    return categorize_dataset(dataset_name)
