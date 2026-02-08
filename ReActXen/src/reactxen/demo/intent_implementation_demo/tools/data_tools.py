"""Data loading tools for datasets and ground truth."""

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import os
import re
from pathlib import Path

DATA_DIR = Path(
    os.environ.get(
        "PHMFORGE_DATA_DIR",
        str(
            Path(__file__).parent.parent
            / "multi_agent_implementation_demo"
            / "PDMBench_Data_Directory"
            / "submission096"
        ),
    )
)


def _extract_fd_number(dataset_name: str) -> str:
    """Extract FD number from CMAPSS dataset name.

    Examples:
    - 'CMAPSS_FD001' -> 'FD001'
    - 'CMAPSS FD001' -> 'FD001'
    - 'FD001' -> 'FD001'
    """
    normalized = dataset_name.replace(" ", "_").upper()
    match = re.search(r"FD(\d+)", normalized)
    if match:
        return f"FD{match.group(1)}"
    if normalized.startswith("FD") and normalized[2:].isdigit():
        return normalized
    return normalized.split("_")[-1] if "_" in normalized else normalized


def _normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset name: convert spaces to underscores, handle CMAPSS."""
    dataset_name = dataset_name.strip().strip("'\"")
    normalized = dataset_name.replace(" ", "_").upper()
    if "CMAPSS" in normalized:
        fd_num = _extract_fd_number(normalized)
        return f"CMAPSS_{fd_num}"
    return normalized


def _strip_quotes(value: str) -> str:
    """Strip single and double quotes from string."""
    if not value:
        return value
    return value.strip().strip("'\"")


class LoadDatasetInput(BaseModel):
    dataset_name: str = Field(
        description="Dataset name. For CMAPSS: 'CMAPSS_FD001' or 'CMAPSS FD001' (files are train_FD001.txt, test_FD001.txt). For others: 'CWRU', 'FEMTO', etc."
    )
    split: str = Field(
        default="train",
        description="Data split: 'train', 'val', or 'test'. For CMAPSS, only 'train' and 'test' are available.",
    )


class LoadDatasetTool(BaseTool):
    name: str = "load_dataset"
    description: str = """Load dataset from PDMBench data directory.

    CORRECT TOOL INVOCATION FORMAT:
    Action: load_dataset
    Action Input: {"dataset_name": "CMAPSS_FD001", "split": "test"}

    IMPORTANT:
    - Action must be just the tool name: "load_dataset" (NOT "load_dataset[CMAPSS_FD001, test]")
    - Action Input must be a JSON object: {"dataset_name": "...", "split": "..."}

    For CMAPSS datasets, the files are named:
    - train_FD001.txt (not CMAPSS_FD001_train.csv)
    - test_FD001.txt (not CMAPSS_FD001_test.csv)

    Parameters:
    - dataset_name: 'CMAPSS_FD001' or 'CMAPSS FD001' (both work, spaces converted to underscores)
    - split: 'train' or 'test' (CMAPSS only has train/test, no val split)

    Examples:
    - Load CMAPSS test data: Action="load_dataset", Action Input={"dataset_name": "CMAPSS_FD001", "split": "test"}
    - Load CWRU train data: Action="load_dataset", Action Input={"dataset_name": "CWRU", "split": "train"}
    """
    args_schema: type = LoadDatasetInput

    def _run(self, dataset_name: str, split: str = "train") -> str:
        """Load dataset with proper split handling."""
        dataset_name = _strip_quotes(dataset_name)
        split = _strip_quotes(split)
        normalized_name = _normalize_dataset_name(dataset_name)

        # Handle CMAPSS special case (uses .txt files with format: train_FD001.txt)
        if "CMAPSS" in normalized_name:
            fd_num = _extract_fd_number(normalized_name)
            if split == "train":
                file_path = DATA_DIR / f"train_{fd_num}.txt"
            elif split == "test":
                file_path = DATA_DIR / f"test_{fd_num}.txt"
            else:
                return f"Error: CMAPSS only has 'train' and 'test' splits, not '{split}'. Available splits: train, test"

            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep=" ", header=None, engine="python")
                    return f"Loaded CMAPSS {fd_num} {split}: {len(df)} rows, {len(df.columns)} columns"
                except Exception as e:
                    return f"Error loading CMAPSS file {file_path}: {e}"

            available_files = list(DATA_DIR.glob(f"*{fd_num}*.txt"))
            return f"Error: CMAPSS {fd_num} {split} file not found at {file_path}. Available CMAPSS files: {[f.name for f in available_files]}"

        # Handle other datasets (CSV format)
        split_file = DATA_DIR / f"{normalized_name}_{split}_split.csv"
        orig_file = DATA_DIR / f"{normalized_name}_train.csv"
        file_path = split_file if split_file.exists() else orig_file

        if file_path.exists():
            df = pd.read_csv(file_path)
            return f"Loaded {normalized_name} {split}: {len(df)} rows, {len(df.columns)} columns"

        available = list(DATA_DIR.glob(f"{normalized_name}*"))
        return f"Error: Dataset {normalized_name} split {split} not found. Available files: {[f.name for f in available[:10]]}"


class LoadGroundTruthInput(BaseModel):
    dataset_name: str = Field(
        description="Dataset name. For CMAPSS: 'CMAPSS_FD001' (file will be RUL_FD001.txt automatically)."
    )
    file: str = Field(
        default="",
        description="Ground truth file name. For CMAPSS_FD001, use 'RUL_FD001.txt'. If not provided, will auto-detect from dataset name.",
    )


class LoadGroundTruthTool(BaseTool):
    name: str = "load_ground_truth"
    description: str = """Load ground truth RUL values from files.

    CORRECT TOOL INVOCATION FORMAT:
    Action: load_ground_truth
    Action Input: {"dataset_name": "CMAPSS_FD001", "file": "RUL_FD001.txt"}

    IMPORTANT:
    - Action must be just the tool name: "load_ground_truth" (NOT "load_ground_truth[CMAPSS_FD001, RUL_FD001.txt]")
    - Action Input must be a JSON object: {"dataset_name": "...", "file": "..."}
    - For CMAPSS, the 'file' parameter is optional - tool will auto-detect RUL_FD001.txt from dataset name

    For CMAPSS_FD001: The file is RUL_FD001.txt (located in submission096 directory).

    Parameters:
    - dataset_name: 'CMAPSS_FD001' or 'CMAPSS FD001'
    - file: 'RUL_FD001.txt' (optional for CMAPSS - will auto-detect if not provided)

    Examples:
    - Load CMAPSS ground truth (auto-detect): Action="load_ground_truth", Action Input={"dataset_name": "CMAPSS_FD001", "file": "RUL_FD001.txt"}
    - Load CMAPSS ground truth (explicit): Action="load_ground_truth", Action Input={"dataset_name": "CMAPSS_FD001", "file": "RUL_FD001.txt"}
    """
    args_schema: type = LoadGroundTruthInput

    def _run(self, dataset_name: str, file: str = None) -> str:
        dataset_name = _strip_quotes(dataset_name)
        if file:
            file = _strip_quotes(file)

        normalized_name = _normalize_dataset_name(dataset_name)

        # For CMAPSS, auto-detect file name if not provided
        if "CMAPSS" in normalized_name:
            fd_num = _extract_fd_number(normalized_name)
            if not file or not (DATA_DIR / file).exists():
                file = f"RUL_{fd_num}.txt"

            gt_path = DATA_DIR / file
            if gt_path.exists():
                with open(gt_path, "r") as f:
                    values = [
                        int(line.strip()) for line in f.readlines() if line.strip()
                    ]
                return f"Loaded {len(values)} ground truth RUL values from {file}"

            return f"Error: Ground truth file {file} not found at {gt_path}. Available RUL files: {[f.name for f in DATA_DIR.glob('RUL*.txt')]}"

        # For other datasets, use provided file
        if not file:
            return "Error: file parameter required for non-CMAPSS datasets"

        gt_path = DATA_DIR / file
        if gt_path.exists():
            with open(gt_path, "r") as f:
                values = [int(line.strip()) for line in f.readlines() if line.strip()]
            return f"Loaded {len(values)} ground truth RUL values from {file}"

        return f"Error: Ground truth file {file} not found at {gt_path}"
