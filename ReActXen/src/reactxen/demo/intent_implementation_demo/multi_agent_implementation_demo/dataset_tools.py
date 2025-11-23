"""
Dataset Tools - Tools for loading and managing datasets.
Created to avoid import issues with agent_implementation_hf.py
"""
from typing import Optional, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.load_data import (
    list_available_datasets,
    load_saved_dataset,
    get_dataset_info,
    load_dataset_for_analysis as load_dataset_for_analysis_impl,
    download_and_save_dataset,
    list_huggingface_datasets,
    download_all_datasets,
    DATA_DIR,
    AVAILABLE_DATASETS,
)


class EmptyInput(BaseModel):
    """Empty input schema for tools with no parameters."""
    pass


class LoadDatasetInput(BaseModel):
    """Input schema for load_dataset tool."""
    dataset_name: Optional[str] = Field(
        default=None,
        description="Name of the dataset to load (e.g., 'CWRU', 'Azure'). If not provided, uses first available."
    )


def load_dataset_for_analysis(dataset_name: str = "CWRU", split: str = "train") -> str:
    """Load a specific dataset and prepare it for RUL prediction analysis."""
    # Use the existing implementation from load_data.py which properly sets global state
    return load_dataset_for_analysis_impl(dataset_name)


class ListDatasetsTool(BaseTool):
    """List all available downloaded datasets."""
    
    name: str = "list_datasets"
    description: str = """List all available datasets in local storage.
    
    Returns a list of dataset names that can be loaded using load_dataset tool.
    """
    args_schema: type[BaseModel] = EmptyInput
    
    def _run(self) -> str:
        available = list_available_datasets()
        
        if not available:
            return "No datasets found. Please download datasets first using download_and_save_dataset() or download_all_datasets()."
        
        result = f"📦 Available Datasets ({len(available)}):\n\n"
        
        for dataset_name in available:
            info = get_dataset_info(dataset_name)
            if 'error' not in info:
                result += f"  • {dataset_name}\n"
                if 'splits' in info:
                    result += f"    Splits: {', '.join(info['splits'])}\n"
                else:
                    result += f"    Rows: {info.get('num_rows', 'N/A')}\n"
                result += f"    Size: {info.get('total_size_mb', 0)} MB\n"
        
        result += "\n\nUse load_dataset tool with dataset_name parameter to load a specific dataset."
        
        return result


class LoadDatasetTool(BaseTool):
    """Load a HuggingFace dataset from local storage."""
    
    name: str = "load_dataset"
    description: str = """Load a HuggingFace dataset from local storage.
    
    This tool loads a previously downloaded dataset from the downloaded_datasets directory.
    The dataset will be prepared for RUL prediction analysis.
    
    Use list_datasets first to see available datasets.
    """
    args_schema: type[BaseModel] = LoadDatasetInput
    
    def _run(self, dataset_name: Optional[str] = None) -> str:
        # Clean the dataset_name to remove any "Observation" text or JSON artifacts
        if dataset_name:
            import json
            import re
            # Remove "Observation" text if present
            dataset_name = dataset_name.replace("Observation", "").strip()
            # Try to extract JSON if it's wrapped in JSON
            json_match = re.search(r'\{[^}]*"dataset_name"[^}]*\}', dataset_name)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    dataset_name = parsed.get("dataset_name", dataset_name)
                except:
                    pass
            # Clean any remaining quotes or braces
            dataset_name = dataset_name.strip('"').strip("'").strip('{').strip('}').strip()
        
        if dataset_name is None or not dataset_name:
            available = list_available_datasets()
            if available:
                dataset_name = available[0]
                result = load_dataset_for_analysis(dataset_name)
                return f"No dataset name provided. Using first available: {dataset_name}.\n{result}"
            else:
                return "Error: No dataset name provided and no datasets available. Please download datasets first."
        
        return load_dataset_for_analysis(dataset_name)


class GetDatasetInfoTool(BaseTool):
    """Get detailed information about a dataset."""
    
    name: str = "get_dataset_info"
    description: str = """Get detailed information about a specific dataset.
    
    Returns metadata including splits, columns, size, and other information.
    """
    args_schema: type[BaseModel] = LoadDatasetInput
    
    def _run(self, dataset_name: Optional[str] = None) -> str:
        if dataset_name is None:
            return "Error: dataset_name is required. Use list_datasets to see available datasets."
        
        info = get_dataset_info(dataset_name)
        
        if 'error' in info:
            return str(info['error'])
        
        import json
        return json.dumps(info, indent=2)


class DownloadDatasetInput(BaseModel):
    """Input schema for download_dataset tool."""
    hf_dataset_path: str = Field(
        description="HuggingFace dataset path (e.g., 'submission096/CWRU', 'submission096/Azure')"
    )
    force_download: bool = Field(
        default=False,
        description="Force re-download even if dataset already exists"
    )


class DownloadDatasetTool(BaseTool):
    """Download a HuggingFace dataset to local storage."""
    
    name: str = "download_dataset"
    description: str = """Download a HuggingFace dataset to local storage.
    
    Available datasets:
    - submission096/XJTU
    - submission096/MAFAULDA
    - submission096/Padeborn
    - submission096/IMS
    - submission096/UoC
    - submission096/RotorBrokenBar
    - submission096/MFPT
    - submission096/HUST
    - submission096/FEMTO
    - submission096/Mendeley
    - submission096/ElectricMotorVibrations
    - submission096/CWRU
    - submission096/Azure
    - submission096/PlanetaryPdM
    
    Returns: Download status and metadata.
    """
    args_schema: type[BaseModel] = DownloadDatasetInput
    
    def _run(self, hf_dataset_path: str, force_download: bool = False) -> str:
        """Download a HuggingFace dataset."""
        try:
            result = download_and_save_dataset(hf_dataset_path, force_download)
            if result["status"] == "success":
                return f"✅ Downloaded {result['dataset_name']}: {result['metadata']}"
            elif result["status"] == "already_exists":
                return f"ℹ️  Dataset {result['dataset_name']} already exists. Use force_download=True to re-download."
            else:
                return f"❌ Error downloading {hf_dataset_path}: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error: {str(e)}"


class ListHuggingFaceDatasetsTool(BaseTool):
    """List all available HuggingFace dataset paths."""
    
    name: str = "list_huggingface_datasets"
    description: str = """List all available HuggingFace dataset paths.
    
    Returns a list of dataset paths that can be downloaded using download_dataset tool.
    """
    args_schema: type[BaseModel] = EmptyInput
    
    def _run(self) -> str:
        """List available HuggingFace datasets."""
        datasets = list_huggingface_datasets()
        result = f"📦 Available HuggingFace Datasets ({len(datasets)}):\n\n"
        for hf_path in datasets:
            dataset_name = hf_path.split("/")[-1]
            result += f"  • {hf_path} (name: {dataset_name})\n"
        result += "\nUse download_dataset tool with hf_dataset_path parameter to download a dataset."
        return result


def create_dataset_tools() -> List[BaseTool]:
    """Create and return dataset-related tools."""
    return [
        ListDatasetsTool(),
        LoadDatasetTool(),
        GetDatasetInfoTool(),
        DownloadDatasetTool(),
        ListHuggingFaceDatasetsTool(),
    ]

