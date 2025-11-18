"""
Dataset Tools - Tools for loading and managing datasets.
Created to avoid import issues with agent_implementation_hf.py
"""
from typing import Optional, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from load_data import (
    list_available_datasets,
    load_saved_dataset,
    get_dataset_info,
    load_dataset_for_analysis as load_dataset_for_analysis_impl,
    DATA_DIR,
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


def create_dataset_tools() -> List[BaseTool]:
    """Create and return dataset-related tools."""
    return [
        ListDatasetsTool(),
        LoadDatasetTool(),
        GetDatasetInfoTool(),
    ]

