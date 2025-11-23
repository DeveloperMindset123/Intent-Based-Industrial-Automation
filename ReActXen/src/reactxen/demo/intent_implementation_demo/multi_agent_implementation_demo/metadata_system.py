"""
Metadata System - Dynamic JSON structure for data tracking.
Tracks dataset information, ML model hyperparameters, and data characteristics.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class MetadataSystem:
    """Manages metadata about datasets, models, and data characteristics."""
    
    def __init__(self, metadata_file: Optional[Path] = None):
        """Initialize metadata system."""
        if metadata_file is None:
            metadata_file = Path(__file__).parent / "outputs" / "data_metadata.json"
        self.metadata_file = metadata_file
        self.metadata_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return self._default_metadata()
        return self._default_metadata()
    
    def _default_metadata(self) -> Dict[str, Any]:
        """Return default metadata structure."""
        return {
            "datasets": {},
            "models": {},
            "hyperparameters": {},
            "data_characteristics": {},
            "experiments": [],
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        self.metadata["metadata"]["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def register_dataset(
        self,
        dataset_name: str,
        dataset_path: Optional[str] = None,
        description: Optional[str] = None,
        characteristics: Optional[Dict[str, Any]] = None
    ):
        """Register a dataset with metadata."""
        if dataset_name not in self.metadata["datasets"]:
            self.metadata["datasets"][dataset_name] = {
                "name": dataset_name,
                "path": dataset_path,
                "description": description,
                "characteristics": characteristics or {},
                "registered_at": datetime.now().isoformat(),
                "usage_count": 0,
                "last_used": None
            }
        else:
            # Update existing
            self.metadata["datasets"][dataset_name]["last_used"] = datetime.now().isoformat()
            self.metadata["datasets"][dataset_name]["usage_count"] += 1
            if characteristics:
                self.metadata["datasets"][dataset_name]["characteristics"].update(characteristics)
        
        self._save_metadata()
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset."""
        return self.metadata["datasets"].get(dataset_name)
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self.metadata["datasets"].keys())
    
    def register_model_experiment(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        performance: Optional[Dict[str, Any]] = None,
        dataset_used: Optional[str] = None
    ):
        """Register a model experiment with hyperparameters."""
        experiment = {
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "performance": performance or {},
            "dataset_used": dataset_used,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metadata["experiments"].append(experiment)
        
        # Track best hyperparameters per model
        if model_name not in self.metadata["hyperparameters"]:
            self.metadata["hyperparameters"][model_name] = {
                "best": hyperparameters,
                "all": [hyperparameters]
            }
        else:
            self.metadata["hyperparameters"][model_name]["all"].append(hyperparameters)
            
            # Update best if performance is better
            if performance:
                current_best_perf = self.metadata["hyperparameters"][model_name].get("best_performance", {})
                if self._is_better_performance(performance, current_best_perf):
                    self.metadata["hyperparameters"][model_name]["best"] = hyperparameters
                    self.metadata["hyperparameters"][model_name]["best_performance"] = performance
        
        self._save_metadata()
    
    def _is_better_performance(self, new: Dict[str, Any], old: Dict[str, Any]) -> bool:
        """Check if new performance is better than old."""
        # Compare by accuracy if available
        if "accuracy" in new and "accuracy" in old:
            return new["accuracy"] > old["accuracy"]
        # Compare by lower error if available
        if "mae" in new and "mae" in old:
            return new["mae"] < old["mae"]
        if "rmse" in new and "rmse" in old:
            return new["rmse"] < old["rmse"]
        return False
    
    def get_best_hyperparameters(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters for a model."""
        if model_name in self.metadata["hyperparameters"]:
            return self.metadata["hyperparameters"][model_name].get("best")
        return None
    
    def update_data_characteristics(
        self,
        dataset_name: str,
        characteristics: Dict[str, Any]
    ):
        """Update data characteristics for a dataset."""
        if dataset_name not in self.metadata["data_characteristics"]:
            self.metadata["data_characteristics"][dataset_name] = {}
        
        self.metadata["data_characteristics"][dataset_name].update(characteristics)
        self.metadata["data_characteristics"][dataset_name]["last_updated"] = datetime.now().isoformat()
        
        self._save_metadata()
    
    def get_data_summary(self) -> str:
        """Get a summary of all data metadata."""
        lines = []
        lines.append("DATA METADATA SUMMARY:")
        lines.append(f"- Registered Datasets: {len(self.metadata['datasets'])}")
        lines.append(f"- Model Experiments: {len(self.metadata['experiments'])}")
        lines.append("")
        
        if self.metadata["datasets"]:
            lines.append("Datasets:")
            for name, info in list(self.metadata["datasets"].items())[:5]:
                lines.append(f"  - {name}: {info.get('description', 'No description')}")
                if info.get("characteristics"):
                    chars = info["characteristics"]
                    if "samples" in chars:
                        lines.append(f"    Samples: {chars['samples']}")
                    if "features" in chars:
                        lines.append(f"    Features: {chars['features']}")
            lines.append("")
        
        if self.metadata["hyperparameters"]:
            lines.append("Best Hyperparameters:")
            for model_name, hyperparams in list(self.metadata["hyperparameters"].items())[:3]:
                best = hyperparams.get("best", {})
                lines.append(f"  - {model_name}: {json.dumps(best, indent=4)}")
        
        return "\n".join(lines)


# Global metadata instance
_global_metadata: Optional[MetadataSystem] = None


def get_metadata_system() -> MetadataSystem:
    """Get or create global metadata system instance."""
    global _global_metadata
    if _global_metadata is None:
        _global_metadata = MetadataSystem()
    return _global_metadata

