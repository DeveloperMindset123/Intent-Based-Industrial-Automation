"""
Ground Truth Verification for RUL Predictions.
Compares predictions with RUL_FD001.txt ground truth.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# Ground truth file path - try multiple possible locations
def find_ground_truth_file():
    """Find the ground truth file in various possible locations."""
    current_file = Path(__file__)
    # Try various relative paths from the current file location
    possible_paths = [
        # From intent_implementation_demo directory
        current_file.parent.parent.parent.parent / "data" / "CMAPSSData" / "RUL_FD001.txt",
        # From project root
        current_file.parent.parent.parent.parent.parent / "data" / "CMAPSSData" / "RUL_FD001.txt",
        # Absolute path (if data is in project root)
        Path("/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData/RUL_FD001.txt"),
        # Relative from current
        current_file.parent / "data" / "CMAPSSData" / "RUL_FD001.txt",
        current_file.parent.parent / "data" / "CMAPSSData" / "RUL_FD001.txt",
        # From shared/data
        current_file.parent.parent / "shared" / "downloaded_datasets" / "CMAPSSData" / "RUL_FD001.txt",
        # From data directory in project
        Path(__file__).parent.parent.parent.parent.parent / "data" / "CMAPSSData" / "RUL_FD001.txt",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None

GROUND_TRUTH_FILE = find_ground_truth_file()


def load_ground_truth() -> List[int]:
    """Load ground truth RUL values from RUL_FD001.txt."""
    gt_file = find_ground_truth_file()
    if gt_file is None:
        return []
    
    try:
        with open(gt_file, 'r') as f:
            rul_values = [int(line.strip()) for line in f if line.strip()]
        return rul_values
    except Exception as e:
        print(f"Error loading ground truth from {gt_file}: {e}")
        return []


def verify_rul_predictions(
    predictions: Dict[int, int],
    ground_truth: Optional[List[int]] = None
) -> Dict[str, any]:
    """
    Verify RUL predictions against ground truth.
    
    Args:
        predictions: Dict mapping engine_id to predicted RUL
        ground_truth: Optional ground truth list (loads from file if not provided)
    
    Returns:
        Dictionary with verification results
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()
    
    if not ground_truth:
        return {
            "status": "error",
            "message": "Ground truth file not found",
            "predictions": predictions
        }
    
    if not predictions:
        return {
            "status": "error",
            "message": "No predictions provided",
            "ground_truth_available": len(ground_truth)
        }
    
    # Match predictions with ground truth
    # Engine IDs are 1-indexed in ground truth
    results = []
    errors = []
    matched_count = 0
    
    for engine_id, predicted_rul in predictions.items():
        # Ground truth is 0-indexed (engine 1 = index 0)
        gt_index = engine_id - 1
        
        if 0 <= gt_index < len(ground_truth):
            actual_rul = ground_truth[gt_index]
            error = abs(predicted_rul - actual_rul)
            relative_error = (error / actual_rul * 100) if actual_rul > 0 else 0
            
            results.append({
                "engine_id": engine_id,
                "predicted_rul": predicted_rul,
                "actual_rul": actual_rul,
                "error": error,
                "relative_error_pct": round(relative_error, 2),
                "match": error <= 5  # Consider match if within 5 cycles
            })
            errors.append(error)
            matched_count += 1
        else:
            results.append({
                "engine_id": engine_id,
                "predicted_rul": predicted_rul,
                "actual_rul": None,
                "error": None,
                "relative_error_pct": None,
                "match": None,
                "note": "No ground truth available for this engine"
            })
    
    # Calculate metrics
    if errors:
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        max_error = max(errors)
        min_error = min(errors)
        accuracy = (matched_count - sum(1 for r in results if r.get("error", 0) > 5)) / matched_count * 100
    else:
        mae = rmse = max_error = min_error = accuracy = None
    
    return {
        "status": "success",
        "total_predictions": len(predictions),
        "matched_predictions": matched_count,
        "accuracy_pct": round(accuracy, 2) if accuracy is not None else None,
        "mae": round(mae, 2) if mae is not None else None,
        "rmse": round(rmse, 2) if rmse is not None else None,
        "max_error": max_error,
        "min_error": min_error,
        "results": results,
        "summary": {
            "correct_predictions": sum(1 for r in results if r.get("match") is True),
            "incorrect_predictions": sum(1 for r in results if r.get("match") is False),
            "no_ground_truth": sum(1 for r in results if r.get("match") is None)
        }
    }


class VerifyRULPredictionsTool(BaseTool):
    """Tool to verify RUL predictions against ground truth."""
    
    name: str = "verify_rul_predictions"
    description: str = """Verify RUL predictions against ground truth data.
    
    Input: JSON with predictions key containing dict mapping engine_id to predicted_rul
    Each prediction maps an engine ID to its predicted remaining useful life in cycles
    
    Returns: Verification results with accuracy metrics.
    """
    
    class VerifyInput(BaseModel):
        predictions: Dict[str, int] = Field(
            description="Dictionary mapping engine_id (as string) to predicted RUL value"
        )
    
    args_schema: type[BaseModel] = VerifyInput
    
    def _run(self, predictions) -> str:
        """Verify predictions against ground truth."""
        import json
        
        # Handle string input (ReactAgent may truncate JSON)
        if isinstance(predictions, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(predictions)
                if isinstance(parsed, dict) and "predictions" in parsed:
                    predictions = parsed["predictions"]
                else:
                    predictions = parsed
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract from truncated string
                # Look for patterns like {"1": 100, "2": 200, ...
                import re
                match = re.search(r'\{[^}]*"(\d+)"\s*:\s*(\d+)', predictions)
                if match:
                    return f"❌ JSON input was truncated. Please use execute_code_simple to verify predictions programmatically. Example: execute_code_simple with code that calls verify_rul_predictions(predictions_dict)."
                return f"❌ Invalid JSON input: {predictions[:200]}"
        
        # Handle dict input
        if not isinstance(predictions, dict):
            return f"❌ Expected dict, got {type(predictions).__name__}: {str(predictions)[:200]}"
        
        # Convert string keys to int
        try:
        predictions_int = {int(k): int(v) for k, v in predictions.items()}
        except (ValueError, TypeError) as e:
            return f"❌ Error converting predictions: {str(e)}. Input: {str(predictions)[:200]}"
        
        result = verify_rul_predictions(predictions_int)
        
        if result["status"] == "error":
            return f"❌ {result['message']}"
        
        # Format as readable string
        output = f"""✅ RUL Prediction Verification Results:

📊 Summary:
   Total Predictions: {result['total_predictions']}
   Matched with Ground Truth: {result['matched_predictions']}
   Accuracy: {result['accuracy_pct']}%
   Mean Absolute Error (MAE): {result['mae']} cycles
   Root Mean Squared Error (RMSE): {result['rmse']} cycles
   Max Error: {result['max_error']} cycles
   Min Error: {result['min_error']} cycles

📈 Detailed Results:
"""
        for r in result['results'][:20]:  # Show first 20
            if r.get('actual_rul') is not None:
                match_icon = "✅" if r.get('match') else "❌"
                output += f"   {match_icon} Engine {r['engine_id']}: Predicted={r['predicted_rul']}, Actual={r['actual_rul']}, Error={r['error']} cycles ({r['relative_error_pct']}%)\n"
            else:
                output += f"   ⚠️  Engine {r['engine_id']}: Predicted={r['predicted_rul']}, No ground truth available\n"
        
        if len(result['results']) > 20:
            output += f"   ... and {len(result['results']) - 20} more engines\n"
        
        return output


def create_ground_truth_tools() -> List[BaseTool]:
    """Create ground truth verification tools."""
    return [VerifyRULPredictionsTool()]

