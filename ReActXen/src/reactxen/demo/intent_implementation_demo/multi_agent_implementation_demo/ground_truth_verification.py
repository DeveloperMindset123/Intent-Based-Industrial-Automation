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


# Ground truth file path
GROUND_TRUTH_FILE = Path(__file__).parent.parent.parent.parent.parent / "data" / "CMAPSSData" / "RUL_FD001.txt"


def load_ground_truth() -> List[int]:
    """Load ground truth RUL values from RUL_FD001.txt."""
    try:
        with open(GROUND_TRUTH_FILE, 'r') as f:
            rul_values = [int(line.strip()) for line in f if line.strip()]
        return rul_values
    except FileNotFoundError:
        # Try alternative path
        alt_path = Path(__file__).parent / "data" / "CMAPSSData" / "RUL_FD001.txt"
        if alt_path.exists():
            with open(alt_path, 'r') as f:
                rul_values = [int(line.strip()) for line in f if line.strip()]
            return rul_values
        return []
    except Exception as e:
        print(f"Error loading ground truth: {e}")
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
    
    Input: JSON with 'predictions' key containing dict of {engine_id: predicted_rul}
    Example: {"predictions": {"1": 18, "2": 15, "3": 12}}
    
    Returns: Verification results with accuracy metrics.
    """
    
    class VerifyInput(BaseModel):
        predictions: Dict[str, int] = Field(
            description="Dictionary mapping engine_id (as string) to predicted RUL value"
        )
    
    args_schema: type[BaseModel] = VerifyInput
    
    def _run(self, predictions: Dict[str, int]) -> str:
        """Verify predictions against ground truth."""
        # Convert string keys to int
        predictions_int = {int(k): int(v) for k, v in predictions.items()}
        
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

