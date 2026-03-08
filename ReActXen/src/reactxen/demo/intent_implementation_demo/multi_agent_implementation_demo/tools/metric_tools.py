"""Metric calculation tools for RUL and fault classification."""

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import numpy as np
import json


def _parse_values(value_str: str) -> list:
    """Parse JSON array or comma-separated values."""
    return (
        json.loads(value_str)
        if value_str.startswith("[")
        else [float(x) for x in value_str.split(",")]
    )


class MAEInput(BaseModel):
    ground_truth: str = Field(description="Ground truth RUL values")
    predictions: str = Field(description="Predicted RUL values")


class CalculateMAETool(BaseTool):
    name: str = "calculate_mae"
    description: str = """Calculate Mean Absolute Error for RUL predictions.
    
    CORRECT TOOL INVOCATION FORMAT:
    Action: calculate_mae
    Action Input: {"ground_truth": "[112, 98, 69]", "predictions": "[110, 100, 65]"}
    
    IMPORTANT:
    - Action must be just the tool name: "calculate_mae" (NOT "calculate_mae[ground_truth, predictions]")
    - Action Input must be a JSON object: {"ground_truth": "...", "predictions": "..."}
    
    Parameters (both must be JSON arrays or comma-separated strings):
    - ground_truth: Array of actual RUL values (e.g., '[112, 98, 69]' or '112,98,69')
    - predictions: Array of predicted RUL values (e.g., '[110, 100, 65]' or '110,100,65')
    
    Example: Action="calculate_mae", Action Input={"ground_truth": "[112,98,69]", "predictions": "[110,100,65]"}
    """
    args_schema: type = MAEInput

    def _run(self, ground_truth: str, predictions: str) -> str:
        try:
            gt, pred = _parse_values(ground_truth), _parse_values(predictions)
            mae = np.mean(np.abs(np.array(pred) - np.array(gt)))
            return f"MAE: {mae:.2f} cycles"
        except Exception as e:
            return f"Error calculating MAE: {e}"


class RMSEInput(BaseModel):
    ground_truth: str = Field(description="Ground truth RUL values")
    predictions: str = Field(description="Predicted RUL values")


class CalculateRMSETool(BaseTool):
    name: str = "calculate_rmse"
    description: str = "Calculate Root Mean Squared Error for RUL predictions"
    args_schema: type = RMSEInput

    def _run(self, ground_truth: str, predictions: str) -> str:
        try:
            gt, pred = _parse_values(ground_truth), _parse_values(predictions)
            rmse = np.sqrt(np.mean((np.array(pred) - np.array(gt)) ** 2))
            return f"RMSE: {rmse:.2f} cycles"
        except Exception as e:
            return f"Error calculating RMSE: {e}"


class VerifyInput(BaseModel):
    ground_truth: str = Field(description="Ground truth RUL values")
    predictions: str = Field(description="Predicted RUL values")


class VerifyGroundTruthTool(BaseTool):
    name: str = "verify_ground_truth"
    description: str = "Verify predictions against ground truth RUL values"
    args_schema: type = VerifyInput

    def _run(self, ground_truth: str, predictions: str) -> str:
        try:
            gt, pred = _parse_values(ground_truth), _parse_values(predictions)
            matches = sum(1 for p, g in zip(pred, gt) if abs(p - g) <= 15)
            return f"Verification: {matches}/{len(pred)} predictions within 15 cycles"
        except Exception as e:
            return f"Error verifying: {e}"


class AccuracyInput(BaseModel):
    ground_truth: str = Field(description="Ground truth labels")
    predictions: str = Field(description="Predicted labels")


class CalculateAccuracyTool(BaseTool):
    name: str = "calculate_accuracy"
    description: str = "Calculate classification accuracy for fault classification"
    args_schema: type = AccuracyInput

    def _run(self, ground_truth: str, predictions: str) -> str:
        try:
            gt = (
                json.loads(ground_truth)
                if ground_truth.startswith("[")
                else ground_truth.split(",")
            )
            pred = (
                json.loads(predictions)
                if predictions.startswith("[")
                else predictions.split(",")
            )
            accuracy = np.mean(np.array(pred) == np.array(gt))
            return f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)"
        except Exception as e:
            return f"Error calculating accuracy: {e}"


class VerifyClassInput(BaseModel):
    ground_truth: str = Field(description="Ground truth labels")
    predictions: str = Field(description="Predicted labels")


class VerifyClassificationTool(BaseTool):
    name: str = "verify_classification"
    description: str = "Verify fault classifications against ground truth"
    args_schema: type = VerifyClassInput

    def _run(self, ground_truth: str, predictions: str) -> str:
        try:
            gt = (
                json.loads(ground_truth)
                if ground_truth.startswith("[")
                else ground_truth.split(",")
            )
            pred = (
                json.loads(predictions)
                if predictions.startswith("[")
                else predictions.split(",")
            )
            correct = sum(1 for p, g in zip(pred, gt) if p == g)
            return f"Verification: {correct}/{len(pred)} classifications correct"
        except Exception as e:
            return f"Error verifying classification: {e}"
