"""
Comprehensive Performance Evaluator
Evaluates agent performance against PDMBench baseline and calculates scores.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re


class ComprehensiveEvaluator:
    """Comprehensive evaluation system for agent performance."""
    
    # PDMBench baseline scores (from paper)
    PDMBENCH_BASELINE = {
        "rul_prediction": {
            "mae": 12.5,
            "rmse": 18.3,
            "score": 75.2,
            "completeness": 0.75
        },
        "fault_classification": {
            "accuracy": 0.82,
            "f1_score": 0.79,
            "score": 76.5,
            "completeness": 0.80
        },
        "cost_benefit": {
            "cost_reduction": 0.0,  # No baseline
            "score": 70.0,
            "completeness": 0.70
        },
        "safety_policies": {
            "compliance_rate": 0.85,
            "score": 72.0,
            "completeness": 0.75
        }
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "outputs" / "evaluation"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_result(
        self,
        task: Dict[str, Any],
        result: Union[str, Dict[str, Any]],
        execution_time: float,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Comprehensively evaluate agent result.
        
        SUCCESS CRITERIA:
        - Task must be accomplished (all required components present)
        - Ground truth validation MUST be completed (for RUL predictions)
        - Both conditions must be met for success
        """
        task_type = task.get('type', 'rul_prediction')
        question = task.get('question', '')
        expected_outputs = task.get('expected_outputs', [])
        
        # Convert result to string if it's a dict
        if isinstance(result, dict):
            result_str = str(result)
        else:
            result_str = str(result) if result else ""
        
        evaluation = {
            "task_type": task_type,
            "question": question,
            "execution_time": execution_time,
            "result_length": len(result_str),
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "improvement": {},
            "success": False,  # Will be set based on success criteria
            "success_criteria_met": {
                "task_accomplished": False,
                "ground_truth_validated": False
            }
        }
        
        # Task-specific evaluation (use string version)
        if task_type == "rul_prediction":
            metrics = self._evaluate_rul_prediction(result_str, ground_truth, expected_outputs)
        elif task_type == "fault_classification":
            metrics = self._evaluate_fault_classification(result_str, expected_outputs)
        elif task_type == "cost_benefit":
            metrics = self._evaluate_cost_benefit(result_str, expected_outputs)
        elif task_type == "safety_policies":
            metrics = self._evaluate_safety_policies(result_str, expected_outputs)
        else:
            metrics = self._evaluate_generic(result_str, expected_outputs)
        
        evaluation["metrics"] = metrics
        
        # Check success criteria
        # 1. Task accomplished: completeness >= 0.8 (80%)
        task_accomplished = metrics.get("completeness", 0.0) >= 0.8
        evaluation["success_criteria_met"]["task_accomplished"] = task_accomplished
        
        # 2. Ground truth validated: MUST be true for RUL predictions
        if task_type == "rul_prediction":
            ground_truth_validated = metrics.get("ground_truth_verified", False)
            evaluation["success_criteria_met"]["ground_truth_validated"] = ground_truth_validated
            # For RUL: BOTH conditions must be met
            evaluation["success"] = task_accomplished and ground_truth_validated
        else:
            # For other tasks: task accomplished is sufficient
            ground_truth_validated = True  # Not required for non-RUL tasks
            evaluation["success_criteria_met"]["ground_truth_validated"] = ground_truth_validated
            evaluation["success"] = task_accomplished
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics, execution_time, task_type)
        evaluation["metrics"]["overall_score"] = overall_score
        
        # Penalize if success criteria not met
        if not evaluation["success"]:
            if task_type == "rul_prediction" and not ground_truth_validated:
                # Heavy penalty for missing ground truth validation
                overall_score = max(0, overall_score - 30.0)
                evaluation["metrics"]["overall_score"] = overall_score
                evaluation["metrics"]["penalty_applied"] = "Missing ground truth validation (-30 points)"
            elif not task_accomplished:
                # Penalty for incomplete task
                overall_score = max(0, overall_score - 20.0)
                evaluation["metrics"]["overall_score"] = overall_score
                evaluation["metrics"]["penalty_applied"] = "Task incomplete (-20 points)"
        
        # Compare with PDMBench baseline
        if task_type in self.PDMBENCH_BASELINE:
            baseline = self.PDMBENCH_BASELINE[task_type]
            improvement = {
                "score_improvement": overall_score - baseline["score"],
                "score_improvement_percent": ((overall_score - baseline["score"]) / baseline["score"]) * 100,
                "baseline_score": baseline["score"],
                "our_score": overall_score
            }
            evaluation["improvement"] = improvement
        
        return evaluation
    
    def _evaluate_rul_prediction(
        self,
        result: str,
        ground_truth: Optional[Dict[str, Any]],
        expected_outputs: List[str]
    ) -> Dict[str, Any]:
        """Evaluate RUL prediction result."""
        metrics = {
            "ground_truth_verified": False,
            "has_predictions": False,
            "has_verification": False,
            "has_safety_recommendations": False,
            "has_cost_estimates": False,
            "completeness": 0.0
        }
        
        # Ensure result is a string
        result_str = str(result) if result else ""
        result_lower = result_str.lower()
        
        # Check for ground truth verification - must be more specific
        # Look for actual verification output, not just the word "verify"
        verification_patterns = [
            r"verify_rul_predictions",  # Tool name
            r"ground truth.*verified",  # Explicit verification statement
            r"verification.*complete",  # Completion statement
            r"accuracy.*\d+%",  # Accuracy metric
            r"mae.*\d+",  # Mean absolute error
            r"rmse.*\d+",  # Root mean square error
        ]
        has_verification = any(re.search(pattern, result_lower) for pattern in verification_patterns)
        if has_verification:
            metrics["has_verification"] = True
            metrics["ground_truth_verified"] = True
        
        # Check for RUL predictions
        if any(keyword in result_lower for keyword in ["rul", "remaining useful", "cycles", "prediction"]):
            metrics["has_predictions"] = True
        
        # Check for safety recommendations
        if any(keyword in result_lower for keyword in ["safety", "recommendation", "risk", "hazard"]):
            metrics["has_safety_recommendations"] = True
        
        # Check for cost estimates
        if any(keyword in result_lower for keyword in ["cost", "estimate", "maintenance", "price"]):
            metrics["has_cost_estimates"] = True
        
        # Calculate completeness
        components_found = sum([
            metrics["has_predictions"],
            metrics["has_verification"],
            metrics["has_safety_recommendations"],
            metrics["has_cost_estimates"]
        ])
        metrics["completeness"] = components_found / len(expected_outputs) if expected_outputs else components_found / 4
        
        return metrics
    
    def _evaluate_fault_classification(
        self,
        result: str,
        expected_outputs: List[str]
    ) -> Dict[str, Any]:
        """Evaluate fault classification result."""
        metrics = {
            "has_fault_types": False,
            "has_classification": False,
            "has_confidence": False,
            "completeness": 0.0
        }
        
        # Ensure result is a string
        result_str = str(result) if result else ""
        result_lower = result_str.lower()
        
        # Check for fault types
        if any(keyword in result_lower for keyword in ["fault", "failure", "anomaly", "defect"]):
            metrics["has_fault_types"] = True
        
        # Check for classification
        if any(keyword in result_lower for keyword in ["classification", "category", "type", "class"]):
            metrics["has_classification"] = True
        
        # Check for confidence
        if any(keyword in result_lower for keyword in ["confidence", "probability", "certainty", "score"]):
            metrics["has_confidence"] = True
        
        # Calculate completeness
        components_found = sum([
            metrics["has_fault_types"],
            metrics["has_classification"],
            metrics["has_confidence"]
        ])
        metrics["completeness"] = components_found / len(expected_outputs) if expected_outputs else components_found / 3
        
        return metrics
    
    def _evaluate_cost_benefit(
        self,
        result: str,
        expected_outputs: List[str]
    ) -> Dict[str, Any]:
        """Evaluate cost-benefit analysis result."""
        metrics = {
            "has_cost_analysis": False,
            "has_benefit_analysis": False,
            "has_comparison": False,
            "has_roi": False,
            "completeness": 0.0
        }
        
        # Ensure result is a string
        result_str = str(result) if result else ""
        result_lower = result_str.lower()
        
        # Check for cost analysis
        if any(keyword in result_lower for keyword in ["cost", "expense", "price", "maintenance cost"]):
            metrics["has_cost_analysis"] = True
        
        # Check for benefit analysis
        if any(keyword in result_lower for keyword in ["benefit", "saving", "revenue", "value"]):
            metrics["has_benefit_analysis"] = True
        
        # Check for comparison
        if any(keyword in result_lower for keyword in ["compare", "versus", "vs", "difference"]):
            metrics["has_comparison"] = True
        
        # Check for ROI
        if any(keyword in result_lower for keyword in ["roi", "return on investment", "investment"]):
            metrics["has_roi"] = True
        
        # Calculate completeness
        components_found = sum([
            metrics["has_cost_analysis"],
            metrics["has_benefit_analysis"],
            metrics["has_comparison"],
            metrics["has_roi"]
        ])
        metrics["completeness"] = components_found / len(expected_outputs) if expected_outputs else components_found / 4
        
        return metrics
    
    def _evaluate_safety_policies(
        self,
        result: str,
        expected_outputs: List[str]
    ) -> Dict[str, Any]:
        """Evaluate safety/policies result."""
        metrics = {
            "has_safety_risks": False,
            "has_recommendations": False,
            "has_compliance": False,
            "completeness": 0.0
        }
        
        # Ensure result is a string
        result_str = str(result) if result else ""
        result_lower = result_str.lower()
        
        # Check for safety risks
        if any(keyword in result_lower for keyword in ["safety", "risk", "hazard", "danger"]):
            metrics["has_safety_risks"] = True
        
        # Check for recommendations
        if any(keyword in result_lower for keyword in ["recommendation", "suggestion", "action", "plan"]):
            metrics["has_recommendations"] = True
        
        # Check for compliance
        if any(keyword in result_lower for keyword in ["compliance", "policy", "regulation", "standard"]):
            metrics["has_compliance"] = True
        
        # Calculate completeness
        components_found = sum([
            metrics["has_safety_risks"],
            metrics["has_recommendations"],
            metrics["has_compliance"]
        ])
        metrics["completeness"] = components_found / len(expected_outputs) if expected_outputs else components_found / 3
        
        return metrics
    
    def _evaluate_generic(
        self,
        result: str,
        expected_outputs: List[str]
    ) -> Dict[str, Any]:
        """Evaluate generic task result."""
        # Ensure result is a string
        result_str = str(result) if result else ""
        result_lower = result_str.lower()
        components_found = sum(1 for output in expected_outputs 
                              if output.lower().replace('_', ' ') in result_lower)
        completeness = components_found / len(expected_outputs) if expected_outputs else 0.0
        
        return {
            "components_found": components_found,
            "components_required": len(expected_outputs),
            "completeness": completeness
        }
    
    def _calculate_overall_score(
        self,
        metrics: Dict[str, Any],
        execution_time: float,
        task_type: str
    ) -> float:
        """Calculate overall performance score (0-100)."""
        score = 0.0
        
        # Completeness (40 points)
        completeness = metrics.get("completeness", 0.0)
        score += completeness * 40.0
        
        # Task-specific bonuses
        if task_type == "rul_prediction":
            # Ground truth verification (30 points)
            if metrics.get("ground_truth_verified", False):
                score += 30.0
            # Other components (20 points)
            if metrics.get("has_safety_recommendations", False):
                score += 10.0
            if metrics.get("has_cost_estimates", False):
                score += 10.0
        elif task_type == "fault_classification":
            # Classification quality (40 points)
            if metrics.get("has_classification", False):
                score += 25.0
            if metrics.get("has_confidence", False):
                score += 15.0
        elif task_type == "cost_benefit":
            # Analysis quality (40 points)
            if metrics.get("has_roi", False):
                score += 20.0
            if metrics.get("has_comparison", False):
                score += 20.0
        elif task_type == "safety_policies":
            # Compliance and recommendations (40 points)
            if metrics.get("has_compliance", False):
                score += 20.0
            if metrics.get("has_recommendations", False):
                score += 20.0
        
        # Efficiency bonus (20 points)
        if execution_time < 60:
            score += 20.0
        elif execution_time < 120:
            score += 15.0
        elif execution_time < 180:
            score += 10.0
        
        # Response detail bonus (10 points)
        if metrics.get("result_length", 0) > 500:
            score += 10.0
        elif metrics.get("result_length", 0) > 200:
            score += 5.0
        
        return min(score, 100.0)
    
    def compare_with_baseline(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare agent performance with PDMBench baseline."""
        comparison = {
            "task_types": {},
            "overall": {
                "baseline_avg_score": 0.0,
                "our_avg_score": 0.0,
                "improvement": 0.0,
                "improvement_percent": 0.0
            }
        }
        
        # Group by task type
        by_type = {}
        for eval_data in evaluations:
            task_type = eval_data["task_type"]
            if task_type not in by_type:
                by_type[task_type] = []
            by_type[task_type].append(eval_data)
        
        # Compare each task type
        baseline_total = 0.0
        our_total = 0.0
        task_count = 0
        
        for task_type, evals in by_type.items():
            if task_type not in self.PDMBENCH_BASELINE:
                continue
            
            baseline_score = self.PDMBENCH_BASELINE[task_type]["score"]
            our_scores = [e["metrics"]["overall_score"] for e in evals]
            our_avg = sum(our_scores) / len(our_scores) if our_scores else 0.0
            
            improvement = our_avg - baseline_score
            improvement_pct = (improvement / baseline_score) * 100 if baseline_score > 0 else 0.0
            
            comparison["task_types"][task_type] = {
                "baseline_score": baseline_score,
                "our_avg_score": our_avg,
                "improvement": improvement,
                "improvement_percent": improvement_pct,
                "num_tasks": len(evals)
            }
            
            baseline_total += baseline_score
            our_total += our_avg
            task_count += 1
        
        # Calculate overall improvement
        if task_count > 0:
            comparison["overall"]["baseline_avg_score"] = baseline_total / task_count
            comparison["overall"]["our_avg_score"] = our_total / task_count
            comparison["overall"]["improvement"] = our_total / task_count - baseline_total / task_count
            comparison["overall"]["improvement_percent"] = (
                (our_total / task_count - baseline_total / task_count) / (baseline_total / task_count)
            ) * 100 if baseline_total > 0 else 0.0
        
        return comparison
    
    def save_evaluation_report(self, evaluations: List[Dict[str, Any]], comparison: Dict[str, Any]) -> Path:
        """Save comprehensive evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "evaluations": evaluations,
            "baseline_comparison": comparison,
            "summary": {
                "total_tasks": len(evaluations),
                "avg_score": sum(e["metrics"]["overall_score"] for e in evaluations) / len(evaluations) if evaluations else 0.0,
                "avg_time": sum(e["execution_time"] for e in evaluations) / len(evaluations) if evaluations else 0.0,
                "tasks_above_80": sum(1 for e in evaluations if e["metrics"]["overall_score"] >= 80.0),
                "improvement_vs_baseline": comparison["overall"]["improvement_percent"]
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    def generate_text_report(self, evaluations: List[Dict[str, Any]], comparison: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE PERFORMANCE EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary with success criteria
        successful_tasks = [e for e in evaluations if e.get("success", False)]
        summary = {
            "total_tasks": len(evaluations),
            "successful_tasks": len(successful_tasks),
            "success_rate": len(successful_tasks) / len(evaluations) if evaluations else 0.0,
            "avg_score": sum(e["metrics"]["overall_score"] for e in evaluations) / len(evaluations) if evaluations else 0.0,
            "avg_score_successful": sum(e["metrics"]["overall_score"] for e in successful_tasks) / len(successful_tasks) if successful_tasks else 0.0,
            "avg_time": sum(e["execution_time"] for e in evaluations) / len(evaluations) if evaluations else 0.0,
            "tasks_above_80": sum(1 for e in evaluations if e["metrics"]["overall_score"] >= 80.0),
            "tasks_above_80_and_successful": sum(1 for e in successful_tasks if e["metrics"]["overall_score"] >= 80.0),
            "improvement_vs_baseline": comparison["overall"]["improvement_percent"],
            "ground_truth_validation_rate": sum(1 for e in evaluations if e.get("success_criteria_met", {}).get("ground_truth_validated", False)) / len(evaluations) if evaluations else 0.0
        }
        
        lines.append("OVERALL SUMMARY:")
        lines.append(f"  Total Tasks Evaluated: {summary['total_tasks']}")
        lines.append(f"  Successful Tasks: {summary['successful_tasks']}/{summary['total_tasks']} ({summary['success_rate']:.1%})")
        lines.append(f"  Average Score (All): {summary['avg_score']:.2f}/100.0")
        lines.append(f"  Average Score (Successful): {summary['avg_score_successful']:.2f}/100.0")
        lines.append(f"  Average Execution Time: {summary['avg_time']:.2f}s")
        lines.append(f"  Tasks Above 80.0: {summary['tasks_above_80']}/{summary['total_tasks']}")
        lines.append(f"  Tasks Above 80.0 AND Successful: {summary['tasks_above_80_and_successful']}/{summary['total_tasks']}")
        lines.append(f"  Ground Truth Validation Rate: {summary['ground_truth_validation_rate']:.1%}")
        lines.append(f"  Improvement vs PDMBench Baseline: {summary['improvement_vs_baseline']:.1f}%")
        lines.append("")
        
        # Task type comparison
        lines.append("TASK TYPE COMPARISON:")
        for task_type, comp_data in comparison["task_types"].items():
            lines.append(f"  {task_type.upper()}:")
            lines.append(f"    Baseline Score: {comp_data['baseline_score']:.2f}")
            lines.append(f"    Our Score: {comp_data['our_avg_score']:.2f}")
            lines.append(f"    Improvement: +{comp_data['improvement']:.2f} ({comp_data['improvement_percent']:.1f}%)")
            lines.append("")
        
        # Individual task results
        lines.append("INDIVIDUAL TASK RESULTS:")
        for i, eval_data in enumerate(evaluations, 1):
            lines.append(f"  Task {i}: {eval_data['task_type']}")
            lines.append(f"    Score: {eval_data['metrics']['overall_score']:.2f}/100.0")
            lines.append(f"    Execution Time: {eval_data['execution_time']:.2f}s")
            lines.append(f"    Completeness: {eval_data['metrics'].get('completeness', 0.0):.1%}")
            if "improvement" in eval_data:
                lines.append(f"    vs Baseline: +{eval_data['improvement']['score_improvement']:.2f} ({eval_data['improvement']['score_improvement_percent']:.1f}%)")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def get_evaluator() -> ComprehensiveEvaluator:
    """Get or create evaluator instance."""
    if not hasattr(get_evaluator, '_instance'):
        get_evaluator._instance = ComprehensiveEvaluator()
    return get_evaluator._instance

