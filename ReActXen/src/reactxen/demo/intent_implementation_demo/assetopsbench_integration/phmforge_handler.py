"""PHMForge scenario handler for AssetOpsBench grading engine.

Loads 75 predictive maintenance scenarios from PHMForge and grades
agent submissions using LLM-based evaluation with ground truth
comparison for metric ranges (MAE, RMSE, accuracy).
"""

import json
import logging
from pathlib import Path

from scenario_server.entities import (
    Scenario,
    ScenarioType,
    ScenarioAnswer,
    ScenarioGrade,
    SubmissionResult,
    SubmissionSummary,
)
from scenario_server.grading import evaluation_agent
from scenario_server.handlers.scenario_handler import ScenarioHandler

logger = logging.getLogger(__name__)

# PHMForge scenarios are loaded from a local JSON file
# (can be replaced with HuggingFace download if published)
_PHMFORGE_SCENARIOS = (
    Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent
    / "ReActXen"
    / "src"
    / "reactxen"
    / "demo"
    / "intent_implementation_demo"
    / "scenarios"
    / "phm_scenarios.json"
)

# Category to expected output characteristics
_CHARACTERISTIC_FORMS = {
    "RUL Prediction": (
        "JSON report with: all_unit_predictions (array of unit_id, predicted_rul, actual_rul, error), "
        "top_N_risk_units (array of 10 unit IDs), overall_mae (float in cycles), "
        "overall_rmse (float in cycles), ground_truth_verification_status (bool)"
    ),
    "Fault Classification": (
        "JSON report with: classification_results (array of unit_id, predicted_class, actual_class), "
        "overall_accuracy (float 0-1), confusion_matrix (optional), "
        "verification_status (bool)"
    ),
    "Engine Health Analysis": (
        "JSON report with: signal_analysis (sensor readings and anomalies), "
        "component_health (status per component), diagnosis (fault pattern), "
        "degradation_trend (trend direction and rate)"
    ),
    "Cost-Benefit Analysis": (
        "JSON report with: maintenance_cost (annual preventive), failure_cost (expected annual), "
        "optimal_schedule (threshold in cycles), recommendation (action)"
    ),
    "Safety/Policy Evaluation": (
        "JSON report with: risk_assessment (RPN and level), compliance_status, "
        "safety_recommendations (prioritized list)"
    ),
}


class PHMForgeScenarios(ScenarioHandler):
    id = "c8f2a1d4-7e3b-4f5a-9c6d-2e8f1b3a5c7d"
    title = "Asset Operations Bench - PHMForge"
    description = (
        "75 predictive maintenance scenarios across RUL prediction, fault classification, "
        "engine health analysis, cost-benefit analysis, and safety/policy evaluation. "
        "Uses 22 MCP tools across 2 servers (prognostics + maintenance) and 18 datasets."
    )

    def __init__(self):
        self.scenario_data: dict[str, dict] = {}
        try:
            scenarios_path = _PHMFORGE_SCENARIOS
            if scenarios_path.exists():
                with open(scenarios_path) as f:
                    data = json.load(f)
                for s in data.get("pdm_scenarios", []):
                    self.scenario_data[s["task_id"]] = s
                logger.info("Loaded %d PHMForge scenarios", len(self.scenario_data))
            else:
                logger.warning("PHMForge scenarios not found at %s", scenarios_path)
        except Exception as e:
            logger.exception("Failed to init PHMForgeScenarios: %s", e)

    def scenario_type(self) -> ScenarioType:
        return ScenarioType(id=self.id, title=self.title, description=self.description)

    def fetch_scenarios(self) -> list[Scenario]:
        scenarios = []
        for task_id, s in self.scenario_data.items():
            metadata = {
                "classification_type": s.get("classification_type", ""),
                "dataset": s.get("dataset", ""),
                "required_tools": s.get("required_tools", []),
            }
            scenarios.append(
                Scenario(
                    id=task_id,
                    query=s.get("input_question", ""),
                    metadata=metadata,
                )
            )
        return scenarios

    def _grade_answer(self, entry_id: str, answer: str) -> ScenarioGrade:
        """Grade a single scenario answer."""
        try:
            unwrap = json.loads(answer)
            scenario = self.scenario_data[entry_id]

            characteristic = _CHARACTERISTIC_FORMS.get(
                scenario.get("classification_type", ""),
                "JSON report with relevant analysis results",
            )
            query = scenario.get("input_question", "")
            result_text = unwrap.get("result", "")
            trace = unwrap.get("trace", "")

            # Use LLM-based evaluation
            correct, details = evaluation_agent(
                actual=result_text,
                charactistic=characteristic,
                query=query,
                trace=trace,
            )

            # Enhance with ground truth range checking for metric scenarios
            gt = scenario.get("ground_truth", {})
            if gt and isinstance(result_text, str):
                gt_details = self._check_ground_truth_ranges(result_text, gt)
                if gt_details:
                    details.append({"ground_truth_check": gt_details})

            return ScenarioGrade(
                scenario_id=entry_id,
                correct=correct,
                details=details,
            )
        except Exception as e:
            logger.exception("Failed to grade %s: %s", entry_id, e)
            return ScenarioGrade(
                scenario_id=entry_id,
                correct=False,
                details=[{"error": f"Failed to grade scenario: {entry_id}"}],
            )

    def _check_ground_truth_ranges(self, result_text: str, gt: dict) -> dict | None:
        """Check if reported metrics fall within expected ranges."""
        checks = {}

        # Check MAE range
        mae_range = gt.get("mae_range") or gt.get("expected_mae_range")
        if mae_range and len(mae_range) == 2:
            try:
                # Try to extract MAE value from result
                import re
                mae_match = re.search(r"[Mm][Aa][Ee]\s*[:=]\s*([\d.]+)", result_text)
                if mae_match:
                    mae_val = float(mae_match.group(1))
                    checks["mae"] = {
                        "value": mae_val,
                        "expected_range": mae_range,
                        "in_range": mae_range[0] <= mae_val <= mae_range[1],
                    }
            except (ValueError, AttributeError):
                pass

        # Check RMSE range
        rmse_range = gt.get("rmse_range") or gt.get("expected_rmse_range")
        if rmse_range and len(rmse_range) == 2:
            try:
                import re
                rmse_match = re.search(r"[Rr][Mm][Ss][Ee]\s*[:=]\s*([\d.]+)", result_text)
                if rmse_match:
                    rmse_val = float(rmse_match.group(1))
                    checks["rmse"] = {
                        "value": rmse_val,
                        "expected_range": rmse_range,
                        "in_range": rmse_range[0] <= rmse_val <= rmse_range[1],
                    }
            except (ValueError, AttributeError):
                pass

        return checks if checks else None

    async def grade_responses(
        self, submission: list[ScenarioAnswer]
    ) -> SubmissionResult:
        correct = 0
        grades = []
        by_type: dict[str, dict] = {}

        for entry in submission:
            entry_id = entry.scenario_id
            if entry_id not in self.scenario_data:
                grades.append(
                    ScenarioGrade(
                        scenario_id=entry_id,
                        correct=False,
                        details=[{"error": f"Unknown scenario id: {entry_id}"}],
                    )
                )
                continue

            g = self._grade_answer(entry_id, entry.answer)
            if g.correct:
                correct += 1

            # Track per-type accuracy
            ctype = self.scenario_data[entry_id].get("classification_type", "unknown")
            if ctype not in by_type:
                by_type[ctype] = {"correct": 0, "total": 0}
            by_type[ctype]["total"] += 1
            if g.correct:
                by_type[ctype]["correct"] += 1

            grades.append(g)

        summary = [
            SubmissionSummary(
                name="Overall",
                value=f"{correct}/{len(self.scenario_data)}",
            ),
        ]
        for ctype, counts in sorted(by_type.items()):
            summary.append(
                SubmissionSummary(
                    name=ctype,
                    value=f"{counts['correct']}/{counts['total']}",
                )
            )

        return SubmissionResult(
            scenario_set_id=self.id,
            summary=summary,
            grades=grades,
        )
