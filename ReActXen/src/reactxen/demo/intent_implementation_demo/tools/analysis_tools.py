"""Analysis tools for Engine Health, Cost-Benefit, and Safety/Policy evaluation."""

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import json


# ---------------------------------------------------------------------------
# Engine Health Analysis Tools
# ---------------------------------------------------------------------------

class AnalyzeEngineSignalsInput(BaseModel):
    sensor_data: str = Field(
        description="JSON string of sensor readings, e.g. '{\"T24\": 518.67, \"T30\": 642.15, \"T50\": 1589.7}'"
    )
    engine_id: str = Field(default="", description="Engine or unit identifier")


class AnalyzeEngineSignalsTool(BaseTool):
    name: str = "analyze_engine_signals"
    description: str = "Parse multi-sensor signal data from scenario text and identify anomalies across sensor channels."
    args_schema: type = AnalyzeEngineSignalsInput

    def _run(self, sensor_data: str, engine_id: str = "") -> str:
        try:
            readings = json.loads(sensor_data) if isinstance(sensor_data, str) else sensor_data
            n_sensors = len(readings)
            anomalies = [k for k, v in readings.items() if isinstance(v, (int, float)) and abs(v) > 1000]
            status = "WARNING" if anomalies else "NORMAL"
            label = f" for engine {engine_id}" if engine_id else ""
            return (
                f"Signal analysis{label}: {n_sensors} sensors parsed, "
                f"{len(anomalies)} anomalies detected ({', '.join(anomalies) if anomalies else 'none'}). "
                f"Overall status: {status}"
            )
        except Exception as e:
            return f"Error analyzing engine signals: {e}"


class AssessComponentHealthInput(BaseModel):
    component: str = Field(description="Component name: Fan, LPC, HPC, HPT, or LPT")
    efficiency: float = Field(description="Current efficiency value (0-1 scale)")
    flow_modifier: float = Field(description="Current flow modifier value (0-1 scale)")


class AssessComponentHealthTool(BaseTool):
    name: str = "assess_component_health"
    description: str = "Evaluate health status of turbofan engine components (Fan/LPC/HPC/HPT/LPT) based on efficiency and flow."
    args_schema: type = AssessComponentHealthInput

    def _run(self, component: str, efficiency: float, flow_modifier: float) -> str:
        valid_components = ["Fan", "LPC", "HPC", "HPT", "LPT"]
        if component not in valid_components:
            return f"Error: component must be one of {valid_components}, got '{component}'"

        if efficiency >= 0.95 and flow_modifier >= 0.95:
            health = "Healthy"
        elif efficiency >= 0.85 and flow_modifier >= 0.85:
            health = "Degraded"
        elif efficiency >= 0.70 and flow_modifier >= 0.70:
            health = "Warning"
        else:
            health = "Critical"

        return (
            f"{component} health assessment: {health} "
            f"(efficiency={efficiency:.3f}, flow_modifier={flow_modifier:.3f})"
        )


class DiagnoseTimingIssuesInput(BaseModel):
    efficiency_deviation: float = Field(description="Efficiency deviation from baseline")
    flow_deviation: float = Field(description="Flow modifier deviation from baseline")


class DiagnoseTimingIssuesTool(BaseTool):
    name: str = "diagnose_timing_issues"
    description: str = "Identify whether degradation is efficiency-driven or flow-modifier-driven fault pattern."
    args_schema: type = DiagnoseTimingIssuesInput

    def _run(self, efficiency_deviation: float, flow_deviation: float) -> str:
        if abs(efficiency_deviation) > abs(flow_deviation) * 2:
            diagnosis = "Efficiency-dominated fault (likely thermal degradation or tip clearance)"
        elif abs(flow_deviation) > abs(efficiency_deviation) * 2:
            diagnosis = "Flow-dominated fault (likely fouling or erosion)"
        else:
            diagnosis = "Combined efficiency-flow fault (likely advanced degradation)"

        severity = "High" if max(abs(efficiency_deviation), abs(flow_deviation)) > 0.05 else "Moderate"
        return (
            f"Timing diagnosis: {diagnosis}. "
            f"Severity: {severity} (eff_dev={efficiency_deviation:.4f}, flow_dev={flow_deviation:.4f})"
        )


class DetectDegradationTrendInput(BaseModel):
    cycle_data: str = Field(
        description="JSON array of cycle measurements, e.g. '[{\"cycle\": 1, \"value\": 0.99}, ...]'"
    )


class DetectDegradationTrendTool(BaseTool):
    name: str = "detect_degradation_trend"
    description: str = "Analyze multi-cycle degradation patterns to detect trend direction and rate."
    args_schema: type = DetectDegradationTrendInput

    def _run(self, cycle_data: str) -> str:
        try:
            data = json.loads(cycle_data) if isinstance(cycle_data, str) else cycle_data
            if len(data) < 2:
                return "Error: need at least 2 data points for trend detection"

            values = [d["value"] for d in data]
            cycles = [d["cycle"] for d in data]
            delta = values[-1] - values[0]
            rate = delta / (cycles[-1] - cycles[0]) if cycles[-1] != cycles[0] else 0

            if delta < -0.01:
                trend = "Degrading"
            elif delta > 0.01:
                trend = "Improving"
            else:
                trend = "Stable"

            return (
                f"Degradation trend: {trend} over {len(data)} cycles. "
                f"Total change: {delta:.4f}, Rate: {rate:.6f}/cycle"
            )
        except Exception as e:
            return f"Error detecting degradation trend: {e}"


# ---------------------------------------------------------------------------
# Cost-Benefit Analysis Tools
# ---------------------------------------------------------------------------

class CalculateMaintenanceCostInput(BaseModel):
    preventive_cost: float = Field(description="Cost of a single preventive maintenance action ($)")
    frequency_per_year: int = Field(description="Number of preventive maintenance actions per year")
    downtime_hours: float = Field(default=4.0, description="Downtime per preventive maintenance (hours)")
    hourly_rate: float = Field(default=500.0, description="Cost per hour of downtime ($)")


class CalculateMaintenanceCostTool(BaseTool):
    name: str = "calculate_maintenance_cost"
    description: str = "Compute annual preventive maintenance costs including downtime."
    args_schema: type = CalculateMaintenanceCostInput

    def _run(self, preventive_cost: float, frequency_per_year: int,
             downtime_hours: float = 4.0, hourly_rate: float = 500.0) -> str:
        direct_cost = preventive_cost * frequency_per_year
        downtime_cost = frequency_per_year * downtime_hours * hourly_rate
        total = direct_cost + downtime_cost
        return (
            f"Annual preventive maintenance cost: ${total:,.2f} "
            f"(direct: ${direct_cost:,.2f}, downtime: ${downtime_cost:,.2f})"
        )


class CalculateFailureCostInput(BaseModel):
    failure_probability: float = Field(description="Annual probability of unplanned failure (0-1)")
    repair_cost: float = Field(description="Cost of unplanned repair ($)")
    downtime_hours: float = Field(default=48.0, description="Downtime for unplanned failure (hours)")
    hourly_rate: float = Field(default=500.0, description="Cost per hour of downtime ($)")
    consequential_cost: float = Field(default=0.0, description="Additional consequential damage cost ($)")


class CalculateFailureCostTool(BaseTool):
    name: str = "calculate_failure_cost"
    description: str = "Estimate expected annual cost of unplanned failures."
    args_schema: type = CalculateFailureCostInput

    def _run(self, failure_probability: float, repair_cost: float,
             downtime_hours: float = 48.0, hourly_rate: float = 500.0,
             consequential_cost: float = 0.0) -> str:
        per_failure = repair_cost + (downtime_hours * hourly_rate) + consequential_cost
        expected_annual = failure_probability * per_failure
        return (
            f"Expected annual failure cost: ${expected_annual:,.2f} "
            f"(per-failure: ${per_failure:,.2f}, probability: {failure_probability:.2%})"
        )


class OptimizeMaintenanceScheduleInput(BaseModel):
    rul_estimate: float = Field(description="Estimated remaining useful life (cycles)")
    preventive_cost: float = Field(description="Cost of preventive maintenance ($)")
    failure_cost: float = Field(description="Cost of unplanned failure ($)")
    safety_margin: float = Field(default=0.2, description="Safety margin as fraction of RUL (0-1)")


class OptimizeMaintenanceScheduleTool(BaseTool):
    name: str = "optimize_maintenance_schedule"
    description: str = "Find cost-optimal RUL threshold for scheduling maintenance."
    args_schema: type = OptimizeMaintenanceScheduleInput

    def _run(self, rul_estimate: float, preventive_cost: float,
             failure_cost: float, safety_margin: float = 0.2) -> str:
        optimal_threshold = rul_estimate * (1 - safety_margin)
        cost_ratio = preventive_cost / failure_cost if failure_cost > 0 else float("inf")
        action = "Schedule preventive maintenance" if cost_ratio < 0.5 else "Monitor and reassess"
        return (
            f"Optimal maintenance threshold: {optimal_threshold:.0f} cycles "
            f"(RUL={rul_estimate:.0f}, margin={safety_margin:.0%}). "
            f"Cost ratio: {cost_ratio:.2f}. Recommendation: {action}"
        )


# ---------------------------------------------------------------------------
# Safety / Policy Evaluation Tools
# ---------------------------------------------------------------------------

class AssessSafetyRiskInput(BaseModel):
    failure_mode: str = Field(description="Description of the failure mode")
    severity: int = Field(description="Severity rating 1-10 (10=catastrophic)")
    probability: int = Field(description="Probability rating 1-10 (10=certain)")
    detectability: int = Field(description="Detectability rating 1-10 (10=undetectable)")


class AssessSafetyRiskTool(BaseTool):
    name: str = "assess_safety_risk"
    description: str = "Classify risk level (low/medium/high/critical) using RPN analysis."
    args_schema: type = AssessSafetyRiskInput

    def _run(self, failure_mode: str, severity: int, probability: int,
             detectability: int) -> str:
        rpn = severity * probability * detectability
        if rpn >= 500:
            level = "CRITICAL"
        elif rpn >= 200:
            level = "HIGH"
        elif rpn >= 80:
            level = "MEDIUM"
        else:
            level = "LOW"

        return (
            f"Safety risk assessment for '{failure_mode}': {level} "
            f"(RPN={rpn}, S={severity}, P={probability}, D={detectability})"
        )


class CheckComplianceInput(BaseModel):
    standard: str = Field(description="Standard to check against (e.g., 'IEC 61508', 'ISO 13849', 'OSHA 1910')")
    safety_integrity_level: int = Field(default=2, description="Required SIL level (1-4)")
    current_pfd: float = Field(default=0.01, description="Current probability of failure on demand")


class CheckComplianceTool(BaseTool):
    name: str = "check_compliance"
    description: str = "Validate system against IEC/ISO/OSHA safety standards."
    args_schema: type = CheckComplianceInput

    def _run(self, standard: str, safety_integrity_level: int = 2,
             current_pfd: float = 0.01) -> str:
        sil_thresholds = {1: 0.1, 2: 0.01, 3: 0.001, 4: 0.0001}
        required_pfd = sil_thresholds.get(safety_integrity_level, 0.01)
        compliant = current_pfd <= required_pfd
        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        return (
            f"Compliance check ({standard}, SIL-{safety_integrity_level}): {status}. "
            f"Required PFD <= {required_pfd}, current PFD = {current_pfd}"
        )


class GenerateSafetyRecommendationsInput(BaseModel):
    risk_level: str = Field(description="Risk level: low, medium, high, or critical")
    failure_mode: str = Field(description="Description of the failure mode")
    current_controls: str = Field(default="none", description="Existing safety controls")


class GenerateSafetyRecommendationsTool(BaseTool):
    name: str = "generate_safety_recommendations"
    description: str = "Produce prioritized safety action items based on risk assessment."
    args_schema: type = GenerateSafetyRecommendationsInput

    def _run(self, risk_level: str, failure_mode: str,
             current_controls: str = "none") -> str:
        recommendations = {
            "critical": [
                "Immediate shutdown and inspection required",
                "Activate emergency response plan",
                "Notify safety officer and regulatory body",
                "Implement redundant safety barriers",
            ],
            "high": [
                "Schedule urgent maintenance within 24 hours",
                "Increase monitoring frequency to continuous",
                "Review and update safety barriers",
            ],
            "medium": [
                "Schedule maintenance within 1 week",
                "Increase inspection frequency",
                "Review current control effectiveness",
            ],
            "low": [
                "Continue routine monitoring",
                "Include in next scheduled maintenance window",
            ],
        }

        level = risk_level.lower()
        actions = recommendations.get(level, recommendations["medium"])
        items = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
        return (
            f"Safety recommendations for '{failure_mode}' ({level} risk):\n{items}\n"
            f"Current controls: {current_controls}"
        )
