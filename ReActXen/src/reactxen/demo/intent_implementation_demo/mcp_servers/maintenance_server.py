"""Intelligent Maintenance MCP Server — Cost-Benefit, Safety/Policy, and Web Search tools.

Migrated to FastMCP framework for compatibility with AssetOpsBench.
Supports both stdio and SSE transports.
"""

import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Ensure tools package is importable
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Logging
_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("maintenance-mcp-server")

mcp = FastMCP("maintenance")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ErrorResult(BaseModel):
    error: str


class MaintenanceCostResult(BaseModel):
    direct_cost: float
    downtime_cost: float
    total_annual_cost: float
    message: str


class FailureCostResult(BaseModel):
    per_failure_cost: float
    failure_probability: float
    expected_annual_cost: float
    message: str


class ScheduleResult(BaseModel):
    optimal_threshold: float
    cost_ratio: float
    recommendation: str
    message: str


class SafetyRiskResult(BaseModel):
    failure_mode: str
    risk_level: str
    rpn: int
    severity: int
    probability: int
    detectability: int


class ComplianceResult(BaseModel):
    standard: str
    sil_level: int
    status: str
    required_pfd: float
    current_pfd: float


class SafetyRecommendationsResult(BaseModel):
    risk_level: str
    failure_mode: str
    recommendations: list[str]
    current_controls: str


class WebSearchResult(BaseModel):
    query: str
    results: str


# ---------------------------------------------------------------------------
# Cost-Benefit Analysis Tools (3)
# ---------------------------------------------------------------------------

@mcp.tool()
def calculate_maintenance_cost(
    preventive_cost: float,
    frequency_per_year: int,
    downtime_hours: float = 4.0,
    hourly_rate: float = 500.0,
) -> MaintenanceCostResult:
    """Compute annual preventive maintenance costs including downtime.

    Parameters:
    - preventive_cost: Cost of a single preventive maintenance action ($)
    - frequency_per_year: Number of preventive maintenance actions per year
    - downtime_hours: Downtime per maintenance (hours, default 4)
    - hourly_rate: Cost per hour of downtime ($, default 500)
    """
    direct_cost = preventive_cost * frequency_per_year
    downtime_cost = frequency_per_year * downtime_hours * hourly_rate
    total = direct_cost + downtime_cost
    return MaintenanceCostResult(
        direct_cost=direct_cost, downtime_cost=downtime_cost,
        total_annual_cost=total,
        message=f"Annual preventive maintenance cost: ${total:,.2f} (direct: ${direct_cost:,.2f}, downtime: ${downtime_cost:,.2f})",
    )


@mcp.tool()
def calculate_failure_cost(
    failure_probability: float,
    repair_cost: float,
    downtime_hours: float = 48.0,
    hourly_rate: float = 500.0,
    consequential_cost: float = 0.0,
) -> FailureCostResult:
    """Estimate expected annual cost of unplanned failures.

    Parameters:
    - failure_probability: Annual probability of unplanned failure (0-1)
    - repair_cost: Cost of unplanned repair ($)
    - downtime_hours: Downtime for unplanned failure (hours, default 48)
    - hourly_rate: Cost per hour of downtime ($, default 500)
    - consequential_cost: Additional consequential damage cost ($, default 0)
    """
    per_failure = repair_cost + (downtime_hours * hourly_rate) + consequential_cost
    expected_annual = failure_probability * per_failure
    return FailureCostResult(
        per_failure_cost=per_failure, failure_probability=failure_probability,
        expected_annual_cost=expected_annual,
        message=f"Expected annual failure cost: ${expected_annual:,.2f} (per-failure: ${per_failure:,.2f}, probability: {failure_probability:.2%})",
    )


@mcp.tool()
def optimize_maintenance_schedule(
    rul_estimate: float,
    preventive_cost: float,
    failure_cost: float,
    safety_margin: float = 0.2,
) -> ScheduleResult:
    """Find cost-optimal RUL threshold for scheduling maintenance.

    Parameters:
    - rul_estimate: Estimated remaining useful life (cycles)
    - preventive_cost: Cost of preventive maintenance ($)
    - failure_cost: Cost of unplanned failure ($)
    - safety_margin: Safety margin as fraction of RUL (0-1, default 0.2)
    """
    optimal_threshold = rul_estimate * (1 - safety_margin)
    cost_ratio = preventive_cost / failure_cost if failure_cost > 0 else float("inf")
    action = "Schedule preventive maintenance" if cost_ratio < 0.5 else "Monitor and reassess"
    return ScheduleResult(
        optimal_threshold=optimal_threshold, cost_ratio=cost_ratio,
        recommendation=action,
        message=f"Optimal threshold: {optimal_threshold:.0f} cycles (RUL={rul_estimate:.0f}, margin={safety_margin:.0%}). Recommendation: {action}",
    )


# ---------------------------------------------------------------------------
# Safety / Policy Evaluation Tools (3)
# ---------------------------------------------------------------------------

@mcp.tool()
def assess_safety_risk(
    failure_mode: str,
    severity: int,
    probability: int,
    detectability: int,
) -> SafetyRiskResult:
    """Classify risk level (low/medium/high/critical) using RPN analysis.

    Parameters:
    - failure_mode: Description of the failure mode
    - severity: Severity rating 1-10 (10=catastrophic)
    - probability: Probability rating 1-10 (10=certain)
    - detectability: Detectability rating 1-10 (10=undetectable)
    """
    rpn = severity * probability * detectability
    if rpn >= 500:
        level = "CRITICAL"
    elif rpn >= 200:
        level = "HIGH"
    elif rpn >= 80:
        level = "MEDIUM"
    else:
        level = "LOW"
    return SafetyRiskResult(
        failure_mode=failure_mode, risk_level=level, rpn=rpn,
        severity=severity, probability=probability, detectability=detectability,
    )


@mcp.tool()
def check_compliance(
    standard: str,
    safety_integrity_level: int = 2,
    current_pfd: float = 0.01,
) -> ComplianceResult:
    """Validate system against IEC/ISO/OSHA safety standards.

    Parameters:
    - standard: Standard to check (e.g., 'IEC 61508', 'ISO 13849', 'OSHA 1910')
    - safety_integrity_level: Required SIL level 1-4 (default 2)
    - current_pfd: Current probability of failure on demand (default 0.01)
    """
    sil_thresholds = {1: 0.1, 2: 0.01, 3: 0.001, 4: 0.0001}
    required_pfd = sil_thresholds.get(safety_integrity_level, 0.01)
    compliant = current_pfd <= required_pfd
    status = "COMPLIANT" if compliant else "NON-COMPLIANT"
    return ComplianceResult(
        standard=standard, sil_level=safety_integrity_level,
        status=status, required_pfd=required_pfd, current_pfd=current_pfd,
    )


@mcp.tool()
def generate_safety_recommendations(
    risk_level: str,
    failure_mode: str,
    current_controls: str = "none",
) -> SafetyRecommendationsResult:
    """Produce prioritized safety action items based on risk assessment.

    Parameters:
    - risk_level: Risk level: low, medium, high, or critical
    - failure_mode: Description of the failure mode
    - current_controls: Existing safety controls (default 'none')
    """
    recommendations_map = {
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
    actions = recommendations_map.get(level, recommendations_map["medium"])
    return SafetyRecommendationsResult(
        risk_level=level, failure_mode=failure_mode,
        recommendations=actions, current_controls=current_controls,
    )


# ---------------------------------------------------------------------------
# Web Search Tool (1)
# ---------------------------------------------------------------------------

@mcp.tool()
def web_search(query: str, count: int = 5) -> WebSearchResult:
    """Search the internet for information using Brave Search API.

    Parameters:
    - query: Search query string
    - count: Number of results (default 5)
    """
    try:
        from langchain_community.tools import BraveSearch

        brave_api_key = os.environ.get("BRAVE_API_KEY", "")
        if brave_api_key:
            search = BraveSearch.from_api_key(api_key=brave_api_key)
            results = search.run(query)
            return WebSearchResult(query=query, results=results[:500])
    except (ImportError, Exception):
        pass

    return WebSearchResult(
        query=query,
        results=f"Web search for '{query}' would return {count} results (API key not configured)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    logger.info(f"Starting maintenance server with {transport} transport")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
