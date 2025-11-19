"""
Table formatting utilities for agent responses.
"""
from typing import List, Dict, Any, Optional

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not available
    def tabulate(data, headers, tablefmt="grid", floatfmt=".2f"):
        """Simple fallback table formatter."""
        if not data:
            return "No data to display"
        result = " | ".join(str(h) for h in headers) + "\n"
        result += "-" * len(result) + "\n"
        for row in data:
            result += " | ".join(str(cell) for cell in row) + "\n"
        return result

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def format_rul_results_table(
    equipment_data: List[Dict[str, Any]],
    include_verification: bool = False
) -> str:
    """
    Format RUL prediction results as a table.
    
    Args:
        equipment_data: List of dicts with equipment information
        include_verification: Whether to include ground truth verification
    
    Returns:
        Formatted table string
    """
    if not equipment_data:
        return "No equipment data to display."
    
    # Prepare table data
    headers = ["Equipment ID", "RUL (cycles)", "Risk Level", "Maintenance Cost", "Maintenance Type"]
    
    if include_verification:
        headers.extend(["Actual RUL", "Error", "Match"])
    
    table_data = []
    for eq in equipment_data:
        row = [
            eq.get("equipment_id", "N/A"),
            eq.get("rul", "N/A"),
            eq.get("risk_level", "N/A"),
            f"${eq.get('cost', 0):,.2f}" if eq.get('cost') else "N/A",
            eq.get("maintenance_type", "N/A")
        ]
        
        if include_verification:
            row.extend([
                eq.get("actual_rul", "N/A"),
                eq.get("error", "N/A"),
                "✅" if eq.get("match") else "❌" if eq.get("match") is False else "N/A"
            ])
        
        table_data.append(row)
    
    return tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f")


def format_safety_recommendations_table(safety_data: List[Dict[str, Any]]) -> str:
    """Format safety recommendations as a table."""
    if not safety_data:
        return "No safety recommendations to display."
    
    headers = ["Equipment ID", "Safety Protocol", "OSHA Reference", "Priority"]
    table_data = []
    
    for safety in safety_data:
        table_data.append([
            safety.get("equipment_id", "N/A"),
            safety.get("protocol", "N/A"),
            safety.get("osha_reference", "N/A"),
            safety.get("priority", "N/A")
        ])
    
    return tabulate(table_data, headers=headers, tablefmt="grid")


def format_cost_analysis_table(cost_data: List[Dict[str, Any]]) -> str:
    """Format cost analysis as a table."""
    if not cost_data:
        return "No cost data to display."
    
    headers = ["Equipment ID", "Preventive Cost", "Corrective Cost", "Recommended", "ROI"]
    table_data = []
    
    total_preventive = 0
    total_corrective = 0
    
    for cost in cost_data:
        preventive = cost.get("preventive_cost", 0)
        corrective = cost.get("corrective_cost", 0)
        total_preventive += preventive
        total_corrective += corrective
        
        table_data.append([
            cost.get("equipment_id", "N/A"),
            f"${preventive:,.2f}",
            f"${corrective:,.2f}",
            cost.get("recommended", "N/A"),
            f"{cost.get('roi', 0):.1f}%"
        ])
    
    # Add total row
    table_data.append([
        "TOTAL",
        f"${total_preventive:,.2f}",
        f"${total_corrective:,.2f}",
        "",
        ""
    ])
    
    return tabulate(table_data, headers=headers, tablefmt="grid")


def format_comprehensive_results_table(
    equipment_list: List[Dict[str, Any]],
    include_all: bool = True
) -> str:
    """
    Format comprehensive results combining all information.
    
    Args:
        equipment_list: List of equipment with all data
        include_all: Whether to include all columns
    
    Returns:
        Formatted comprehensive table
    """
    if not equipment_list:
        return "No equipment data to display."
    
    # Determine headers based on available data
    headers = ["Equipment ID", "RUL (cycles)"]
    
    sample = equipment_list[0] if equipment_list else {}
    
    if include_all or "risk_level" in sample:
        headers.append("Risk Level")
    if include_all or "cost" in sample:
        headers.append("Cost")
    if include_all or "maintenance_type" in sample:
        headers.append("Maintenance Type")
    if include_all or "safety_protocol" in sample:
        headers.append("Safety Protocol")
    if include_all or "actual_rul" in sample:
        headers.append("Actual RUL")
    if include_all or "error" in sample:
        headers.append("Error")
    
    table_data = []
    for eq in equipment_list:
        row = [
            eq.get("equipment_id", "N/A"),
            eq.get("rul", "N/A")
        ]
        
        if include_all or "risk_level" in eq:
            row.append(eq.get("risk_level", "N/A"))
        if include_all or "cost" in eq:
            cost = eq.get("cost", 0)
            row.append(f"${cost:,.2f}" if cost else "N/A")
        if include_all or "maintenance_type" in eq:
            row.append(eq.get("maintenance_type", "N/A"))
        if include_all or "safety_protocol" in eq:
            row.append(eq.get("safety_protocol", "N/A")[:50] + "..." if len(str(eq.get("safety_protocol", ""))) > 50 else eq.get("safety_protocol", "N/A"))
        if include_all or "actual_rul" in eq:
            row.append(eq.get("actual_rul", "N/A"))
        if include_all or "error" in eq:
            error = eq.get("error")
            row.append(f"{error} cycles" if error is not None else "N/A")
        
        table_data.append(row)
    
    return tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f")


class FormatTableTool(BaseTool):
    """Tool to format data as tables."""
    
    name: str = "format_table"
    description: str = """Format data as a readable table.
    
    Input: JSON with:
    - data: List of dictionaries with data to format
    - table_type: Type of table ('rul', 'safety', 'cost', 'comprehensive')
    - include_verification: Whether to include ground truth verification (for RUL tables)
    
    Returns: Formatted table string.
    """
    
    class FormatInput(BaseModel):
        data: List[Dict[str, Any]] = Field(description="List of dictionaries to format")
        table_type: str = Field(default="comprehensive", description="Type of table: 'rul', 'safety', 'cost', 'comprehensive'")
        include_verification: bool = Field(default=False, description="Include ground truth verification")
    
    args_schema: type[BaseModel] = FormatInput
    
    def _run(
        self,
        data: List[Dict[str, Any]],
        table_type: str = "comprehensive",
        include_verification: bool = False
    ) -> str:
        """Format data as table."""
        try:
            if table_type == "rul":
                return format_rul_results_table(data, include_verification)
            elif table_type == "safety":
                return format_safety_recommendations_table(data)
            elif table_type == "cost":
                return format_cost_analysis_table(data)
            elif table_type == "comprehensive":
                return format_comprehensive_results_table(data, include_all=True)
            else:
                return f"Unknown table type: {table_type}. Use 'rul', 'safety', 'cost', or 'comprehensive'"
        except Exception as e:
            return f"Error formatting table: {str(e)}"


def create_table_formatting_tools() -> List[BaseTool]:
    """Create table formatting tools."""
    return [FormatTableTool()]

