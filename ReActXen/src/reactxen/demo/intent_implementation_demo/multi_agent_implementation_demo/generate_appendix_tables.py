#!/usr/bin/env python3
"""
Generate LaTeX appendix tables for PHMForge KDD paper.

Reads acronyms_dictionary.json and tool definitions, outputs:
  - appendix_acronyms.tex  (Appendix table: Acronyms Dictionary)
  - appendix_tools.tex     (Appendix table: Complete Tools Inventory)

These can be included in the main paper via LaTeX input commands.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCENARIOS_DIR = BASE_DIR / "scenarios"
OUTPUT_DIR = Path(__file__).parent


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def generate_acronyms_table() -> str:
    """Generate LaTeX for the acronyms dictionary appendix table."""
    acronyms_file = SCENARIOS_DIR / "acronyms_dictionary.json"
    with open(acronyms_file, "r") as f:
        data = json.load(f)

    # Define category ordering and display names
    category_map = {
        "predictive_maintenance_core": "Predictive Maintenance Core",
        "dataset_acronyms": "Dataset Acronyms",
        "aircraft_engine_components": "Aircraft Engine Components",
        "machine_learning_models": "Machine Learning Models",
        "regulatory_and_standards": "Regulatory Standards",
        "technical_acronyms": "Technical Acronyms",
        "data_format_acronyms": "Data Formats",
        "evaluation_metrics": "Evaluation Metrics",
    }

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\caption{Complete acronyms dictionary used in PHMForge benchmark scenarios. "
                 r"Acronyms are grouped by functional category.}")
    lines.append(r"\label{app:acronyms}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lll@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Acronym} & \textbf{Full Form} & \textbf{Usage Context} \\")
    lines.append(r"\midrule")

    first_category = True
    for key, display_name in category_map.items():
        if key not in data:
            continue
        category_data = data[key]
        if not isinstance(category_data, dict):
            continue

        if not first_category:
            lines.append(r"\midrule")
        first_category = False

        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{\textit{{{display_name}}}}}}} \\")

        for acronym, info in category_data.items():
            if not isinstance(info, dict):
                continue
            full_form = escape_latex(info.get("full_form", acronym))
            usage = info.get("usage", info.get("definition", ""))
            # Truncate long usage strings
            if len(usage) > 80:
                usage = usage[:77] + "..."
            usage = escape_latex(usage)
            acronym_escaped = escape_latex(acronym)
            lines.append(rf"\texttt{{{acronym_escaped}}} & {full_form} & {usage} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_tools_table() -> str:
    """Generate LaTeX for the complete tools inventory appendix table."""

    # Tool definitions organized by functional category
    tool_groups = [
        {
            "group": "Data Loading",
            "server": "Prognostics",
            "tools": [
                {
                    "name": "load_dataset",
                    "description": "Loads train/test splits for PHM datasets (CMAPSS, CWRU, FEMTO, etc.)",
                    "params": "dataset\\_name: str",
                },
                {
                    "name": "load_ground_truth",
                    "description": "Loads ground truth RUL values or fault labels for validation",
                    "params": "dataset\\_name: str",
                },
            ],
        },
        {
            "group": "Model Training \\& Prediction",
            "server": "Prognostics",
            "tools": [
                {
                    "name": "train_rul_model",
                    "description": "Trains a remaining useful life prediction model",
                    "params": "dataset\\_name: str, model\\_type: str",
                },
                {
                    "name": "predict_rul",
                    "description": "Generates RUL predictions for test units",
                    "params": "dataset\\_name: str, unit\\_ids: list",
                },
                {
                    "name": "train_fault_classifier",
                    "description": "Trains a multi-class fault classifier",
                    "params": "dataset\\_name: str, model\\_type: str",
                },
                {
                    "name": "classify_faults",
                    "description": "Classifies fault types from sensor data",
                    "params": "dataset\\_name: str, samples: list",
                },
            ],
        },
        {
            "group": "Metrics \\& Verification",
            "server": "Prognostics",
            "tools": [
                {
                    "name": "calculate_mae",
                    "description": "Computes mean absolute error between predicted and actual RUL",
                    "params": "predictions: list, actuals: list",
                },
                {
                    "name": "calculate_rmse",
                    "description": "Computes root mean squared error for RUL predictions",
                    "params": "predictions: list, actuals: list",
                },
                {
                    "name": "verify_ground_truth",
                    "description": "Validates predictions against ground truth values",
                    "params": "predictions: dict, ground\\_truth: dict",
                },
                {
                    "name": "calculate_accuracy",
                    "description": "Computes classification accuracy for fault detection",
                    "params": "predictions: list, labels: list",
                },
                {
                    "name": "verify_classification",
                    "description": "Validates fault classification against labeled data",
                    "params": "results: dict, ground\\_truth: dict",
                },
            ],
        },
        {
            "group": "Engine Health Analysis",
            "server": "Prognostics",
            "tools": [
                {
                    "name": "analyze_engine_signals",
                    "description": "Parses multi-sensor engine telemetry data",
                    "params": "engine\\_id: str, sensors: list",
                },
                {
                    "name": "assess_component_health",
                    "description": "Evaluates Fan/LPC/HPC/HPT/LPT component status",
                    "params": "engine\\_id: str, component: str",
                },
                {
                    "name": "diagnose_timing_issues",
                    "description": "Identifies efficiency vs.\\ flow faults in turbofan",
                    "params": "engine\\_id: str, sensor\\_data: dict",
                },
                {
                    "name": "detect_degradation_trend",
                    "description": "Detects degradation patterns over operational cycles",
                    "params": "engine\\_id: str, window: int",
                },
            ],
        },
        {
            "group": "Cost-Benefit Analysis",
            "server": "Maintenance",
            "tools": [
                {
                    "name": "calculate_maintenance_cost",
                    "description": "Computes preventive maintenance costs with labor overhead",
                    "params": "equipment\\_type: str, strategy: str",
                },
                {
                    "name": "calculate_failure_cost",
                    "description": "Computes reactive failure costs including downtime impact",
                    "params": "equipment\\_type: str, failure\\_mode: str",
                },
                {
                    "name": "optimize_maintenance_schedule",
                    "description": "Finds cost-optimal RUL threshold for maintenance",
                    "params": "fleet\\_data: dict, cost\\_params: dict",
                },
            ],
        },
        {
            "group": "Safety/Policy Evaluation",
            "server": "Maintenance",
            "tools": [
                {
                    "name": "assess_safety_risk",
                    "description": "Classifies unit risk level (low/medium/high/critical)",
                    "params": "unit\\_id: str, sensor\\_data: dict",
                },
                {
                    "name": "check_compliance",
                    "description": "Validates against IEC/ISO/FAA regulatory standards",
                    "params": "unit\\_id: str, standards: list",
                },
                {
                    "name": "generate_safety_recommendations",
                    "description": "Produces prioritized action items for safety mitigation",
                    "params": "risk\\_assessment: dict",
                },
            ],
        },
        {
            "group": "Web Search",
            "server": "Maintenance",
            "tools": [
                {
                    "name": "web_search",
                    "description": "Searches the web for domain-specific information",
                    "params": "query: str",
                },
            ],
        },
    ]

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\caption{Complete inventory of 22 PHMForge tools across two MCP servers. "
                 r"Tools are grouped by functional category. "
                 r"$^\dagger$Prognostics Server. $^\ddagger$Intelligent Maintenance Server.}")
    lines.append(r"\label{tab:full_tools}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llll@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Tool Name} & \textbf{Server} & \textbf{Description} & \textbf{Input Parameters} \\")
    lines.append(r"\midrule")

    first_group = True
    for group in tool_groups:
        if not first_group:
            lines.append(r"\midrule")
        first_group = False

        group_name = group["group"]
        server_marker = r"$^\dagger$" if group["server"] == "Prognostics" else r"$^\ddagger$"
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{\textit{{{group_name}}}}}{server_marker}}} \\")

        for tool in group["tools"]:
            name = tool["name"].replace("_", r"\_")
            desc = tool["description"]
            params = tool["params"]
            lines.append(rf"\texttt{{{name}}} & {group['server']} & {desc} & \texttt{{{params}}} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    print("Generating appendix tables...")

    # Generate acronyms table
    acronyms_tex = generate_acronyms_table()
    output_path = OUTPUT_DIR / "appendix_acronyms.tex"
    with open(output_path, "w") as f:
        f.write(acronyms_tex)
    print(f"  Written: {output_path}")

    # Generate tools table
    tools_tex = generate_tools_table()
    output_path = OUTPUT_DIR / "appendix_tools.tex"
    with open(output_path, "w") as f:
        f.write(tools_tex)
    print(f"  Written: {output_path}")

    print("Done! Include in paper with:")
    print(r"  \input{appendix_acronyms}")
    print(r"  \input{appendix_tools}")


if __name__ == "__main__":
    main()
