#!/usr/bin/env python3
"""
Generate PHMForge performance comparison figure (TabArena style).
Creates a grouped bar chart comparing framework+model combinations across task categories.
Reads benchmark results from ../results/paper_results.json.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np

# Set style similar to TabArena
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12

# Path to results JSON (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "..", "results", "paper_results.json")

# Category display labels with scenario counts
CATEGORY_LABELS = [
    "RUL Prediction (15)",
    "Fault Classification (15)",
    "Engine Health (30)",
    "Cost-Benefit (5)",
    "Safety/Policy (10)",
]

# Category keys as they appear in the JSON
CATEGORY_KEYS = [
    "RUL Prediction",
    "Fault Classification",
    "Engine Health Analysis",
    "Cost-Benefit Analysis",
    "Safety/Policy Evaluation",
]

# Colors: Blue=RUL, Orange=Fault, Green=Health, Purple=Cost, Red=Safety
COLORS = ["#5B9BD5", "#ED7D31", "#70AD47", "#8E24AA", "#E53935"]


def load_results():
    """Load and sort results from paper_results.json by overall_score ascending."""
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)
    results = data["results"]
    # Sort by overall_score ascending
    results.sort(key=lambda r: r["overall_score"])
    return results


def make_label(entry):
    """Format a model label as 'Framework\\n+Model\\n(agent_type)'."""
    framework = entry["framework"]
    model = entry["model"]
    agent_type = "single" if entry["agent_type"] == "single_agent" else "multi"
    return f"{framework}\n+{model}\n({agent_type})"


def generate_grouped_bar_chart():
    """Main grouped bar chart: 11 models x 5 categories."""
    results = load_results()

    models = [make_label(r) for r in results]
    n_models = len(models)

    # Extract per-category accuracy percentages
    category_data = []
    for key in CATEGORY_KEYS:
        category_data.append(
            [r["scores"][key]["accuracy"] * 100 for r in results]
        )

    # Find best overall score for the reference line
    best_overall = max(r["overall_score"] for r in results) * 100

    x = np.arange(n_models)
    width = 0.15
    offsets = [-2 * width, -width, 0, width, 2 * width]

    fig, ax = plt.subplots(figsize=(18, 6))

    # Create grouped bars for each category
    all_bars = []
    for i, (cat_label, cat_values, color, offset) in enumerate(
        zip(CATEGORY_LABELS, category_data, COLORS, offsets)
    ):
        bars = ax.bar(
            x + offset,
            cat_values,
            width,
            label=cat_label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        all_bars.append(bars)

    # Reference line for best overall performance
    ax.axhline(
        y=best_overall,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Best Overall ({best_overall:.0f}%)",
    )

    # Labels and formatting
    ax.set_ylabel("Task Completion Rate (%)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Framework + Model", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only label if tall enough
                ax.annotate(
                    f"{height:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Legend
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig("agentic_performance_chart.png", dpi=300, bbox_inches="tight")
    plt.savefig("agentic_performance_chart.pdf", bbox_inches="tight")
    plt.close()

    print(
        "Figure saved as 'agentic_performance_chart.png' and 'agentic_performance_chart.pdf'"
    )


def generate_horizontal_chart():
    """Overall Completion Rate Focus (horizontal bar chart)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    results = load_results()

    # Labels and overall scores (already sorted ascending by load_results)
    models_sorted = [
        f"{r['framework']} + {r['model']} ({'single' if r['agent_type'] == 'single_agent' else 'multi'})"
        for r in results
    ]
    overall_sorted = [r["overall_score"] * 100 for r in results]

    # Find best overall score for the reference line
    best_overall = max(overall_sorted)

    # Color gradient from red (poor) to green (best)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models_sorted)))

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(
        models_sorted, overall_sorted, color=colors, edgecolor="white", linewidth=0.8
    )

    # Add value labels
    for bar, val in zip(bars, overall_sorted):
        ax.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Overall Task Completion Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, best_overall + 15)
    ax.axvline(x=best_overall, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        best_overall + 1,
        len(models_sorted) - 0.5,
        f"Best: {best_overall:.0f}%",
        fontsize=12,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig("overall_performance_horizontal.png", dpi=300, bbox_inches="tight")
    plt.savefig("overall_performance_horizontal.pdf", bbox_inches="tight")
    plt.close()

    print(
        "Horizontal figure saved as 'overall_performance_horizontal.png' and 'overall_performance_horizontal.pdf'"
    )


if __name__ == "__main__":
    generate_grouped_bar_chart()
    generate_horizontal_chart()
