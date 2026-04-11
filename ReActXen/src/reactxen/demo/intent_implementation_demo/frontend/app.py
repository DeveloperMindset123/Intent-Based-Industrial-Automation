"""PHMForge Benchmark Dashboard — Streamlit application."""

import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
SCENARIOS_DIR = BASE_DIR / "scenarios"
FRONTEND_DIR = Path(__file__).parent
TRAJECTORIES_DIR = RESULTS_DIR / "trajectories"

# Category color map
CATEGORY_COLORS = {
    "RUL Prediction": "#1E88E5",
    "Fault Classification": "#43A047",
    "Engine Health Analysis": "#FB8C00",
    "Cost-Benefit Analysis": "#8E24AA",
    "Safety/Policy Evaluation": "#E53935",
}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_scenarios() -> list[dict]:
    """Load all 75 scenarios."""
    scenario_file = SCENARIOS_DIR / "phm_scenarios.json"
    if not scenario_file.exists():
        return []
    with open(scenario_file, "r") as f:
        data = json.load(f)
    return data.get("pdm_scenarios", [])


@st.cache_data
def load_paper_results() -> dict | None:
    """Load pre-populated paper results."""
    paper_file = RESULTS_DIR / "paper_results.json"
    if not paper_file.exists():
        return None
    with open(paper_file, "r") as f:
        return json.load(f)


@st.cache_data
def load_run_results() -> list[dict]:
    """Load all benchmark run result files (excluding paper_results)."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "paper_results.json":
            continue
        with open(f, "r") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    item["_source_file"] = f.name
                results.extend(data)
    return results


@st.cache_data
def load_tools_inventory() -> dict | None:
    """Load tool inventory for Tool Explorer."""
    inv_file = FRONTEND_DIR / "tools_inventory.json"
    if not inv_file.exists():
        return None
    with open(inv_file, "r") as f:
        return json.load(f)


@st.cache_data
def load_trajectories() -> dict[str, dict]:
    """Load all trajectory replay files."""
    trajectories = {}
    if not TRAJECTORIES_DIR.exists():
        return trajectories
    for f in sorted(TRAJECTORIES_DIR.glob("trajectory_*.json")):
        with open(f, "r") as fh:
            trajectories[f.stem] = json.load(fh)
    return trajectories


@st.cache_data
def load_ablation_results() -> dict | None:
    """Load ablation study results."""
    ablation_file = RESULTS_DIR / "ablation_results.json"
    if not ablation_file.exists():
        return None
    with open(ablation_file, "r") as f:
        return json.load(f)


def scenarios_to_df(scenarios: list[dict]) -> pd.DataFrame:
    """Convert scenario list to a DataFrame with key columns."""
    rows = []
    for s in scenarios:
        gt = s.get("ground_truth", {})
        rows.append(
            {
                "task_id": s.get("task_id", ""),
                "category": s.get("classification_type", "Unknown"),
                "dataset": s.get("dataset", ""),
                "required_tools": ", ".join(s.get("required_tools", [])),
                "n_tools": len(s.get("required_tools", [])),
                "has_ground_truth": bool(gt),
                "expected_format": gt.get("expected_output_format", ""),
                "question_preview": (s.get("input_question", "") or "")[:120] + "...",
            }
        )
    return pd.DataFrame(rows)


def paper_results_to_df(paper_data: dict) -> pd.DataFrame:
    """Convert paper results to a flat DataFrame."""
    rows = []
    for entry in paper_data.get("results", []):
        for category, scores in entry.get("scores", {}).items():
            rows.append(
                {
                    "framework": entry["framework"],
                    "model": entry["model"],
                    "agent_type": entry.get("agent_type", "single_agent"),
                    "category": category,
                    "accuracy": scores["accuracy"],
                    "completed": scores["completed"],
                    "total": scores["total"],
                    "overall_score": entry.get("overall_score", 0),
                    "label": f"{entry['framework']} + {entry['model']}",
                    "config": f"{entry['framework']} + {entry['model']} ({entry.get('agent_type', 'single').replace('_', ' ')})",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PHMForge Benchmark Dashboard",
    page_icon="wrench",
    layout="wide",
)

st.title("PHMForge Benchmark Dashboard")
st.caption("PHMForge: A Scenario-Driven Agentic Benchmark for Industrial Asset Lifecycle Maintenance | 75 Scenarios | 5 Categories | KDD 2026")

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

scenarios = load_scenarios()
paper_data = load_paper_results()
run_results = load_run_results()
tools_inventory = load_tools_inventory()
trajectories = load_trajectories()
ablation_data = load_ablation_results()
scenario_df = scenarios_to_df(scenarios) if scenarios else pd.DataFrame()
paper_df = paper_results_to_df(paper_data) if paper_data else pd.DataFrame()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    all_categories = (
        sorted(scenario_df["category"].unique().tolist())
        if not scenario_df.empty
        else []
    )
    selected_categories = st.multiselect(
        "Categories",
        options=all_categories,
        default=all_categories,
    )

    all_datasets = (
        sorted(scenario_df["dataset"].unique().tolist())
        if not scenario_df.empty
        else []
    )
    selected_datasets = st.multiselect(
        "Datasets",
        options=all_datasets,
        default=all_datasets,
    )

    agent_type_filter = st.radio(
        "Agent Type (Paper Results)",
        ["All", "Single Agent", "Multi Agent"],
        horizontal=True,
    )

    st.divider()
    st.markdown(
        f"**{len(scenarios)}** scenarios loaded  \n"
        f"**{len(all_datasets)}** unique datasets  \n"
        f"**{len(paper_df['config'].unique()) if not paper_df.empty else 0}** model configurations"
    )

# Apply sidebar filters
if not scenario_df.empty:
    filtered_scenarios = scenario_df[
        (scenario_df["category"].isin(selected_categories))
        & (scenario_df["dataset"].isin(selected_datasets))
    ]
else:
    filtered_scenarios = pd.DataFrame()

if not paper_df.empty:
    filtered_paper = paper_df[paper_df["category"].isin(selected_categories)]
    if agent_type_filter == "Single Agent":
        filtered_paper = filtered_paper[filtered_paper["agent_type"] == "single_agent"]
    elif agent_type_filter == "Multi Agent":
        filtered_paper = filtered_paper[filtered_paper["agent_type"] == "multi_agent"]
else:
    filtered_paper = pd.DataFrame()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Overview",
        "Scenarios",
        "Benchmark Results",
        "Model Comparison",
        "Ablation Study",
        "Run History",
        "Tool Explorer",
        "Playground",
        "MCP Servers",
    ]
)

# ---- Tab 1: Overview ----
with tab1:
    st.header("Benchmark Overview")

    if not scenario_df.empty:
        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Scenarios", len(scenarios))
        col2.metric("Categories", len(all_categories))
        col3.metric("Datasets", len(all_datasets))
        if not paper_df.empty:
            col4.metric(
                "Best Score",
                f"{paper_df.groupby('config')['overall_score'].first().max():.0%}",
            )
        else:
            col4.metric("Best Score", "N/A")

        st.subheader("Scenarios by Category")
        cat_counts = scenario_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.bar(
            cat_counts,
            x="Category",
            y="Count",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            text="Count",
        )
        fig_cat.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

        # Datasets per category
        st.subheader("Dataset Distribution")
        ds_cat = (
            scenario_df.groupby(["category", "dataset"])
            .size()
            .reset_index(name="count")
        )
        fig_ds = px.treemap(
            ds_cat,
            path=["category", "dataset"],
            values="count",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig_ds.update_layout(height=450)
        st.plotly_chart(fig_ds, use_container_width=True)

        # Required tools summary
        st.subheader("Tools Required per Category")
        tool_data = []
        for _, row in scenario_df.iterrows():
            for tool in row["required_tools"].split(", "):
                if tool.strip():
                    tool_data.append(
                        {"category": row["category"], "tool": tool.strip()}
                    )
        if tool_data:
            tool_df = pd.DataFrame(tool_data)
            tool_counts = (
                tool_df.groupby(["category", "tool"]).size().reset_index(name="count")
            )
            fig_tools = px.bar(
                tool_counts.sort_values("count", ascending=True).tail(20),
                x="count",
                y="tool",
                color="category",
                orientation="h",
                color_discrete_map=CATEGORY_COLORS,
                title="Top 20 Most Required Tools",
                height=500,
            )
            st.plotly_chart(fig_tools, use_container_width=True)

# ---- Tab 2: Scenarios ----
with tab2:
    st.header("Scenario Explorer")

    if not filtered_scenarios.empty:
        st.dataframe(
            filtered_scenarios[
                [
                    "task_id",
                    "category",
                    "dataset",
                    "n_tools",
                    "expected_format",
                    "question_preview",
                ]
            ],
            use_container_width=True,
            height=400,
            column_config={
                "task_id": st.column_config.TextColumn("Task ID", width="small"),
                "category": st.column_config.TextColumn("Category", width="medium"),
                "dataset": st.column_config.TextColumn("Dataset", width="small"),
                "n_tools": st.column_config.NumberColumn("Tools", width="small"),
                "expected_format": st.column_config.TextColumn(
                    "Output Format", width="small"
                ),
                "question_preview": st.column_config.TextColumn(
                    "Question", width="large"
                ),
            },
        )
        st.caption(f"Showing {len(filtered_scenarios)} of {len(scenario_df)} scenarios")

        # Scenario detail expander
        st.subheader("Scenario Details")
        selected_id = st.selectbox(
            "Select a scenario to view details",
            options=filtered_scenarios["task_id"].tolist(),
        )

        if selected_id:
            scenario = next(
                (s for s in scenarios if s.get("task_id") == selected_id), None
            )
            if scenario:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"**Category:** {scenario.get('classification_type', '')}"
                    )
                    st.markdown(f"**Dataset:** {scenario.get('dataset', '')}")
                    st.markdown(
                        f"**Required Tools:** {', '.join(scenario.get('required_tools', []))}"
                    )
                with col2:
                    gt = scenario.get("ground_truth", {})
                    st.markdown(
                        f"**Output Format:** {gt.get('expected_output_format', 'N/A')}"
                    )
                    st.markdown(
                        f"**Verification Required:** {gt.get('verification_required', 'N/A')}"
                    )

                st.markdown("---")
                st.markdown("**Input Question:**")
                st.text_area(
                    "",
                    value=scenario.get("input_question", ""),
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                )

                if scenario.get("dependency_analysis"):
                    with st.expander("Dependency Analysis"):
                        st.text(scenario["dependency_analysis"])

                if gt:
                    with st.expander("Ground Truth"):
                        st.json(gt)

                if scenario.get("procedure"):
                    with st.expander("Procedure"):
                        st.json(scenario["procedure"])
    else:
        st.info("No scenarios match the current filters.")

# ---- Tab 3: Benchmark Results ----
with tab3:
    st.header("Benchmark Results (Paper)")

    if not filtered_paper.empty:
        # Accuracy by config and category
        st.subheader("Accuracy by Configuration and Category")
        fig_bench = px.bar(
            filtered_paper,
            x="config",
            y="accuracy",
            color="category",
            barmode="group",
            color_discrete_map=CATEGORY_COLORS,
            text=filtered_paper["accuracy"].apply(lambda x: f"{x:.0%}"),
        )
        fig_bench.update_layout(
            xaxis_tickangle=-25,
            height=500,
            xaxis_title="",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        # Completed scenarios heatmap
        st.subheader("Scenarios Completed (out of total)")
        pivot_completed = (
            filtered_paper.pivot_table(
                index="config", columns="category", values="completed", aggfunc="first"
            )
            .fillna(0)
            .astype(int)
        )
        pivot_total = (
            filtered_paper.pivot_table(
                index="config", columns="category", values="total", aggfunc="first"
            )
            .fillna(0)
            .astype(int)
        )

        # Display as annotated text
        display_df = pivot_completed.copy()
        for col in display_df.columns:
            display_df[col] = [
                f"{c}/{t}" for c, t in zip(pivot_completed[col], pivot_total[col])
            ]
        st.dataframe(display_df, use_container_width=True)

        # Accuracy heatmap
        st.subheader("Accuracy Heatmap")
        pivot_acc = filtered_paper.pivot_table(
            index="config", columns="category", values="accuracy", aggfunc="first"
        ).fillna(0)

        fig_heat = px.imshow(
            pivot_acc,
            text_auto=".0%",
            color_continuous_scale="RdYlGn",
            zmin=0,
            zmax=1,
            aspect="auto",
        )
        fig_heat.update_layout(height=400, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)

        # Overall scores ranking
        st.subheader("Overall Score Ranking")
        overall = (
            filtered_paper.groupby("config")
            .agg(
                overall_score=("overall_score", "first"),
                framework=("framework", "first"),
                model=("model", "first"),
                agent_type=("agent_type", "first"),
            )
            .sort_values("overall_score", ascending=False)
            .reset_index()
        )

        fig_rank = px.bar(
            overall,
            x="config",
            y="overall_score",
            color="agent_type",
            text=overall["overall_score"].apply(lambda x: f"{x:.0%}"),
            color_discrete_map={"single_agent": "#42A5F5", "multi_agent": "#EF5350"},
        )
        fig_rank.update_layout(
            height=400,
            xaxis_tickangle=-25,
            xaxis_title="",
            yaxis_title="Overall Score",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        # Single vs Multi agent comparison
        st.subheader("Single Agent vs Multi Agent")
        agent_comparison = (
            filtered_paper.groupby(["agent_type", "category"])
            .agg(
                avg_accuracy=("accuracy", "mean"),
            )
            .reset_index()
        )

        if len(agent_comparison["agent_type"].unique()) > 1:
            fig_agent = px.bar(
                agent_comparison,
                x="category",
                y="avg_accuracy",
                color="agent_type",
                barmode="group",
                text=agent_comparison["avg_accuracy"].apply(lambda x: f"{x:.0%}"),
                color_discrete_map={
                    "single_agent": "#42A5F5",
                    "multi_agent": "#EF5350",
                },
            )
            fig_agent.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="Avg Accuracy",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_agent, use_container_width=True)

    else:
        st.info("No benchmark results match the current filters.")

# ---- Tab 4: Model Comparison ----
with tab4:
    st.header("Model & Framework Comparison")

    if not filtered_paper.empty:
        # Radar chart
        st.subheader("Performance Radar")
        categories_list = filtered_paper["category"].unique().tolist()

        fig_radar = go.Figure()
        for config in filtered_paper["config"].unique():
            subset = filtered_paper[filtered_paper["config"] == config]
            vals = []
            for c in categories_list:
                match = subset[subset["category"] == c]
                vals.append(match["accuracy"].values[0] if len(match) > 0 else 0)
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories_list + [categories_list[0]],
                    name=config,
                    fill="toself",
                    opacity=0.3,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=600,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Framework comparison
        st.subheader("Framework Comparison (ReAct vs ReActXen)")
        fw_comparison = (
            filtered_paper.groupby(["framework", "category"])
            .agg(
                avg_accuracy=("accuracy", "mean"),
            )
            .reset_index()
        )

        fig_fw = px.bar(
            fw_comparison,
            x="category",
            y="avg_accuracy",
            color="framework",
            barmode="group",
            text=fw_comparison["avg_accuracy"].apply(lambda x: f"{x:.0%}"),
            color_discrete_map={"ReAct": "#78909C", "ReActXen": "#1E88E5"},
        )
        fig_fw.update_layout(
            height=400,
            xaxis_title="",
            yaxis_title="Avg Accuracy",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_fw, use_container_width=True)

        # Model comparison
        st.subheader("Model Comparison")
        model_comparison = (
            filtered_paper.groupby(["model", "category"])
            .agg(
                avg_accuracy=("accuracy", "mean"),
            )
            .reset_index()
        )

        fig_model = px.bar(
            model_comparison,
            x="category",
            y="avg_accuracy",
            color="model",
            barmode="group",
            text=model_comparison["avg_accuracy"].apply(lambda x: f"{x:.0%}"),
        )
        fig_model.update_layout(
            height=400,
            xaxis_title="",
            yaxis_title="Avg Accuracy",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_model, use_container_width=True)

        # Detailed results table
        st.subheader("Detailed Results Table")
        detail_table = filtered_paper[
            ["config", "category", "accuracy", "completed", "total"]
        ].copy()
        detail_table["accuracy_pct"] = detail_table["accuracy"].apply(
            lambda x: f"{x:.0%}"
        )
        detail_table["completion"] = detail_table.apply(
            lambda r: f"{r['completed']}/{r['total']}", axis=1
        )
        st.dataframe(
            detail_table[["config", "category", "accuracy_pct", "completion"]].rename(
                columns={
                    "config": "Configuration",
                    "category": "Category",
                    "accuracy_pct": "Accuracy",
                    "completion": "Completed/Total",
                }
            ),
            use_container_width=True,
            height=500,
        )

    else:
        st.info("No paper results available for comparison.")

# ---- Tab 5: Ablation Study ----
with tab5:
    st.header("Ablation Study: MCP Tool Utility Validation")
    st.caption(
        "Systematic ablation comparing full MCP tools vs data-loader-only vs no-tools (MLE-Bench style) conditions. "
        "Best-performing configuration: Claude Code + Claude Opus 4.6 (multi-agent)."
    )

    if ablation_data:
        ablation_results = ablation_data.get("ablation_results", [])
        categories = [
            "RUL Prediction",
            "Fault Classification",
            "Engine Health Analysis",
            "Cost-Benefit Analysis",
            "Safety/Policy Evaluation",
        ]

        # Summary metrics
        st.subheader("Overall Performance by Condition")
        cols = st.columns(3)
        condition_colors = {
            "Full Tools (Baseline)": "#43A047",
            "Data Loader Only": "#FB8C00",
            "No Tools (MLE-Bench Style)": "#E53935",
        }
        for i, entry in enumerate(ablation_results):
            with cols[i]:
                st.metric(
                    entry["condition"],
                    f"{entry['overall_score']:.0%}",
                    delta=f"{entry['overall_score'] - ablation_results[0]['overall_score']:.0%}"
                    if i > 0
                    else None,
                    delta_color="inverse" if i > 0 else "off",
                )
                st.caption(f"Tools available: {entry['tools_available']}")

        # Bar chart comparison
        st.subheader("Performance by Category and Condition")
        bar_data = []
        for entry in ablation_results:
            for cat in categories:
                score = entry["scores"].get(cat, {})
                bar_data.append(
                    {
                        "Condition": entry["condition"],
                        "Category": cat,
                        "Accuracy": score.get("accuracy", 0),
                        "Completed": score.get("completed", 0),
                        "Total": score.get("total", 0),
                    }
                )

        bar_df = pd.DataFrame(bar_data)
        fig_ablation = px.bar(
            bar_df,
            x="Category",
            y="Accuracy",
            color="Condition",
            barmode="group",
            text=bar_df["Accuracy"].apply(lambda x: f"{x:.0%}"),
            color_discrete_map=condition_colors,
        )
        fig_ablation.update_layout(
            height=500,
            xaxis_title="",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ablation, use_container_width=True)

        # Delta table
        st.subheader("Performance Delta (Full Tools vs Ablated)")
        delta_rows = []
        for cat in categories:
            full = ablation_results[0]["scores"][cat]["accuracy"]
            loader = ablation_results[1]["scores"][cat]["accuracy"]
            no_tools = ablation_results[2]["scores"][cat]["accuracy"]
            delta_rows.append(
                {
                    "Category": cat,
                    "Full Tools": f"{full:.0%}",
                    "Data Loader Only": f"{loader:.0%}",
                    "No Tools": f"{no_tools:.0%}",
                    "Drop (Full vs No Tools)": f"-{(full - no_tools):.0%}",
                    "Drop (Full vs Loader Only)": f"-{(full - loader):.0%}",
                }
            )
        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

        # Error analysis
        st.subheader("Error Analysis by Condition")
        err_col1, err_col2 = st.columns(2)

        with err_col1:
            st.markdown("**Data Loader Only — Failure Breakdown**")
            loader_failures = ablation_results[1].get("failure_analysis", {})
            if loader_failures:
                fail_df = pd.DataFrame(
                    [
                        {"Error Type": k.replace("_", " ").title(), "Rate": v}
                        for k, v in loader_failures.items()
                    ]
                )
                fig_fail1 = px.pie(
                    fail_df, names="Error Type", values="Rate", hole=0.4
                )
                fig_fail1.update_layout(height=300)
                st.plotly_chart(fig_fail1, use_container_width=True)

        with err_col2:
            st.markdown("**No Tools — Failure Breakdown**")
            no_tool_failures = ablation_results[2].get("failure_analysis", {})
            if no_tool_failures:
                fail_df2 = pd.DataFrame(
                    [
                        {"Error Type": k.replace("_", " ").title(), "Rate": v}
                        for k, v in no_tool_failures.items()
                    ]
                )
                fig_fail2 = px.pie(
                    fail_df2, names="Error Type", values="Rate", hole=0.4
                )
                fig_fail2.update_layout(height=300)
                st.plotly_chart(fig_fail2, use_container_width=True)

        # Primary errors table
        st.subheader("Primary Error Modes per Category")
        error_rows = []
        for cat in categories:
            error_rows.append(
                {
                    "Category": cat,
                    "Full Tools Error": ablation_results[0]["scores"][cat].get(
                        "primary_error", "—"
                    ),
                    "Data Loader Only Error": ablation_results[1]["scores"][cat].get(
                        "primary_error", "—"
                    ),
                    "No Tools Error": ablation_results[2]["scores"][cat].get(
                        "primary_error", "—"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(error_rows), use_container_width=True, hide_index=True)

        # Auxiliary metrics section
        st.divider()
        st.subheader("Auxiliary Process Metrics (Full Tools Condition)")
        aux = ablation_data.get("auxiliary_metrics", {})
        if aux:
            am1, am2, am3, am4 = st.columns(4)
            am1.metric(
                "Tool Accuracy",
                f"{aux.get('tool_accuracy', {}).get('value', 0):.0%}",
            )
            am2.metric(
                "Required Tool Coverage",
                f"{aux.get('required_tool_coverage', {}).get('value', 0):.0%}",
            )
            am3.metric(
                "Excess Tool Calls",
                f"{aux.get('excess_tool_calls', {}).get('value', 0):.0%}",
            )
            am4.metric(
                "Tool Sequencing Accuracy",
                f"{aux.get('tool_sequencing_accuracy', {}).get('value', 0):.0%}",
            )

            # Tool accuracy breakdown
            tool_acc = aux.get("tool_accuracy", {}).get("breakdown", {})
            if tool_acc:
                st.markdown("**Tool Accuracy by Category**")
                ta_df = pd.DataFrame(
                    [
                        {"Tool Category": k.replace("_", " ").title(), "Accuracy": v}
                        for k, v in tool_acc.items()
                    ]
                )
                fig_ta = px.bar(
                    ta_df,
                    x="Tool Category",
                    y="Accuracy",
                    text=ta_df["Accuracy"].apply(lambda x: f"{x:.0%}"),
                    color_discrete_sequence=["#1E88E5"],
                )
                fig_ta.update_layout(
                    height=300, yaxis=dict(range=[0, 1.05]), xaxis_title=""
                )
                st.plotly_chart(fig_ta, use_container_width=True)

        # Summary finding
        summary = ablation_data.get("summary", {})
        if summary:
            st.info(f"**Key Finding:** {summary.get('key_finding', '')}")
    else:
        st.info(
            "No ablation study results found. Expected at `results/ablation_results.json`."
        )

# ---- Tab 6: Run History (Enhanced) ----
with tab6:
    st.header("Run History")

    if run_results:
        run_df = pd.DataFrame(
            [
                {
                    "task_id": r.get("task_id", ""),
                    "category": r.get("classification_type", "Unknown"),
                    "dataset": r.get("dataset", ""),
                    "status": r.get("status", "unknown"),
                    "agent_type": r.get("agent_type", "unknown"),
                    "execution_time": r.get("execution_time", 0),
                    "source_file": r.get("_source_file", ""),
                }
                for r in run_results
            ]
        )

        # Summary metrics row
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Runs", len(run_df))
        completed_count = len(run_df[run_df["status"] == "completed"])
        m2.metric("Completed", completed_count)
        failed_count = len(run_df[run_df["status"] == "failed"])
        m3.metric("Failed", failed_count)
        avg_time = run_df[run_df["execution_time"] > 0]["execution_time"].mean()
        m4.metric(
            "Avg Execution Time", f"{avg_time:.1f}s" if pd.notna(avg_time) else "N/A"
        )

        # Status filter
        status_filter = st.radio(
            "Filter by Status",
            ["All", "Completed", "Failed"],
            horizontal=True,
            key="run_status_filter",
        )
        display_run_df = run_df.copy()
        if status_filter == "Completed":
            display_run_df = display_run_df[display_run_df["status"] == "completed"]
        elif status_filter == "Failed":
            display_run_df = display_run_df[display_run_df["status"] == "failed"]

        # Category breakdown bar chart
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Category Breakdown")
            cat_breakdown = display_run_df["category"].value_counts().reset_index()
            cat_breakdown.columns = ["Category", "Count"]
            fig_cat_run = px.bar(
                cat_breakdown,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                text="Count",
            )
            fig_cat_run.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_cat_run, use_container_width=True)

        with col_right:
            st.subheader("Execution Time Distribution")
            time_data = display_run_df[display_run_df["execution_time"] > 0]
            if not time_data.empty:
                fig_time = px.histogram(
                    time_data,
                    x="execution_time",
                    color="agent_type",
                    nbins=15,
                    labels={"execution_time": "Execution Time (s)"},
                    color_discrete_map={
                        "single_agent": "#42A5F5",
                        "multi_agent": "#EF5350",
                    },
                )
                fig_time.update_layout(height=300)
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("No execution time data available.")

        # Per-run file comparison
        st.subheader("Results by Source File")
        source_files = display_run_df["source_file"].unique().tolist()
        if source_files:
            file_summary = []
            for sf in source_files:
                sf_df = display_run_df[display_run_df["source_file"] == sf]
                file_summary.append(
                    {
                        "File": sf,
                        "Scenarios": len(sf_df),
                        "Completed": len(sf_df[sf_df["status"] == "completed"]),
                        "Failed": len(sf_df[sf_df["status"] == "failed"]),
                        "Agent Type": (
                            sf_df["agent_type"].mode().iloc[0]
                            if not sf_df["agent_type"].mode().empty
                            else "unknown"
                        ),
                        "Avg Time (s)": f"{sf_df['execution_time'].mean():.1f}",
                    }
                )
            st.dataframe(pd.DataFrame(file_summary), use_container_width=True)

        # Full results table
        st.subheader("Detailed Run Results")
        st.dataframe(display_run_df, use_container_width=True, height=400)
    else:
        st.info(
            "No benchmark runs recorded yet. Run benchmarks with the CLI:\n\n"
            "```bash\n"
            "python single_agent_implementation/run.py --limit 5\n"
            "python multi_agent_implementation/run.py --limit 5\n"
            "```"
        )

    # List result files
    st.subheader("Result Files")
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if result_files:
        for f in result_files:
            size_kb = f.stat().st_size / 1024
            st.text(f"{f.name} ({size_kb:.1f} KB)")
    else:
        st.text("No result files found.")

# ---- Tab 7: Tool Explorer ----
with tab7:
    st.header("Tool Explorer")

    if tools_inventory:
        # Build sunburst data
        sunburst_data = []
        for server, categories in tools_inventory.items():
            for category, tools_list in categories.items():
                for tool in tools_list:
                    sunburst_data.append(
                        {
                            "server": server,
                            "category": category,
                            "tool": tool["name"],
                            "description": tool.get("description", ""),
                            "params": ", ".join(tool.get("params", [])),
                            "value": 1,
                        }
                    )

        sunburst_df = pd.DataFrame(sunburst_data)

        st.subheader("MCP Server / Category / Tool Hierarchy")
        fig_sun = px.sunburst(
            sunburst_df,
            path=["server", "category", "tool"],
            values="value",
            color="server",
            color_discrete_map={
                "Prognostics Server": "#1E88E5",
                "Maintenance Server": "#E53935",
            },
        )
        fig_sun.update_layout(height=600)
        st.plotly_chart(fig_sun, use_container_width=True)

        # Tool detail panel
        st.subheader("Tool Details")
        all_tool_names = sunburst_df["tool"].tolist()
        selected_tool = st.selectbox(
            "Select a tool to view details", options=all_tool_names
        )

        if selected_tool:
            tool_row = sunburst_df[sunburst_df["tool"] == selected_tool].iloc[0]
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"**Tool:** `{tool_row['tool']}`")
                st.markdown(f"**Server:** {tool_row['server']}")
                st.markdown(f"**Category:** {tool_row['category']}")
            with tc2:
                st.markdown(f"**Description:** {tool_row['description']}")
                st.markdown(f"**Parameters:** `{tool_row['params']}`")

            # Cross-reference with scenarios
            if scenarios:
                matching_scenarios = [
                    s for s in scenarios if selected_tool in s.get("required_tools", [])
                ]
                if matching_scenarios:
                    st.markdown(f"**Used in {len(matching_scenarios)} scenarios:**")
                    scenario_refs = []
                    for s in matching_scenarios:
                        scenario_refs.append(
                            {
                                "Task ID": s.get("task_id", ""),
                                "Category": s.get("classification_type", ""),
                                "Dataset": s.get("dataset", ""),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(scenario_refs),
                        use_container_width=True,
                        height=200,
                    )
                else:
                    st.info(
                        "This tool is not directly listed in any scenario's required_tools."
                    )

        # Summary stats
        st.subheader("Tool Distribution Summary")
        server_counts = sunburst_df.groupby("server").size().reset_index(name="Tools")
        cat_counts = (
            sunburst_df.groupby(["server", "category"]).size().reset_index(name="Tools")
        )
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Tools per Server**")
            st.dataframe(server_counts, use_container_width=True)
        with s2:
            st.markdown("**Tools per Category**")
            st.dataframe(cat_counts, use_container_width=True)
    else:
        st.info(
            "Tool inventory file not found. Expected at `frontend/tools_inventory.json`."
        )

# ---- Tab 8: Interactive Playground ----
with tab8:
    st.header("Interactive Playground")
    st.caption("Replay pre-recorded agent execution trajectories step-by-step.")

    if trajectories:
        # Scenario selector
        traj_options = {}
        for key, traj in trajectories.items():
            scenario_info = traj.get("scenario", {})
            label = f"{scenario_info.get('task_id', key)} — {scenario_info.get('classification_type', 'Unknown')} ({scenario_info.get('dataset', '')})"
            traj_options[label] = key

        selected_traj_label = st.selectbox(
            "Select a scenario to replay",
            options=list(traj_options.keys()),
        )
        selected_traj_key = traj_options[selected_traj_label]
        traj_data = trajectories[selected_traj_key]

        # Agent type and replay speed
        col_a, col_b = st.columns(2)
        with col_a:
            agent_display = (
                traj_data.get("agent_type", "multi_agent").replace("_", " ").title()
            )
            st.markdown(f"**Agent Type:** {agent_display}")
            st.markdown(f"**Total Time:** {traj_data.get('total_time', 0):.1f}s")
        with col_b:
            replay_speed = st.slider(
                "Replay Speed",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                format="%.1fx",
            )

        # Scenario info
        scenario_info = traj_data.get("scenario", {})
        with st.expander("Scenario Details", expanded=False):
            st.markdown(f"**Task ID:** {scenario_info.get('task_id', '')}")
            st.markdown(f"**Category:** {scenario_info.get('classification_type', '')}")
            st.markdown(f"**Dataset:** {scenario_info.get('dataset', '')}")
            st.markdown(f"**Question:** {scenario_info.get('question_preview', '')}")

        # Live mode notice
        st.info(
            "**Trajectory Replay Mode** — Replaying a pre-recorded agent execution trace. Live Mode (requiring local API keys) coming soon."
        )

        # Run button
        if st.button("Run Scenario", type="primary"):
            steps = traj_data.get("steps", [])
            if not steps:
                st.warning("No steps found in this trajectory.")
            else:
                # Step type styling
                step_icons = {
                    "thought": "brain",
                    "action": "hammer_and_wrench",
                    "observation": "mag",
                    "final_answer": "white_check_mark",
                }
                step_colors = {
                    "thought": "blue",
                    "action": "orange",
                    "observation": "green",
                    "final_answer": "rainbow",
                }

                for step in steps:
                    step_type = step.get("type", "thought")
                    step_num = step.get("step", 0)
                    content = step.get("content", "")
                    tool_name = step.get("tool", "")
                    step_time = step.get("time", 1.0)
                    step_input = step.get("input", {})

                    # Simulate delay based on replay speed
                    delay = step_time / replay_speed
                    time.sleep(min(delay, 2.0))  # Cap at 2s per step

                    icon = step_icons.get(step_type, "arrow_right")
                    label = step_type.replace("_", " ").title()

                    if step_type == "action":
                        with st.status(
                            f"Step {step_num}: {label} — `{tool_name}`",
                            state="complete",
                        ):
                            st.markdown(f"**Tool:** `{tool_name}`")
                            if step_input:
                                st.json(step_input)
                            st.markdown(content)
                            st.caption(f"{step_time:.1f}s")
                    elif step_type == "final_answer":
                        st.success(f"**Step {step_num}: Final Answer**\n\n{content}")
                        st.caption(
                            f"Total replay time simulated: {traj_data.get('total_time', 0):.1f}s"
                        )
                    elif step_type == "thought":
                        with st.status(f"Step {step_num}: {label}", state="complete"):
                            st.markdown(f"*{content}*")
                            st.caption(f"{step_time:.1f}s")
                    else:  # observation
                        with st.status(f"Step {step_num}: {label}", state="complete"):
                            st.code(content, language="text")
                            st.caption(f"{step_time:.1f}s")

        # Static view option
        with st.expander("View Full Trajectory (Static)"):
            steps = traj_data.get("steps", [])
            for step in steps:
                step_type = step.get("type", "thought")
                step_num = step.get("step", 0)
                content = step.get("content", "")
                tool_name = step.get("tool", "")

                if step_type == "thought":
                    st.markdown(f"**Step {step_num} — Thought:** *{content}*")
                elif step_type == "action":
                    st.markdown(
                        f"**Step {step_num} — Action:** `{tool_name}` — {content}"
                    )
                elif step_type == "observation":
                    st.markdown(f"**Step {step_num} — Observation:**")
                    st.code(content, language="text")
                elif step_type == "final_answer":
                    st.markdown(f"**Step {step_num} — Final Answer:**")
                    st.success(content)
    else:
        st.info(
            "No trajectory files found. Expected in `results/trajectories/`.\n\n"
            "Trajectory files are pre-recorded agent execution traces in JSON format."
        )


# ---- Tab 9: MCP Servers ----
with tab9:
    st.header("MCP Server Status & Metrics")

    # Server architecture overview
    st.subheader("Architecture")
    st.markdown(
        "PHMForge uses **2 MCP servers** with **22 tools** total, "
        "connected via the Model Context Protocol (FastMCP framework)."
    )

    # Server definitions (static — matches the actual servers)
    MCP_SERVERS = {
        "prognostics": {
            "display_name": "Prognostics Server",
            "tools": 15,
            "categories": [
                "Data Loading",
                "Model Training",
                "Prediction",
                "Metrics",
                "Engine Health",
            ],
            "transport": "stdio (SSE-capable)",
            "tool_list": [
                ("load_dataset", "Data Loading", "Load dataset from PDMBench"),
                ("load_ground_truth", "Data Loading", "Load ground truth RUL values"),
                ("train_rul_model", "Model Training", "Train RUL prediction model"),
                ("predict_rul", "Prediction", "Predict remaining useful life"),
                (
                    "train_fault_classifier",
                    "Model Training",
                    "Train fault classification model",
                ),
                ("classify_faults", "Prediction", "Classify faults for test units"),
                ("calculate_mae", "Metrics", "Calculate Mean Absolute Error"),
                ("calculate_rmse", "Metrics", "Calculate Root Mean Squared Error"),
                (
                    "verify_ground_truth",
                    "Metrics",
                    "Verify predictions against ground truth",
                ),
                ("calculate_accuracy", "Metrics", "Calculate classification accuracy"),
                ("verify_classification", "Metrics", "Verify fault classifications"),
                (
                    "analyze_engine_signals",
                    "Engine Health",
                    "Parse multi-sensor signal data",
                ),
                (
                    "assess_component_health",
                    "Engine Health",
                    "Evaluate component health status",
                ),
                ("diagnose_timing_issues", "Engine Health", "Identify fault patterns"),
                (
                    "detect_degradation_trend",
                    "Engine Health",
                    "Analyze degradation trends",
                ),
            ],
        },
        "maintenance": {
            "display_name": "Maintenance Server",
            "tools": 7,
            "categories": ["Cost-Benefit", "Safety/Policy", "Utility"],
            "transport": "stdio (SSE-capable)",
            "tool_list": [
                (
                    "calculate_maintenance_cost",
                    "Cost-Benefit",
                    "Compute annual preventive maintenance costs",
                ),
                (
                    "calculate_failure_cost",
                    "Cost-Benefit",
                    "Estimate expected annual failure cost",
                ),
                (
                    "optimize_maintenance_schedule",
                    "Cost-Benefit",
                    "Find cost-optimal maintenance threshold",
                ),
                (
                    "assess_safety_risk",
                    "Safety/Policy",
                    "Classify risk using RPN analysis",
                ),
                (
                    "check_compliance",
                    "Safety/Policy",
                    "Validate against safety standards",
                ),
                (
                    "generate_safety_recommendations",
                    "Safety/Policy",
                    "Produce safety action items",
                ),
                ("web_search", "Utility", "Search internet via Brave API"),
            ],
        },
    }

    # Server cards
    col1, col2 = st.columns(2)

    for (server_name, info), col in zip(MCP_SERVERS.items(), [col1, col2]):
        with col:
            st.metric(info["display_name"], f"{info['tools']} tools")
            st.caption(f"Transport: {info['transport']}")
            st.markdown(f"Categories: {', '.join(info['categories'])}")

            # Tool table
            tool_df = pd.DataFrame(
                info["tool_list"],
                columns=["Tool Name", "Category", "Description"],
            )
            st.dataframe(
                tool_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # MCP Eval Metrics section
    st.subheader("MCP Evaluation Metrics")
    st.markdown("PHMForge tracks MCP-specific quality dimensions beyond task accuracy:")

    mcp_eval_file = RESULTS_DIR / "mcp_eval_results.json"
    if mcp_eval_file.exists():
        with open(mcp_eval_file) as f:
            mcp_results = json.load(f)

        agg = mcp_results.get("aggregate", {})
        if agg:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scenarios Evaluated", agg.get("scenarios_evaluated", 0))
            m2.metric(
                "Avg Tool Success Rate", f"{agg.get('avg_tool_success_rate', 0):.1%}"
            )
            m3.metric(
                "Avg Schema Compliance",
                f"{agg.get('avg_schema_compliance_rate', 0):.1%}",
            )
            m4.metric(
                "Avg Discovery Latency",
                f"{agg.get('avg_tool_discovery_latency_ms', 0):.0f}ms",
            )

            # Per-scenario results
            scenario_results = mcp_results.get("scenarios", [])
            if scenario_results:
                eval_df = pd.DataFrame(scenario_results)
                if "mcp_metrics" in eval_df.columns:
                    metrics_df = pd.json_normalize(eval_df["mcp_metrics"])
                    metrics_df["scenario_id"] = eval_df["scenario_id"]
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No MCP eval results found. Run the benchmark with MCP evaluation enabled "
            "to generate `results/mcp_eval_results.json`.\n\n"
            "**Tracked metrics:**\n"
            "- Tool discovery latency (ms)\n"
            "- Tool call success rate\n"
            "- Schema compliance rate\n"
            "- Routing accuracy\n"
            "- Context efficiency (chars consumed)"
        )

    st.divider()

    # Tool usage across scenarios
    st.subheader("Tool Usage Across Scenarios")
    if scenarios:
        tool_counts = {}
        for s in scenarios:
            for t in s.get("required_tools", []):
                tool_counts[t] = tool_counts.get(t, 0) + 1

        usage_df = pd.DataFrame(
            sorted(tool_counts.items(), key=lambda x: -x[1]),
            columns=["Tool", "Scenarios"],
        )

        # Add server column
        prog_tools = {t[0] for t in MCP_SERVERS["prognostics"]["tool_list"]}
        usage_df["Server"] = usage_df["Tool"].apply(
            lambda t: "prognostics" if t in prog_tools else "maintenance"
        )

        fig = px.bar(
            usage_df,
            x="Tool",
            y="Scenarios",
            color="Server",
            color_discrete_map={"prognostics": "#1E88E5", "maintenance": "#E53935"},
            title="Tool Usage Distribution Across 75 Scenarios",
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
