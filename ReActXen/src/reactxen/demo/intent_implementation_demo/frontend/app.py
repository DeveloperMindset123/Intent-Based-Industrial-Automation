"""PHMForge Benchmark Dashboard — Streamlit application."""

import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
SCENARIOS_DIR = BASE_DIR / "scenarios"

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


def scenarios_to_df(scenarios: list[dict]) -> pd.DataFrame:
    """Convert scenario list to a DataFrame with key columns."""
    rows = []
    for s in scenarios:
        gt = s.get("ground_truth", {})
        rows.append({
            "task_id": s.get("task_id", ""),
            "category": s.get("classification_type", "Unknown"),
            "dataset": s.get("dataset", ""),
            "required_tools": ", ".join(s.get("required_tools", [])),
            "n_tools": len(s.get("required_tools", [])),
            "has_ground_truth": bool(gt),
            "expected_format": gt.get("expected_output_format", ""),
            "question_preview": (s.get("input_question", "") or "")[:120] + "...",
        })
    return pd.DataFrame(rows)


def paper_results_to_df(paper_data: dict) -> pd.DataFrame:
    """Convert paper results to a flat DataFrame."""
    rows = []
    for entry in paper_data.get("results", []):
        for category, scores in entry.get("scores", {}).items():
            rows.append({
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
            })
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
st.caption("Intent-Based Industrial Automation | 75 Scenarios | 5 Categories | KDD 2025")

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

scenarios = load_scenarios()
paper_data = load_paper_results()
run_results = load_run_results()
scenario_df = scenarios_to_df(scenarios) if scenarios else pd.DataFrame()
paper_df = paper_results_to_df(paper_data) if paper_data else pd.DataFrame()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    all_categories = sorted(scenario_df["category"].unique().tolist()) if not scenario_df.empty else []
    selected_categories = st.multiselect(
        "Categories",
        options=all_categories,
        default=all_categories,
    )

    all_datasets = sorted(scenario_df["dataset"].unique().tolist()) if not scenario_df.empty else []
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Scenarios",
    "Benchmark Results",
    "Model Comparison",
    "Run History",
])

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
        ds_cat = scenario_df.groupby(["category", "dataset"]).size().reset_index(name="count")
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
                    tool_data.append({"category": row["category"], "tool": tool.strip()})
        if tool_data:
            tool_df = pd.DataFrame(tool_data)
            tool_counts = tool_df.groupby(["category", "tool"]).size().reset_index(name="count")
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
            filtered_scenarios[["task_id", "category", "dataset", "n_tools", "expected_format", "question_preview"]],
            use_container_width=True,
            height=400,
            column_config={
                "task_id": st.column_config.TextColumn("Task ID", width="small"),
                "category": st.column_config.TextColumn("Category", width="medium"),
                "dataset": st.column_config.TextColumn("Dataset", width="small"),
                "n_tools": st.column_config.NumberColumn("Tools", width="small"),
                "expected_format": st.column_config.TextColumn("Output Format", width="small"),
                "question_preview": st.column_config.TextColumn("Question", width="large"),
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
            scenario = next((s for s in scenarios if s.get("task_id") == selected_id), None)
            if scenario:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Category:** {scenario.get('classification_type', '')}")
                    st.markdown(f"**Dataset:** {scenario.get('dataset', '')}")
                    st.markdown(f"**Required Tools:** {', '.join(scenario.get('required_tools', []))}")
                with col2:
                    gt = scenario.get("ground_truth", {})
                    st.markdown(f"**Output Format:** {gt.get('expected_output_format', 'N/A')}")
                    st.markdown(f"**Verification Required:** {gt.get('verification_required', 'N/A')}")

                st.markdown("---")
                st.markdown("**Input Question:**")
                st.text_area("", value=scenario.get("input_question", ""), height=150, disabled=True, label_visibility="collapsed")

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
        pivot_completed = filtered_paper.pivot_table(
            index="config", columns="category", values="completed", aggfunc="first"
        ).fillna(0).astype(int)
        pivot_total = filtered_paper.pivot_table(
            index="config", columns="category", values="total", aggfunc="first"
        ).fillna(0).astype(int)

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
            zmin=0, zmax=1,
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
        agent_comparison = filtered_paper.groupby(["agent_type", "category"]).agg(
            avg_accuracy=("accuracy", "mean"),
        ).reset_index()

        if len(agent_comparison["agent_type"].unique()) > 1:
            fig_agent = px.bar(
                agent_comparison,
                x="category",
                y="avg_accuracy",
                color="agent_type",
                barmode="group",
                text=agent_comparison["avg_accuracy"].apply(lambda x: f"{x:.0%}"),
                color_discrete_map={"single_agent": "#42A5F5", "multi_agent": "#EF5350"},
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
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories_list + [categories_list[0]],
                name=config,
                fill="toself",
                opacity=0.3,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=600,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Framework comparison
        st.subheader("Framework Comparison (ReAct vs ReActXen)")
        fw_comparison = filtered_paper.groupby(["framework", "category"]).agg(
            avg_accuracy=("accuracy", "mean"),
        ).reset_index()

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
        model_comparison = filtered_paper.groupby(["model", "category"]).agg(
            avg_accuracy=("accuracy", "mean"),
        ).reset_index()

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
        detail_table = filtered_paper[["config", "category", "accuracy", "completed", "total"]].copy()
        detail_table["accuracy_pct"] = detail_table["accuracy"].apply(lambda x: f"{x:.0%}")
        detail_table["completion"] = detail_table.apply(lambda r: f"{r['completed']}/{r['total']}", axis=1)
        st.dataframe(
            detail_table[["config", "category", "accuracy_pct", "completion"]].rename(columns={
                "config": "Configuration",
                "category": "Category",
                "accuracy_pct": "Accuracy",
                "completion": "Completed/Total",
            }),
            use_container_width=True,
            height=500,
        )

    else:
        st.info("No paper results available for comparison.")

# ---- Tab 5: Run History ----
with tab5:
    st.header("Run History")

    if run_results:
        run_df = pd.DataFrame([
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
        ])
        st.dataframe(run_df, use_container_width=True, height=400)
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
