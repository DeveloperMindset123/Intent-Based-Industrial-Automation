"""PHMForge Benchmark Dashboard — Streamlit application."""

import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_all_results() -> list[dict]:
    """Load all JSON result files from the results directory."""
    all_results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f, "r") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    item["_source_file"] = f.name
                all_results.extend(data)
            elif isinstance(data, dict):
                data["_source_file"] = f.name
                all_results.append(data)
    return all_results


@st.cache_data
def load_paper_results() -> dict | None:
    """Load pre-populated paper results."""
    paper_file = RESULTS_DIR / "paper_results.json"
    if paper_file.exists():
        with open(paper_file, "r") as f:
            return json.load(f)
    return None


def paper_results_to_df(paper_data: dict) -> pd.DataFrame:
    """Convert paper results to a flat DataFrame for visualization."""
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
                "label": f"{entry['framework']} + {entry['model']} ({entry.get('agent_type', 'single')})",
            })
    return pd.DataFrame(rows)


def run_results_to_df(results: list[dict]) -> pd.DataFrame:
    """Convert run results list to DataFrame."""
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        if "scores" in r:
            continue  # skip paper-format entries
        rows.append({
            "task_id": r.get("task_id", ""),
            "category": r.get("classification_type", "Unknown"),
            "dataset": r.get("dataset", ""),
            "status": r.get("status", "unknown"),
            "agent_type": r.get("agent_type", "unknown"),
            "execution_time": r.get("execution_time", 0),
            "source_file": r.get("_source_file", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PHMForge Benchmark Dashboard",
    page_icon="🔧",
    layout="wide",
)

st.title("PHMForge Benchmark Dashboard")
st.caption("Industrial Predictive Maintenance — 75 Scenarios, 5 Categories")

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

paper_data = load_paper_results()
all_results = load_all_results()
run_df = run_results_to_df(all_results)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Scenario Results",
    "Model Comparison",
    "Living Benchmark",
])

# ---- Tab 1: Overview ----
with tab1:
    st.header("Overview")

    if paper_data:
        df = paper_results_to_df(paper_data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scenarios", 75)
        col2.metric("Framework+Model Combos", len(df["label"].unique()))
        col3.metric(
            "Best Overall Score",
            f"{df.groupby('label')['overall_score'].first().max():.0%}",
        )

        # Agent type filter
        agent_filter = st.radio(
            "Agent Type",
            ["All", "single_agent", "multi_agent"],
            horizontal=True,
        )
        if agent_filter != "All":
            df = df[df["agent_type"] == agent_filter]

        fig = px.bar(
            df,
            x="label",
            y="accuracy",
            color="category",
            barmode="group",
            title="Accuracy by Framework+Model and Category",
            labels={"accuracy": "Accuracy", "label": ""},
        )
        fig.update_layout(xaxis_tickangle=-30, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No paper results found. Run benchmarks to populate data.")

# ---- Tab 2: Scenario Results ----
with tab2:
    st.header("Scenario Results")

    if not run_df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            cat_filter = st.multiselect(
                "Category",
                options=run_df["category"].unique().tolist(),
                default=run_df["category"].unique().tolist(),
            )
        with col2:
            status_filter = st.multiselect(
                "Status",
                options=run_df["status"].unique().tolist(),
                default=run_df["status"].unique().tolist(),
            )
        with col3:
            agent_filter2 = st.multiselect(
                "Agent Type",
                options=run_df["agent_type"].unique().tolist(),
                default=run_df["agent_type"].unique().tolist(),
            )

        filtered = run_df[
            (run_df["category"].isin(cat_filter))
            & (run_df["status"].isin(status_filter))
            & (run_df["agent_type"].isin(agent_filter2))
        ]
        st.dataframe(filtered, use_container_width=True, height=400)

        st.metric("Showing", f"{len(filtered)} / {len(run_df)} scenarios")
    else:
        st.info(
            "No run results yet. Use the CLI to run benchmarks:\n\n"
            "```bash\npython single_agent_implementation/run.py --limit 5\n"
            "python multi_agent_implementation/run.py --limit 5\n```"
        )

# ---- Tab 3: Model Comparison ----
with tab3:
    st.header("Model Comparison")

    if paper_data:
        df = paper_results_to_df(paper_data)

        fig = go.Figure()
        categories = df["category"].unique().tolist()

        for label in df["label"].unique():
            subset = df[df["label"] == label]
            vals = [subset[subset["category"] == c]["accuracy"].values[0] if len(subset[subset["category"] == c]) > 0 else 0 for c in categories]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=label,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Category Performance Radar",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Overall score table
        overall = (
            df.groupby("label")
            .agg(overall_score=("overall_score", "first"))
            .sort_values("overall_score", ascending=False)
            .reset_index()
        )
        overall.columns = ["Configuration", "Overall Score"]
        overall["Overall Score"] = overall["Overall Score"].apply(lambda x: f"{x:.0%}")
        st.table(overall)
    else:
        st.info("No paper results available for comparison.")

# ---- Tab 4: Living Benchmark ----
with tab4:
    st.header("Living Benchmark")

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    non_paper = [f for f in result_files if f.name != "paper_results.json"]

    if non_paper:
        timeline_data = []
        for f in non_paper:
            # Parse timestamp from filename like "single_agent_20250101_120000.json"
            parts = f.stem.split("_")
            try:
                ts_str = "_".join(parts[-2:])
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except (ValueError, IndexError):
                ts = datetime.fromtimestamp(f.stat().st_mtime)

            with open(f, "r") as fh:
                data = json.load(fh)
                n_scenarios = len(data) if isinstance(data, list) else 1
                n_completed = sum(1 for d in (data if isinstance(data, list) else [data]) if d.get("status") == "completed")

            timeline_data.append({
                "file": f.name,
                "timestamp": ts,
                "scenarios": n_scenarios,
                "completed": n_completed,
            })

        tl_df = pd.DataFrame(timeline_data).sort_values("timestamp", ascending=False)
        st.dataframe(tl_df, use_container_width=True)

        fig = px.bar(
            tl_df,
            x="timestamp",
            y="completed",
            text="file",
            title="Benchmark Runs Over Time",
            labels={"completed": "Completed Scenarios", "timestamp": "Run Date"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No benchmark runs recorded yet. Results will appear here "
            "after running the CLI benchmarks."
        )

    st.subheader("Result Files")
    for f in result_files:
        size_kb = f.stat().st_size / 1024
        st.text(f"  {f.name} ({size_kb:.1f} KB)")
