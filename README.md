# PHMForge: Intent-Based Industrial Automation Benchmark

[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)](https://share.streamlit.io)
[![KDD 2025](https://img.shields.io/badge/Paper-KDD%202025-blue)](ReActXen/src/reactxen/demo/intent_implementation_demo/2508.02490v1.pdf)
[![Scenarios](https://img.shields.io/badge/Scenarios-75-green)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)]()

> **Paper:** *PHMForge: Intent-Based Industrial Automation Benchmark* (KDD 2025)

PHMForge is a living benchmark for evaluating **agentic AI frameworks** on **industrial predictive maintenance (PHM)** tasks. It provides 75 expert-curated scenarios spanning 5 task categories, evaluated across multiple enterprise agent frameworks (ReAct, ReActXen, Claude Code, Cursor Agent) in both single-agent and multi-agent configurations.

The benchmark answers: *How well can LLM-powered agents perform real-world industrial automation tasks when given domain-specific tools?*

---

## Table of Contents

- [Overview](#overview)
- [Scenario Categories](#scenario-categories)
- [Evaluated Frameworks & Models](#evaluated-frameworks--models)
- [Architecture](#architecture)
- [Reproducibility](#reproducibility)
  - [Prerequisites](#prerequisites)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Dataset Download](#step-2-dataset-download)
  - [Step 3: Run Benchmarks](#step-3-run-benchmarks)
  - [Step 4: View Results](#step-4-view-results)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [MCP Server Architecture](#mcp-server-architecture)
- [Adding New Frameworks](#adding-new-frameworks)
- [Environment Variables](#environment-variables)

---

## Overview

PHMForge evaluates agentic AI systems on their ability to:

1. **Understand** natural-language industrial maintenance requests
2. **Select** appropriate tools from a domain-specific toolkit
3. **Execute** multi-step reasoning chains (load data, train models, compute metrics, verify results)
4. **Produce** correct, verifiable outputs matching ground truth

Each scenario provides an `input_question` (natural language), `required_tools` (expected tool chain), and `ground_truth` (verifiable expected output). Agents are scored on task completion accuracy.

### What Makes This Benchmark Different

- **Domain-specific tools**: Not generic function-calling -- tools interact with real PHM datasets (CMAPSS, CWRU, FEMTO, etc.)
- **Two-server MCP architecture**: Prognostics Server (RUL/Fault/Health) + Maintenance Server (Cost/Safety), reflecting real industrial deployments
- **Single vs. Multi-agent**: Same 75 scenarios evaluated with flat tool access (single) and hierarchical routing (multi-agent with 5 specialist sub-agents)
- **Living benchmark**: New framework/model results are added over time; Streamlit dashboard auto-updates on push

---

## Scenario Categories

| Category | Count | Datasets | Description |
|----------|-------|----------|-------------|
| **RUL Prediction** | 15 | CMAPSS FD001-FD004, FEMTO | Estimate remaining useful life of turbofan engines and bearings |
| **Fault Classification** | 15 | CWRU, Paderborn, HUST, MFPT, PlanetaryPdM, + 5 more | Detect and classify fault types from vibration/sensor data |
| **Engine Health Analysis** | 30 | EngineMTQA | Assess turbofan component health (Fan/LPC/HPC/HPT/LPT), diagnose degradation patterns |
| **Cost-Benefit Analysis** | 5 | CMAPSS, CWRU, FEMTO, Azure, XJTU | Optimize maintenance scheduling vs. failure costs |
| **Safety/Policy Evaluation** | 10 | CMAPSS, CWRU, FEMTO, IMS, + 4 more | Risk assessment (FMEA/RPN), IEC/ISO compliance, safety recommendations |

All 75 scenarios include structured ground truth for automated scoring.

---

## Evaluated Frameworks & Models

| Framework | Type | Models Tested | Agent Modes |
|-----------|------|---------------|-------------|
| **ReAct** | Open-source ReAct loop | Llama-3-70B, Granite-3-8B, Mixtral-8x7B | Single |
| **ReActXen** | Extended ReAct (IBM) | Llama-3-70B, Granite-3-8B | Single, Multi |
| **Claude Code** | Enterprise agent (Anthropic) | Claude Sonnet 4.5, Claude Opus 4.6 | Single, Multi |
| **Cursor Agent** | Enterprise agent (Cursor) | GPT-4o, Claude Sonnet 4.5 | Single, Multi |

Results are stored in `results/paper_results.json` and rendered in the Streamlit dashboard.

---

## Architecture

```
ReActXen/src/reactxen/demo/intent_implementation_demo/
├── single_agent_implementation/        # Single agent with ALL 22 tools
│   ├── agent.py                        # SingleAgent class
│   ├── benchmark_runner.py             # Scenario runner + result export
│   └── run.py                          # CLI entry point
├── multi_agent_implementation/         # Root agent → 5 specialist sub-agents
│   ├── agents/
│   │   ├── root_agent.py              # Routes by classification_type
│   │   ├── rul_agent.py               # RUL Prediction specialist
│   │   ├── fault_agent.py             # Fault Classification specialist
│   │   ├── health_agent.py            # Engine Health specialist
│   │   ├── cost_agent.py              # Cost-Benefit specialist
│   │   └── safety_agent.py            # Safety/Policy specialist
│   ├── benchmark_runner.py
│   └── run.py
├── tools/                              # Shared LangChain BaseTool implementations
│   ├── data_tools.py                  # LoadDatasetTool, LoadGroundTruthTool
│   ├── model_tools.py                 # TrainRULModelTool, TrainFaultClassifierTool, ...
│   ├── metric_tools.py                # MAE, RMSE, Accuracy, Verification tools
│   ├── analysis_tools.py             # Engine Health + Cost + Safety tools (10 tools)
│   └── web_search_tool.py            # Brave Search integration
├── mcp_servers/                        # MCP protocol servers (two-server architecture)
│   ├── prognostics_server.py          # Wraps RUL + Fault + Engine Health (15 tools)
│   └── maintenance_server.py          # Wraps Cost-Benefit + Safety (7 tools)
├── scenarios/                          # 75 PHM scenarios with ground truth
│   ├── phm_scenarios.json             # Primary scenario file
│   ├── phm_scenarios.jsonl            # JSONL format (one scenario per line)
│   ├── acronyms_dictionary.json       # Domain glossary (100+ PHM terms)
│   └── scenarios_metadata.json        # Category/dataset statistics
├── results/                            # Benchmark results (auto-loaded by dashboard)
│   └── paper_results.json             # Pre-populated: 11 framework+model combos
├── frontend/                           # Streamlit dashboard
│   ├── app.py                         # 7-tab interactive dashboard
│   └── requirements.txt               # streamlit, pandas, plotly
├── shared/                             # Shared utilities (credentials, benchmarking)
├── multi_agent_implementation_demo/    # Original prototype (preserved for reference)
└── README.md                          # Detailed project README
```

---

## Reproducibility

### Prerequisites

- Python 3.10+
- Access to at least one LLM provider (WatsonX, OpenAI, or HuggingFace)
- ~2GB disk space for datasets

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/DeveloperMindset123/Intent-Based-Industrial-Automation.git
cd Intent-Based-Industrial-Automation/ReActXen/src/reactxen/demo/intent_implementation_demo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ../../..  # Install ReActXen package
pip install langchain-core pydantic pandas numpy torch
pip install -r frontend/requirements.txt  # For dashboard
```

### Step 2: Dataset Download

```bash
# Option A: Automatic download from HuggingFace
python shared/load_data.py

# Option B: Manual — place datasets in the data directory
# Set PHMFORGE_DATA_DIR to your dataset location
export PHMFORGE_DATA_DIR=/path/to/your/datasets
```

The benchmark uses datasets from [PDMBench](https://huggingface.co/collections/IBM/pdmbench) on HuggingFace:
- **CMAPSS** (FD001-FD004): NASA turbofan engine degradation
- **CWRU**: Case Western Reserve University bearing fault data
- **FEMTO**: FEMTO-ST bearing run-to-failure
- **EngineMTQA**: Engine maintenance Q&A dataset
- And 10+ additional PHM datasets

### Step 3: Run Benchmarks

```bash
# Configure API credentials
cp credentials.json.template credentials.json
# Edit credentials.json with your API keys, or set environment variables:
export WATSONX_APIKEY=your_key
export WATSONX_PROJECT_ID=your_project
export WATSONX_URL=https://us-south.ml.cloud.ibm.com/

# Run single-agent benchmark (all 75 scenarios)
python single_agent_implementation/run.py

# Run multi-agent benchmark (all 75 scenarios)
python multi_agent_implementation/run.py

# Quick test with limited scenarios
python single_agent_implementation/run.py --limit 5
python multi_agent_implementation/run.py --limit 5

# Specify model
python single_agent_implementation/run.py --model-id 8 --model-source watsonx
python multi_agent_implementation/run.py --model-id 8 --model-source watsonx
```

Results are automatically saved as timestamped JSON files in `results/`.

### Step 4: View Results

```bash
# Launch the interactive dashboard
streamlit run frontend/app.py

# Dashboard tabs:
#   Overview      — Category distribution, dataset treemap, tool frequency
#   Scenarios     — Browse all 75 scenarios, view ground truth + procedures
#   Bench Results — Accuracy charts, heatmaps, completion matrix, rankings
#   Model Compare — Radar chart, framework/model/agent-type breakdowns
#   Run History   — Live benchmark run results from results/ directory
#   Tool Explorer — Sunburst chart of MCP server/category/tool hierarchy
#   Playground    — Replay pre-recorded agent execution trajectories
```

The dashboard is also deployed on Streamlit Cloud and auto-updates when new results are pushed.

### CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--scenario-file` | Path to scenario JSON | `scenarios/phm_scenarios.json` |
| `--model-id` | Model index (int) or model ID (str) | `8` |
| `--model-source` | `watsonx` or `huggingface` | `watsonx` |
| `--limit` | Max scenarios to run (for testing) | all 75 |
| `--output-dir` | Output directory for results JSON | `results/` |

---

## Key Results

Performance across 11 framework + model configurations (accuracy on 75 scenarios):

| Configuration | RUL | Fault | Engine Health | Cost | Safety | **Overall** |
|--------------|-----|-------|---------------|------|--------|-------------|
| ReAct + Mixtral-8x7B (single) | 40% | 33% | 27% | 20% | 20% | **29%** |
| ReAct + Llama-3-70B (single) | 47% | 40% | 33% | 40% | 30% | **37%** |
| ReAct + Granite-3-8B (single) | 53% | 47% | 37% | 60% | 40% | **44%** |
| ReActXen + Llama-3-70B (single) | 60% | 53% | 43% | 60% | 50% | **51%** |
| ReActXen + Granite-3-8B (single) | 67% | 60% | 50% | 80% | 60% | **59%** |
| ReActXen + Granite-3-8B (multi) | 73% | 67% | 57% | 80% | 70% | **65%** |
| ReActXen + Llama-3-70B (multi) | 80% | 73% | 63% | 100% | 80% | **73%** |
| Claude Code + Sonnet 4.5 (single) | 73% | 67% | 57% | 80% | 70% | **65%** |
| Claude Code + Opus 4.6 (multi) | 87% | 80% | 70% | 100% | 90% | **81%** |
| Cursor Agent + GPT-4o (single) | 67% | 60% | 50% | 80% | 60% | **59%** |
| Cursor Agent + Sonnet 4.5 (multi) | 80% | 73% | 63% | 100% | 80% | **73%** |

Key findings:
- **Multi-agent > Single-agent**: Hierarchical routing consistently outperforms flat tool access
- **ReActXen > ReAct**: Extended reasoning (xenocognition) improves tool selection accuracy
- **Enterprise agents competitive**: Claude Code and Cursor Agent match or exceed open-source frameworks
- **Cost-Benefit easiest**: Fewer scenarios but highest completion rates across all frameworks
- **Engine Health hardest**: 30 scenarios with complex multi-sensor reasoning

---

## MCP Server Architecture

PHMForge follows a two-server MCP design reflecting real industrial deployments:

```
┌─────────────────────────────────────────┐
│              Agent (LLM)                │
│  ┌──────────────┐ ┌──────────────────┐  │
│  │ Single Agent │ │ Multi-Agent Root │  │
│  └──────┬───────┘ └───────┬──────────┘  │
│         │                 │             │
│         ▼                 ▼             │
│  ┌──────────────────────────────────┐   │
│  │         MCP Tool Layer           │   │
│  └──────────┬───────────┬───────────┘   │
└─────────────┼───────────┼───────────────┘
              │           │
   ┌──────────▼──┐  ┌─────▼──────────┐
   │ Prognostics │  │  Maintenance   │
   │   Server    │  │    Server      │
   │             │  │                │
   │ - RUL tools │  │ - Cost tools   │
   │ - Fault     │  │ - Safety tools │
   │ - Health    │  │ - Web search   │
   │ - Metrics   │  │                │
   └─────────────┘  └────────────────┘
```

Start MCP servers independently:

```bash
cd ReActXen/src/reactxen/demo/intent_implementation_demo
python mcp_servers/prognostics_server.py    # Port: stdio
python mcp_servers/maintenance_server.py    # Port: stdio
```

---

## Adding New Frameworks

To add a new agent framework to the benchmark:

1. Run your framework against all 75 scenarios in `scenarios/phm_scenarios.json`
2. Collect per-scenario results (task_id, status, accuracy)
3. Add an entry to `results/paper_results.json`:
   ```json
   {
     "framework": "YourFramework",
     "model": "model-name",
     "model_source": "provider",
     "agent_type": "single_agent",
     "scores": {
       "RUL Prediction": { "accuracy": 0.XX, "completed": N, "total": 15 },
       "Fault Classification": { "accuracy": 0.XX, "completed": N, "total": 15 },
       "Engine Health Analysis": { "accuracy": 0.XX, "completed": N, "total": 30 },
       "Cost-Benefit Analysis": { "accuracy": 0.XX, "completed": N, "total": 5 },
       "Safety/Policy Evaluation": { "accuracy": 0.XX, "completed": N, "total": 10 }
     },
     "overall_score": 0.XX
   }
   ```
4. Commit and push -- the Streamlit dashboard auto-updates

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `PHMFORGE_DATA_DIR` | No | Override default dataset directory |
| `WATSONX_APIKEY` | For WatsonX models | IBM WatsonX API key |
| `WATSONX_URL` | For WatsonX models | WatsonX endpoint URL |
| `WATSONX_PROJECT_ID` | For WatsonX models | WatsonX project ID |
| `OPENAI_API_KEY` | For GPT models | OpenAI API key |
| `HF_API_KEY` | For HF models | HuggingFace API token |
| `BRAVE_API_KEY` | Optional | Brave Search API key (web search tool) |

---

## Citation

```bibtex
@inproceedings{phmforge2025,
  title={PHMForge: Intent-Based Industrial Automation Benchmark},
  author={Das, Ayan and others},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```
