# PHMForge Benchmark

**Industrial Predictive Maintenance Benchmark for Agentic AI Systems**

75 expert-curated scenarios across 5 categories, evaluated with single-agent and multi-agent architectures using the ReActXen framework.

> Paper: *PHMForge: Intent-Based Industrial Automation Benchmark* (KDD 2025)

## Categories

| Category | Scenarios | Description |
|----------|-----------|-------------|
| RUL Prediction | 15 | Remaining useful life estimation |
| Fault Classification | 15 | Fault detection and diagnosis |
| Engine Health Analysis | 30 | Turbofan engine health assessment |
| Cost-Benefit Analysis | 5 | Maintenance cost optimization |
| Safety/Policy Evaluation | 10 | Safety risk and compliance |

## Architecture

```
intent_implementation_demo/
├── single_agent_implementation/   # Single agent with ALL tools
├── multi_agent_implementation/    # Root agent routes to 5 specialists
│   └── agents/                    # RUL, Fault, Health, Cost, Safety
├── tools/                         # Shared LangChain BaseTool implementations
├── mcp_servers/                   # MCP protocol servers
│   ├── prognostics_server.py      # RUL + Fault + Engine Health tools
│   └── maintenance_server.py      # Cost + Safety tools
├── frontend/                      # Streamlit dashboard
├── results/                       # Benchmark results (JSON)
└── scenarios/                     # 75 PHM scenarios
```

## Setup

```bash
# Install dependencies
pip install -e .
# or
uv sync

# Download datasets
python data/download_datasets.py
```

## Usage

### Run Single-Agent Benchmark

```bash
python single_agent_implementation/run.py --limit 5
python single_agent_implementation/run.py --model-id 8 --model-source watsonx
```

### Run Multi-Agent Benchmark

```bash
python multi_agent_implementation/run.py --limit 5
python multi_agent_implementation/run.py --model-id 8 --model-source watsonx
```

### Launch Dashboard

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

### MCP Servers

```bash
python mcp_servers/prognostics_server.py    # RUL, Fault, Engine Health
python mcp_servers/maintenance_server.py    # Cost-Benefit, Safety
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--scenario-file` | Path to scenario JSON | `scenarios/phm_scenarios.json` |
| `--model-id` | Model index or ID | `8` |
| `--model-source` | `watsonx` or `huggingface` | `watsonx` |
| `--limit` | Max scenarios to run | all |
| `--output-dir` | Results output directory | `results/` |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PHMFORGE_DATA_DIR` | Override default dataset directory |
| `WATSONX_APIKEY` | IBM WatsonX API key |
| `WATSONX_URL` | WatsonX endpoint URL |
| `WATSONX_PROJECT_ID` | WatsonX project ID |
| `HF_API_KEY` | HuggingFace API token |
| `BRAVE_API_KEY` | Brave Search API key |
