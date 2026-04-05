# PHMForge Agent Specifications

## Overview

PHMForge supports both single-agent and multi-agent configurations for predictive maintenance benchmarking. All agents use the ReActXen framework (ReAct with Reflection) and interact with tools exclusively through MCP (Model Context Protocol) servers.

## Architecture

```
User Intent
    │
    ▼
┌─────────────┐     ┌─────────────────────────┐
│  Root Agent  │────▶│  MCP Client (Router)    │
│  (ReActXen)  │     │  - Tool Discovery       │
└──────┬───────┘     │  - Call Routing          │
       │             │  - Context Management    │
       ▼             └──────────┬──────────────┘
┌─────────────┐                 │
│ Sub-Agents  │         ┌───────┴───────┐
│ (optional)  │         │               │
└─────────────┘    Prognostics     Maintenance
                   Server (15)     Server (7)
```

## Agent Types

### Single-Agent Configuration
- **File**: `single_agent_implementation/agent.py`
- **Description**: One ReActXen agent handles all 75 scenarios using all 22 tools
- **LLM**: Configurable (WatsonX or HuggingFace models)
- **Best for**: Simple scenarios, baseline benchmarking

### Multi-Agent Configuration
- **File**: `multi_agent_implementation/agents/`
- **Description**: Root agent routes to specialized sub-agents based on scenario type
- **Sub-agents**:
  - **RUL Agent** (`rul_agent.py`): Handles RUL prediction scenarios using data loading, model training, prediction, and metric tools
  - **Fault Agent** (`fault_agent.py`): Handles fault classification scenarios using training, classification, and verification tools
  - **Health Agent** (`health_agent.py`): Handles engine health analysis using signal analysis, component assessment, and trend detection tools
  - **Cost Agent** (`cost_agent.py`): Handles cost-benefit analysis using maintenance cost, failure cost, and schedule optimization tools
  - **Safety Agent** (`safety_agent.py`): Handles safety/policy evaluation using risk assessment, compliance checking, and recommendation tools

### External Agent Configurations (Benchmarked)
- **Claude Code**: Anthropic's Claude as coding agent (externally benchmarked)
- **Cursor Agent**: Cursor IDE's AI agent (externally benchmarked)

## Agent Capabilities

| Capability | Single-Agent | Multi-Agent | Claude Code | Cursor |
|------------|:---:|:---:|:---:|:---:|
| Tool Discovery (MCP) | Yes | Yes | N/A | N/A |
| Multi-step Reasoning | Yes | Yes | Yes | Yes |
| Reflection/Self-correction | Yes | Yes | Limited | Limited |
| Task Routing | N/A | Yes | N/A | N/A |
| Context Management | Yes | Yes | Built-in | Built-in |
| Streaming Progress | Yes | Yes | N/A | N/A |

## MCP Integration

### Servers
- **Prognostics Server** (`prognostics_server.py`): 15 tools across data loading, model training, prediction, metrics, and engine health analysis
- **Maintenance Server** (`maintenance_server.py`): 7 tools across cost-benefit analysis, safety/policy evaluation, and web search

### Client
- **MCP Client** (`mcp_client.py`): Connects to both servers via stdio, discovers tools dynamically, routes calls automatically

### Protocol Features
- Tool discovery via `list_tools()`
- Typed responses via Pydantic models
- Configurable transport (stdio for benchmarking, SSE for live deployment)
- Context management to prevent window bloating
- Call metrics for MCP-specific evaluation

## Scenario Coverage

| Scenario Type | Count | Primary Agent | Key Tools |
|---------------|:---:|---|---|
| RUL Prediction | 15 | RUL Agent | load_dataset, train_rul_model, predict_rul, calculate_mae/rmse |
| Fault Classification | 15 | Fault Agent | load_dataset, train_fault_classifier, classify_faults, calculate_accuracy |
| Engine Health Analysis | 30 | Health Agent | analyze_engine_signals, assess_component_health, diagnose_timing_issues |
| Cost-Benefit Analysis | 5 | Cost Agent | calculate_maintenance_cost, calculate_failure_cost, optimize_maintenance_schedule |
| Safety/Policy Evaluation | 10 | Safety Agent | assess_safety_risk, check_compliance, generate_safety_recommendations |

**Total**: 75 scenarios across 18 datasets using 22 MCP tools
