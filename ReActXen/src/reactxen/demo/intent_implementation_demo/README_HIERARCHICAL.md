# Hierarchical Agent System for Intent-Based Industrial Automation

## Overview

This implementation provides a hierarchical agent system where a **Root Agent** orchestrates specialized **Sub-Agents** for end-to-end predictive maintenance and industrial automation tasks.

## Architecture

```
Root Agent
├── Data Scientist Agent (ML model training/evaluation)
├── Predictive Maintenance Agent (RUL prediction/risk assessment)
├── Cost-Benefit Analysis Agent (cost estimation/ROI)
└── Safety/Policy Agent (safety protocols/compliance)
```

## File Structure

### Core Components
- **agent_tool_wrapper.py** (90 lines) - Wraps agents as tools for hierarchical structure
- **shared_utils.py** (116 lines) - Common utilities reused across modules
- **ml_models_state.py** (25 lines) - Global state management for trained models
- **ml_data_prep.py** (56 lines) - Common data preparation utilities

### ML Framework Tools (100-158 lines each)
- **sklearn_tools.py** (100 lines) - Scikit-learn model training
- **pytorch_tools.py** (120 lines) - PyTorch model training
- **tensorflow_tools.py** (100 lines) - TensorFlow/Keras model training
- **ml_prediction_tools.py** (158 lines) - Prediction and evaluation tools
- **ml_framework_tools.py** (21 lines) - Factory that combines all ML tools

### Sub-Agents (33-53 lines each)
- **sub_agent_base.py** (48 lines) - Base utilities for creating sub-agents
- **data_scientist_agent.py** (48 lines) - ML model development agent
- **predictive_maintenance_agent.py** (53 lines) - RUL prediction agent
- **cost_benefit_agent.py** (52 lines) - Financial analysis agent
- **safety_policy_agent.py** (33 lines) - Safety/compliance agent
- **hierarchical_agents.py** (53 lines) - Factory for creating sub-agent tools

### Main Implementation
- **root_agent_implementation.py** (192 lines) - Root agent orchestrator
- **main_hierarchical.py** (168 lines) - Main execution script

## Features

### Supported ML Frameworks
- **Scikit-learn**: Random Forest, Linear Regression, SVR, Gradient Boosting
- **PyTorch**: MLP, LSTM, CNN, Transformer architectures
- **TensorFlow/Keras**: Sequential, Functional, LSTM, CNN models
- **WatsonX**: IBM WatsonX API integration
- **HuggingFace**: (if available) HuggingFace model integration

### Sub-Agent Capabilities

1. **Data Scientist Agent**
   - Dataset loading and preprocessing
   - Multi-framework model training
   - Model evaluation and selection
   - Feature engineering

2. **Predictive Maintenance Agent**
   - RUL (Remaining Useful Life) prediction
   - Equipment risk assessment
   - Maintenance scheduling recommendations
   - Failure prediction

3. **Cost-Benefit Analysis Agent**
   - Maintenance cost estimation
   - ROI analysis
   - Budget planning
   - Cost optimization recommendations

4. **Safety/Policy Agent**
   - OSHA compliance protocols
   - Safety procedure recommendations
   - Regulatory compliance checking
   - Risk identification

## Usage

### Basic Usage

```python
from root_agent_implementation import run_root_agent

question = "Which equipment are likely to fail in the next 20 cycles? Provide equipment IDs with safety recommendations and cost estimates."

results = run_root_agent(
    question=question,
    react_llm_model_id=15,
    max_steps=25,
    debug=True
)
```

### Running the Main Script

```bash
python main_hierarchical.py
```

## Code Reusability

The implementation maximizes code reuse through:

1. **Shared Utilities** (`shared_utils.py`)
   - Common environment setup
   - Tool description generation
   - Agent prompt templates
   - Dataset and search tool factories

2. **ML State Management** (`ml_models_state.py`)
   - Centralized model storage
   - Metadata management
   - Model retrieval functions

3. **Data Preparation** (`ml_data_prep.py`)
   - Standardized data preprocessing
   - Feature extraction
   - Data validation

4. **Sub-Agent Base** (`sub_agent_base.py`)
   - Standardized sub-agent creation
   - Common configuration patterns
   - Reusable prompt templates

## Design Principles

- **Modularity**: Each file is 100-200 lines, focused on a single responsibility
- **Reusability**: Common functionality extracted into shared modules
- **Extensibility**: Easy to add new sub-agents or ML frameworks
- **Maintainability**: Clear separation of concerns

## Dependencies

- ReActXen framework
- LangChain (tools and prompts)
- Scikit-learn, PyTorch, TensorFlow (optional, for ML frameworks)
- WatsonX API (for IBM cloud services)
- HuggingFace datasets (optional)

## Notes

- All files are kept under 200 lines for maintainability
- Code is organized for maximum reusability
- Graceful fallbacks for optional dependencies
- Comprehensive error handling

