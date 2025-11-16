# Intent-Based Industrial Automation with ReActXen

This repository implements an agentic framework for predictive maintenance and industrial automation using **ReActXen** (ReAct eXtended), a language agent framework that combines reasoning and acting capabilities for complex industrial IoT data analysis tasks.

## What is ReActXen?

**ReActXen** (ReAct eXtended) is an enhanced ReAct (Reasoning + Acting) framework designed for industrial applications. It extends the classic ReAct paradigm with:

- **Multi-Agent Architecture**: Supports multiple helper agents (assessment, reviewer, etc.) working together
- **Dual Action Styles**: Supports both text-based and code-based action execution
- **Tool Orchestration**: Seamless integration with external tools (WatsonX, HuggingFace, web search)
- **Reflection Capabilities**: Built-in reflection and review mechanisms for improved decision-making
- **Context Management**: Advanced context handling for long-running conversations and data analysis

The framework has been published at **EMNLP 2025 Industry Track** for "ReAct Meets Industrial IoT: Language Agents for Data Access".

**Note**: ReActXen is currently not available as a pip package, so it is included as a submodule within this repository. The implementation code is located in `ReActXen/src/reactxen/demo/intent_implementation_demo/`.

## Project Overview

This implementation demonstrates ReActXen's capabilities through:

- **Predictive Maintenance**: RUL (Remaining Useful Life) prediction using PDMBench datasets
- **Dataset Management**: Automated download and management of HuggingFace datasets
- **Agentic Decision Making**: Autonomous agents that can load data, train models, and make recommendations
- **Cost-Benefit Analysis**: Integration of maintenance cost estimation and safety protocol recommendations

## Prerequisites

- **Python**: 3.10 or higher (3.12+ recommended)
- **Git**: For cloning the repository
- **Credentials**: WatsonX API access (see [Credentials Setup](#credentials-setup))

## Setup Instructions

### Option 1: Using `uv` (Recommended)

`uv` is a fast Python package installer and resolver. It's particularly efficient for managing dependencies.

#### Step 1: Install `uv`

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

#### Step 2: Clone the Repository

```bash
git clone <repository-url>
cd Intent-Based-Industrial-Automation
```

#### Step 3: Navigate to Demo Directory

```bash
cd ReActXen/src/reactxen/demo/intent_implementation_demo
```

#### Step 4: Create Virtual Environment and Install Dependencies

```bash
# uv will automatically create a virtual environment and install dependencies
uv sync
```

This command will:

- Create a virtual environment (if it doesn't exist)
- Install all dependencies from `pyproject.toml`
- Activate the virtual environment

#### Step 5: Activate the Virtual Environment

```bash
# The virtual environment is typically located at .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

#### Step 6: Verify Installation

```bash
python -c "import reactxen; print('ReActXen installed successfully')"
```

### Option 2: Using `pip` with Virtual Environment

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Intent-Based-Industrial-Automation
```

#### Step 2: Navigate to Demo Directory

```bash
cd ReActXen/src/reactxen/demo/intent_implementation_demo
```

#### Step 3: Create Virtual Environment

```bash
# Using Python's built-in venv
python3.10 -m venv venv  # or python3.12 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

#### Step 4: Install Dependencies

```bash
# Install from pyproject.toml (requires pip 24.0+)
pip install -e .

# Or install dependencies manually
pip install datasets>=4.4.1 \
    huggingface-hub>=0.36.0 \
    ibm-watsonx-ai>=1.3.42 \
    langchain>=1.0.5 \
    langchain-community>=0.4.1 \
    langchain-core>=1.0.4 \
    langchain-ibm>=1.0.0 \
    pandas>=2.2.3 \
    numpy>=2.2.6 \
    scikit-learn>=1.7.2 \
    torch>=2.9.0 \
    transformers>=4.57.1 \
    xgboost>=3.1.1 \
    python-dotenv>=1.2.1 \
    ddgs>=9.9.0 \
    haversine>=2.9.0 \
    matplotlib>=3.10.7 \
    sentencepiece>=0.2.1
```

#### Step 5: Install ReActXen Package

```bash
# Navigate to ReActXen root directory
cd ../../../../

# Install ReActXen package
pip install -e .
```

## Credentials Setup

### Required Credentials

The following credentials are required for the agent to function:

1. **WatsonX API Credentials** (Required)

   - `WATSONX_APIKEY`: Your WatsonX API key
   - `WATSONX_URL`: WatsonX service URL (e.g., `https://us-south.ml.cloud.ibm.com/`)
   - `WATSONX_PROJECT_ID`: Your WatsonX project ID

2. **Optional Credentials**
   - `OPENAI_API_KEY`: For OpenAI API access (if using OpenAI models)
   - `HF_APIKEY` or `HF_BEARER_TOKEN`: For accessing HuggingFace datasets and models
   - `BRAVE_API_KEY`: For enhanced web search capabilities (falls back to DuckDuckGo if not provided)

### Setting Up Credentials

The application supports two methods for providing credentials, with automatic fallback:

1. **Primary Method**: Environment variables (via `.env` file or system environment)
2. **Fallback Method**: `credentials.json` file (used if environment variables are not detected or fail)

#### Method 1: Using `.env` File (Recommended)

1. Navigate to the demo directory:

```bash
cd ReActXen/src/reactxen/demo/intent_implementation_demo
```

2. Copy the environment template:

```bash
# From demo directory
cp ../../env/.env_template .env

# Or from project root
cp ReActXen/env/.env_template ReActXen/src/reactxen/demo/intent_implementation_demo/.env
```

3. Edit the `.env` file with your actual credentials:

```env
# WatsonX configuration (Required)
WATSONX_APIKEY="your_actual_watsonx_api_key"
WATSONX_URL="https://us-south.ml.cloud.ibm.com/"
WATSONX_PROJECT_ID="your_actual_project_id"

# OpenAI configuration (Optional)
OPENAI_API_KEY="your_openai_api_key"

# HuggingFace configuration (Optional)
HF_APIKEY="your_huggingface_api_key"
HF_BEARER_TOKEN="your_huggingface_token"

# Brave Search API (Optional)
BRAVE_API_KEY="your_brave_api_key"
```

4. The application will automatically load these variables using `python-dotenv`.

#### Method 2: Using `credentials.json` (Fallback)

If environment variables are not detected or fail to authenticate, the application will automatically fall back to `credentials.json`. This is useful when:

- Environment variables are not properly loaded
- Running in environments where `.env` files are not accessible
- Need a local credential file for testing

1. Navigate to the demo directory:

```bash
cd ReActXen/src/reactxen/demo/intent_implementation_demo
```

2. Copy the credentials template:

```bash
cp credentials.json.template credentials.json
```

3. Edit `credentials.json` with your actual credentials:

```json
{
  "watsonx_apikey": "your_actual_watsonx_api_key",
  "watsonx_url": "https://us-south.ml.cloud.ibm.com/",
  "watsonx_project_id": "your_actual_project_id",
  "openai_api_key": "your_openai_api_key",
  "hf_api_key": "your_huggingface_api_key",
  "brave_api_key": "your_brave_api_key"
}
```

**Important**: The `credentials.json` file is **not tracked by version control** (it's in `.gitignore`). Never commit your actual credentials to the repository.

#### Method 3: Export Environment Variables

```bash
export WATSONX_APIKEY="your_actual_watsonx_api_key"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com/"
export WATSONX_PROJECT_ID="your_actual_project_id"
export OPENAI_API_KEY="your_openai_api_key"  # Optional
export HF_APIKEY="your_huggingface_api_key"  # Optional
export BRAVE_API_KEY="your_brave_api_key"  # Optional
```

### Obtaining Credentials

#### WatsonX Credentials

1. **IBM Cloud Account**: Sign up at [IBM Cloud](https://cloud.ibm.com/)
2. **WatsonX Service**: Navigate to WatsonX in IBM Cloud catalog
3. **Create Project**: Create a new project in WatsonX
4. **API Key**: Generate an API key from IBM Cloud IAM
5. **Project ID**: Found in your WatsonX project settings
6. **Service URL**: Region-specific URL (e.g., `https://us-south.ml.cloud.ibm.com/`)

#### Brave Search API (Optional)

1. Sign up at [Brave Search API](https://brave.com/search/api/)
2. Generate an API key
3. Add to your `.env` file

#### HuggingFace Token (Optional)

1. Sign up at [HuggingFace](https://huggingface.co/)
2. Generate a token from Settings → Access Tokens
3. Add to your `.env` file if accessing private datasets

## How the Setup Process Works

### 1. Virtual Environment Isolation

Both `uv` and `pip` create isolated Python environments that:

- Prevent dependency conflicts with system Python
- Allow project-specific package versions
- Enable clean uninstallation

### 2. Dependency Resolution

- **`uv`**: Uses a fast Rust-based resolver for quick dependency resolution
- **`pip`**: Uses Python-based resolver (slower but more compatible)

### 3. Package Installation

The setup installs:

- **ReActXen Core**: The main agent framework
- **LangChain**: For LLM integration and tool orchestration
- **WatsonX SDK**: For IBM WatsonX API access
- **HuggingFace Libraries**: For dataset management and model access
- **ML Libraries**: PyTorch, scikit-learn, XGBoost for model training
- **Data Processing**: Pandas, NumPy for data manipulation

### 4. Environment Variable Loading

The application uses `python-dotenv` to automatically load credentials from:

1. `.env` file in the current directory (`ReActXen/src/reactxen/demo/intent_implementation_demo/`)
2. `.env` file in the project root
3. System environment variables (takes precedence)
4. **Fallback**: `credentials.json` file in the demo directory (if environment variables fail or are not detected)

## Running the Agent

### Basic Usage

```bash
# From the demo directory
cd ReActXen/src/reactxen/demo/intent_implementation_demo

# Using uv
uv run python agent_implementation_hf.py

# Or with activated virtual environment
python agent_implementation_hf.py
```

### Running Benchmarks

```bash
python benchmark.py
```

### Running Individual Tests

```bash
python run_agent_test.py
```

## Replicating the Logic

### Understanding the Architecture

1. **Agent Implementation** (`agent_implementation_hf.py`):

   - Main entry point for the agentic workflow
   - Defines tools, prompts, and agent configuration
   - Orchestrates the complete decision-making process

2. **Tool Logic** (`tools_logic.py`):

   - Implements all available tools (WatsonX, HuggingFace, cost analysis)
   - Provides the interface between agent and external services
   - Handles data processing and model training

3. **Data Loading** (`load_data.py`):

   - Manages dataset download from HuggingFace
   - Handles local dataset storage and retrieval
   - Provides dataset metadata and information

4. **Benchmarking** (`benchmark.py`):
   - Evaluates agent performance
   - Generates metrics and reports
   - Compares different configurations

### Key Components to Customize

1. **Agent Prompt**: Modify prompts in `agent_implementation_hf.py` to change agent behavior
2. **Tool Definitions**: Add or modify tools in `tools_logic.py`
3. **Dataset Sources**: Update dataset URLs in `load_data.py`
4. **Model Selection**: Change model IDs in agent configuration

### Workflow Overview

```
User Query
    ↓
Agent Initialization (ReActXen)
    ↓
Tool Selection & Execution
    ├── Dataset Loading
    ├── Model Training
    ├── Cost Analysis
    └── Safety Search
    ↓
Reflection & Review
    ↓
Final Answer Generation
```

## Project Structure

```
Intent-Based-Industrial-Automation/
├── ReActXen/                          # ReActXen framework (not available as pip package)
│   ├── src/reactxen/
│   │   ├── agents/                    # Agent implementations
│   │   ├── demo/
│   │   │   └── intent_implementation_demo/  # Your implementation code
│   │   │       ├── agent_implementation_hf.py
│   │   │       ├── tools_logic.py
│   │   │       ├── load_data.py
│   │   │       ├── benchmark.py
│   │   │       ├── credentials.json.template  # Template for credentials.json
│   │   │       ├── .env                      # Your local .env file (not tracked)
│   │   │       ├── credentials.json         # Your local credentials (not tracked)
│   │   │       └── downloaded_datasets/     # Local dataset storage
│   │   └── utils/                     # Utility functions
│   ├── env/
│   │   └── .env_template             # Environment variable template
│   └── pyproject.toml                # ReActXen package dependencies
└── README.md                         # This file
```

## Troubleshooting

### Common Issues

1. **Import Errors**:

   ```bash
   # Ensure ReActXen is installed
   pip install -e ReActXen/
   ```

2. **Credential Errors**:

   - Verify `.env` file exists in `ReActXen/src/reactxen/demo/intent_implementation_demo/` and contains correct values
   - Check that environment variables are exported correctly
   - Ensure WatsonX project ID is valid
   - If environment variables fail, create `credentials.json` in the demo directory as a fallback:
     ```bash
     cd ReActXen/src/reactxen/demo/intent_implementation_demo
     cp credentials.json.template credentials.json
     # Then edit credentials.json with your actual values
     ```

3. **Dataset Download Failures**:

   - Check internet connection
   - Verify HuggingFace token if accessing private datasets
   - Ensure sufficient disk space (datasets can be large)

4. **Virtual Environment Issues**:
   ```bash
   # Recreate virtual environment
   rm -rf venv .venv
   uv sync  # or pip install -r requirements.txt
   ```

### Getting Help

- Check the [ReActXen README](ReActXen/README.md) for framework-specific documentation
- Review [README_AGENT_HF.md](ReActXen/src/reactxen/demo/intent_implementation_demo/README_AGENT_HF.md) for agent implementation details
- Consult the [EMNLP 2025 paper](https://openreview.net/forum?id=luETrQw0j6) for theoretical background

## Next Steps

1. **Customize the Agent**: Modify prompts and tools for your specific use case
2. **Add Datasets**: Include additional PDMBench datasets or your own data
3. **Extend Tools**: Add new tools for domain-specific operations
4. **Optimize Performance**: Tune model parameters and agent configuration
5. **Deploy**: Integrate the agent into production systems

## Citation

If you use ReActXen in your research, please cite:

```bibtex
@inproceedings{patel2025react,
  title     = {ReAct Meets Industrial IoT: Language Agents for Data Access},
  author    = {James T. Rayfield and Shuxin Lin and Nianjun Zhou and Dhaval C. Patel},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  year      = {2025},
  url       = {https://openreview.net/forum?id=luETrQw0j6}
}
```

## License

See the project LICENSE file for details.
