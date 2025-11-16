# Agent-Based Industrial Automation Demo with HuggingFace Datasets

This implementation provides a complete agentic workflow for predictive maintenance and RUL (Remaining Useful Life) prediction using HuggingFace PDMBench datasets.

## Features

- **Dataset Management**: Download and store HuggingFace datasets locally
- **Data Loading**: Load datasets from local storage with automatic split handling
- **Agentic Decision Making**: ReActXen-based agent with tool orchestration
- **RUL Prediction**: Train and evaluate models for remaining useful life prediction
- **Risk Assessment**: Identify equipment at risk with RUL ≤ 20 cycles
- **Cost Analysis**: Estimate maintenance costs and perform cost-benefit analysis
- **Safety Recommendations**: Integrate OSHA safety protocols via web search

## Files

- **`load_data.py`**: Enhanced with functions to download and save HuggingFace datasets locally
- **`agent_implementation_hf.py`**: Complete agentic implementation script
- **`tools_logic.py`**: Shared tool implementations (WatsonX, HuggingFace, cost analysis)
- **`downloaded_datasets/`**: Directory where datasets are stored locally

## Setup

### 1. Install Dependencies

Using `uv` (recommended - already set up with virtual environment):

```bash
cd ReActXen/src/reactxen/demo/intent_implementation_demo
uv sync  # This will install all dependencies from pyproject.toml
```

Or install manually with uv:

```bash
uv pip install datasets huggingface_hub langchain langchain-ibm langchain-community \
    haversine pandas ibm-watsonx-ai numpy scikit-learn matplotlib xgboost \
    langchain-core transformers sentencepiece torch torchvision torchaudio ddgs pyarrow
```

### 2. Set Environment Variables

The script automatically loads environment variables from `.env` file. Create or update `ReActXen/env/.env`:

```bash
cp ReActXen/env/.env_template ReActXen/env/.env
# Then edit ReActXen/env/.env with your actual values
```

The `.env` file should contain:

```env
WATSONX_APIKEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com/
BRAVE_API_KEY=your_brave_api_key  # Optional but recommended
HF_BEARER_TOKEN=your_huggingface_token  # Optional for private datasets
```

Alternatively, you can export them in your shell:

```bash
export WATSONX_APIKEY="your_watsonx_api_key"
export WATSONX_PROJECT_ID="your_project_id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com/"
```

### 3. Download Datasets

The implementation includes support for the following PDMBench datasets:

- `submission096/XJTU`
- `submission096/MAFAULDA`
- `submission096/Padeborn`
- `submission096/IMS`
- `submission096/UoC`
- `submission096/RotorBrokenBar`
- `submission096/MFPT`
- `submission096/HUST`
- `submission096/FEMTO`
- `submission096/Mendeley`
- `submission096/ElectricMotorVibrations`
- `submission096/CWRU`
- `submission096/Azure`
- `submission096/PlanetaryPdM`

Datasets will be automatically downloaded when you run the agent, or you can download them manually:

```python
from load_data import download_all_datasets, download_and_save_dataset

# Download all datasets (takes time)
download_results = download_all_datasets()

# Or download specific datasets
download_and_save_dataset("submission096/PlanetaryPdM")
download_and_save_dataset("submission096/CWRU")
```

## Usage

### Running the Agent Script

```bash
cd ReActXen/src/reactxen/demo/intent_implementation_demo
uv run python agent_implementation_hf.py
```

Or if your virtual environment is already activated:

```bash
python agent_implementation_hf.py
```

### Programmatic Usage

```python
from load_data import (
    download_and_save_dataset,
    load_saved_dataset,
    list_available_datasets,
    get_dataset_info
)

# Download a dataset
result = download_and_save_dataset("submission096/PlanetaryPdM")
print(result)

# List available datasets
datasets = list_available_datasets()
print(f"Available datasets: {datasets}")

# Get dataset info
info = get_dataset_info("PlanetaryPdM")
print(info)

# Load a dataset
train_data = load_saved_dataset("PlanetaryPdM", split="train", format="pandas")
print(train_data.head())
```

## Dataset Storage Structure

Datasets are stored in `downloaded_datasets/` with the following structure:

```
downloaded_datasets/
├── PlanetaryPdM/
│   ├── metadata.json          # Dataset metadata
│   ├── train/
│   │   ├── train.csv
│   │   ├── train.parquet
│   │   └── train.json
│   ├── test/
│   │   ├── test.csv
│   │   ├── test.parquet
│   │   └── test.json
│   └── dataset/               # HuggingFace dataset format
├── CWRU/
│   └── ...
└── ...
```

## Agent Workflow

The agent follows this workflow:

1. **List Datasets**: List available datasets using `list_datasets` tool
2. **Load Dataset**: Load a specific dataset using `load_dataset` tool
3. **Initialize WatsonX**: Set up WatsonX API using `initialize_watsonx_api` tool
4. **Get Models**: Retrieve available models using `get_chat_models_list` tool
5. **Set Model**: Select a model using `set_model_id` tool
6. **Train Model**: Train a RUL prediction model using `train_agentic_model` tool
7. **Identify Risk**: Find equipment with RUL ≤ 20 cycles
8. **Safety Search**: Search for OSHA safety protocols using `smart_search` tool
9. **Cost Estimation**: Estimate maintenance costs using `estimate_maintenance_cost` tool
10. **Cost-Benefit Analysis**: Perform analysis using `cost_benefit_analysis` tool
11. **Format Results**: Present results in table format

## Available Tools

### Dataset Tools

- `list_datasets`: List all available downloaded datasets
- `load_dataset`: Load a specific dataset for analysis
- `get_dataset_info`: Get detailed metadata about a dataset

### WatsonX Tools

- `initialize_watsonx_api`: Initialize WatsonX API connection
- `get_chat_models_list`: Get available chat models
- `set_model_id`: Set the model ID for inference
- `get_model_details`: Get details about a specific model

### HuggingFace Tools

- `retrieve_ml_models`: Retrieve ML models from HuggingFace Hub
- `select_optimal_model`: Select optimal model based on criteria
- `train_agentic_model`: Train a model using agentic approach

### Analysis Tools

- `estimate_maintenance_cost`: Estimate maintenance costs
- `cost_benefit_analysis`: Perform cost-benefit analysis
- `smart_search`: Web search with fallback (Brave → DuckDuckGo)

## Output Files

After running the agent, the following files will be generated:

- `agent_trajectory_hf_datasets.json`: Complete agent execution trajectory
- `agent_metrics_hf_datasets.json`: Performance metrics
- `agent_summary_hf_datasets.json`: Experiment summary

## Troubleshooting

### Dataset Download Issues

- **Connection Errors**: Check your internet connection and HuggingFace Hub access
- **Storage Issues**: Ensure sufficient disk space (datasets can be large)
- **Authentication**: Set `HF_BEARER_TOKEN` for private datasets

### Agent Execution Issues

- **InvalidCredentialsError**: Ensure WatsonX credentials are correctly set and not overridden
- **Tool Errors**: Check that all required environment variables are set
- **Search Tool Failures**: The agent automatically falls back to DuckDuckGo if Brave Search fails

### Data Loading Issues

- **File Not Found**: Ensure datasets are downloaded before loading
- **Format Errors**: Datasets are saved in multiple formats (CSV, Parquet, JSON) for compatibility
- **Missing Splits**: Some datasets may not have explicit train/test splits

## Differences from Intent-Implementation-Demo

This implementation differs from `intent-implementation-demo.ipynb` in the following ways:

1. **Data Source**: Uses HuggingFace datasets instead of CMAPSS local files
2. **Dataset Management**: Includes functions to download and manage datasets locally
3. **Flexible Dataset Loading**: Supports multiple datasets with automatic split handling
4. **Tool Integration**: Adds dataset-specific tools (`list_datasets`, `load_dataset`, `get_dataset_info`)

## Next Steps

- Customize the agent prompt for your specific use case
- Add additional datasets to `data_url_link` in `load_data.py`
- Implement custom preprocessing functions for specific datasets
- Add visualization tools for RUL predictions
- Integrate with real-time data streams

## License

See the main project LICENSE file.
