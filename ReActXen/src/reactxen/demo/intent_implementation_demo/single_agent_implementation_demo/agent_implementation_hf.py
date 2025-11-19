"""
Agent-Based Industrial Automation Demo with HuggingFace Datasets

This script demonstrates a ReActXen-based agent for predictive maintenance and RUL 
(Remaining Useful Life) prediction using HuggingFace PDMBench datasets.

Features:
- Download and store HuggingFace datasets locally
- Data loading and preprocessing from downloaded datasets
- Multiple ML models (Random Forest, Linear Regression, SVR)
- RUL prediction and evaluation
- Engine risk assessment
- Agentic decision making with ReActXen framework
"""

import os
import sys
import traceback
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from io import StringIO

# Load environment variables from .env file
# Try multiple possible .env file locations
# __file__ is: ReActXen/src/reactxen/demo/intent_implementation_demo/agent_implementation_hf.py
# So we need to go up: demo -> reactxen -> src -> ReActXen -> (workspace root)
env_file_paths = [
    Path(__file__).parent.parent.parent.parent.parent.parent / ".env",  # Workspace root .env
    Path(__file__).parent.parent.parent.parent.parent / "env" / ".env",  # ReActXen/env/.env
    Path(__file__).parent.parent.parent.parent / "env" / ".env",  # Alternative path
    Path(__file__).parent / ".env",  # Local .env
    Path(__file__).parent.parent.parent.parent.parent / ".env",  # ReActXen/.env
]

env_loaded = False
for env_path in env_file_paths:
    if env_path.exists():
        print(f"📁 Loading environment variables from: {env_path}")
        load_dotenv(env_path, override=False)
        env_loaded = True
        break

if not env_loaded:
    # Fallback to default .env loading
    load_dotenv(override=False)
    print("⚠️  No .env file found in expected locations. Using environment variables.")

# Add source path for reactxen imports
# __file__ is: ReActXen/src/reactxen/demo/intent_implementation_demo/agent_implementation_hf.py
# Need to add ReActXen/src to path for reactxen imports
# parent: intent_implementation_demo, parent.parent: demo, parent.parent.parent: reactxen, parent.parent.parent.parent: src
reactxen_src = Path(__file__).parent.parent.parent.parent  # ReActXen/src
if str(reactxen_src) not in sys.path:
    sys.path.insert(0, str(reactxen_src))

# Add current directory to path
demo_dir = Path(__file__).parent
if str(demo_dir) not in sys.path:
    sys.path.insert(0, str(demo_dir))

# Import load data functions
from load_data import (
    data_url_link,
    download_and_save_dataset,
    download_all_datasets,
    load_saved_dataset,
    list_available_datasets,
    get_dataset_info,
    DATA_DIR,
)

# Import tools
from tools_logic import (
    InitializeWatsonXTool,
    GetChatModelsListTool,
    SetModelIDTool,
    GetModelDetailsTool,
    RetrieveMLModelsTool,
    SelectOptimalModelTool,
    TrainAgenticModelTool,
    CostEstimationTool,
    CostBenefitAnalysisTool,
    PredictRULTool,
    GetEnginesAtRiskTool,
    create_watsonx_tools,
    create_huggingface_tools,
    create_cost_benefit_analysis_tools,
    create_rul_prediction_tools,
    WatsonXAPIState,
    HuggingFaceModelState,
    get_reference_train_data,
    get_reference_test_data,
    get_ground_truth,
)

# Import agent creation
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Type

# Import search tools
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun


# Global variables for data
train_data = None
test_data = None
ground_truth = None
trained_models = {}
scalers = {}
selected_dataset_name = None


def load_dataset_for_analysis(dataset_name: str = "PlanetaryPdM", split: str = "train"):
    """
    Load a specific dataset and prepare it for RUL prediction analysis.
    
    Args:
        dataset_name: Name of the dataset to load (default: "PlanetaryPdM")
        split: Which split to load (default: "train")
    
    Returns:
        DataFrame with loaded data or error message
    """
    global train_data, test_data, ground_truth, selected_dataset_name
    
    try:
        print(f"\n📥 Loading dataset: {dataset_name}")
        print("="*70)
        
        # Get dataset info first
        info = get_dataset_info(dataset_name)
        if 'error' in info:
            return f"Error: {info['error']}"
        
        selected_dataset_name = dataset_name
        
        # Try to load as pandas DataFrame
        try:
            # Check if dataset has splits
            if 'splits' in info and info['splits']:
                splits = info['splits']
                print(f"Available splits: {', '.join(splits)}")
                
                # Load train split
                if 'train' in splits:
                    train_data = load_saved_dataset(dataset_name, split='train', format='pandas')
                    print(f"✅ Loaded train split: {train_data.shape}")
                elif splits:
                    # Use first available split as train
                    train_data = load_saved_dataset(dataset_name, split=splits[0], format='pandas')
                    print(f"✅ Loaded {splits[0]} split as train: {train_data.shape}")
                
                # Load test split if available
                if 'test' in splits:
                    test_data = load_saved_dataset(dataset_name, split='test', format='pandas')
                    print(f"✅ Loaded test split: {test_data.shape}")
                
                # Load validation split if available (use as test if no test split)
                if 'validation' in splits and test_data is None:
                    test_data = load_saved_dataset(dataset_name, split='validation', format='pandas')
                    print(f"✅ Loaded validation split as test: {test_data.shape}")
                
                # Try to extract ground truth if RUL column exists
                if test_data is not None:
                    if 'RUL' in test_data.columns:
                        ground_truth = test_data[['RUL']].copy()
                        if 'unit' not in ground_truth.columns and 'unit' in test_data.columns:
                            ground_truth['unit'] = test_data['unit']
                        elif 'unit' not in ground_truth.columns:
                            ground_truth['unit'] = range(1, len(ground_truth) + 1)
                        print(f"✅ Extracted ground truth: {len(ground_truth)} samples")
            else:
                # Single dataset without splits
                train_data = load_saved_dataset(dataset_name, format='pandas')
                print(f"✅ Loaded dataset: {train_data.shape}")
                
                # Split into train/test if no explicit splits
                if len(train_data) > 100:
                    split_idx = int(len(train_data) * 0.8)
                    test_data = train_data.iloc[split_idx:].copy()
                    train_data = train_data.iloc[:split_idx].copy()
                    print(f"✅ Split dataset: Train={len(train_data)}, Test={len(test_data)}")
                    
                    # Extract ground truth if RUL exists
                    if 'RUL' in test_data.columns:
                        ground_truth = test_data[['RUL']].copy()
                        ground_truth['unit'] = range(1, len(ground_truth) + 1)
                        print(f"✅ Extracted ground truth: {len(ground_truth)} samples")
        
        except Exception as e:
            print(f"⚠️  Warning: Could not load as pandas DataFrame: {e}")
            print("Trying to load as HuggingFace dataset...")
            
            # Try loading as HuggingFace dataset
            dataset = load_saved_dataset(dataset_name, format='dataset')
            
            if isinstance(dataset, dict) and 'train' in dataset:
                train_data = dataset['train'].to_pandas()
            if isinstance(dataset, dict) and 'test' in dataset:
                test_data = dataset['test'].to_pandas()
            elif hasattr(dataset, 'to_pandas'):
                train_data = dataset.to_pandas()
        
        # IMPORTANT: Set global variables in tools_logic module so tools can access them
        import tools_logic as tl
        tl.train_data = train_data
        tl.test_data = test_data
        tl.ground_truth = ground_truth
        
        # Prepare summary
        summary = f"\n✅ Successfully loaded dataset: {dataset_name}\n"
        
        if train_data is not None:
            summary += f"\n📊 Train Data:"
            summary += f"\n   Shape: {train_data.shape}"
            cols = list(train_data.columns)[:10]
            summary += f"\n   Columns: {cols}{'...' if len(train_data.columns) > 10 else ''}"
        
        if test_data is not None:
            summary += f"\n\n📊 Test Data:"
            summary += f"\n   Shape: {test_data.shape}"
            cols = list(test_data.columns)[:10]
            summary += f"\n   Columns: {cols}{'...' if len(test_data.columns) > 10 else ''}"
        
        if ground_truth is not None:
            summary += f"\n\n📊 Ground Truth:"
            summary += f"\n   Samples: {len(ground_truth)}"
            if 'RUL' in ground_truth.columns:
                summary += f"\n   RUL Range: {ground_truth['RUL'].min()} to {ground_truth['RUL'].max()}"
        
        print(summary)
        
        return summary
        
    except Exception as e:
        error_msg = f"Error loading dataset {dataset_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg


# Create dataset-specific tools
class EmptyInput(BaseModel):
    pass


class LoadDatasetInput(BaseModel):
    dataset_name: Optional[str] = Field(
        default=None,
        description="Name of the dataset to load (e.g., 'PlanetaryPdM', 'CWRU'). If not provided, uses default."
    )


class ListDatasetsTool(BaseTool):
    """List all available downloaded datasets."""
    
    name: str = "list_datasets"
    description: str = """List all available datasets in local storage.
    
    Returns a list of dataset names that can be loaded using load_dataset tool.
    """
    args_schema: Type[BaseModel] = EmptyInput
    
    def _run(self) -> str:
        available = list_available_datasets()
        
        if not available:
            return "No datasets found. Please download datasets first using download_and_save_dataset() or download_all_datasets()."
        
        result = f"📦 Available Datasets ({len(available)}):\n\n"
        
        for dataset_name in available:
            info = get_dataset_info(dataset_name)
            if 'error' not in info:
                result += f"  • {dataset_name}\n"
                if 'splits' in info:
                    result += f"    Splits: {', '.join(info['splits'])}\n"
                else:
                    result += f"    Rows: {info.get('num_rows', 'N/A')}\n"
                result += f"    Size: {info.get('total_size_mb', 0)} MB\n"
        
        result += "\n\nUse load_dataset tool with dataset_name parameter to load a specific dataset."
        
        return result


class LoadDatasetTool(BaseTool):
    """Load a HuggingFace dataset from local storage."""
    
    name: str = "load_dataset"
    description: str = """Load a HuggingFace dataset from local storage.
    
    This tool loads a previously downloaded dataset from the downloaded_datasets directory.
    The dataset will be prepared for RUL prediction analysis.
    
    Use list_datasets first to see available datasets.
    """
    args_schema: Type[BaseModel] = LoadDatasetInput
    
    def _run(self, dataset_name: Optional[str] = None) -> str:
        # Clean the dataset_name to remove any "Observation" text or JSON artifacts
        if dataset_name:
            # Remove "Observation" text if present
            dataset_name = dataset_name.replace("Observation", "").strip()
            # Try to extract JSON if it's wrapped in JSON
            import json
            import re
            # Check if it's a JSON string
            json_match = re.search(r'\{[^}]*"dataset_name"[^}]*\}', dataset_name)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    dataset_name = parsed.get("dataset_name", dataset_name)
                except:
                    pass
            # Clean any remaining quotes or braces
            dataset_name = dataset_name.strip('"').strip("'").strip('{').strip('}').strip()
        
        if dataset_name is None or not dataset_name:
            available = list_available_datasets()
            if available:
                dataset_name = available[0]
                return f"No dataset name provided. Using first available: {dataset_name}. {load_dataset_for_analysis(dataset_name)}"
            else:
                return "Error: No dataset name provided and no datasets available. Please download datasets first."
        
        return load_dataset_for_analysis(dataset_name)


class GetDatasetInfoTool(BaseTool):
    """Get detailed information about a dataset."""
    
    name: str = "get_dataset_info"
    description: str = """Get detailed information about a specific dataset.
    
    Returns metadata including splits, columns, size, and other information.
    """
    args_schema: Type[BaseModel] = LoadDatasetInput
    
    def _run(self, dataset_name: Optional[str] = None) -> str:
        if dataset_name is None:
            return "Error: dataset_name is required. Use list_datasets to see available datasets."
        
        info = get_dataset_info(dataset_name)
        
        if 'error' in info:
            return str(info['error'])
        
        import json
        return json.dumps(info, indent=2)


def create_dataset_tools() -> List[BaseTool]:
    """Create and return dataset-related tools."""
    return [
        ListDatasetsTool(),
        LoadDatasetTool(),
        GetDatasetInfoTool(),
    ]


# Search tool wrapper
class SearchToolWrapper(BaseTool):
    """Wrapper that provides fallback between Brave Search and DuckDuckGo"""

    name: str = "smart_search"
    description: str = """Search the web for information. Automatically falls back to DuckDuckGo if Brave Search fails.
    
    Use this instead of brave_search or duckduckgo_search directly.
    Input: JSON with "query" parameter containing your search query.
    """
    args_schema: Type[BaseModel] = LoadDatasetInput

    def _run(self, query: str) -> str:
        """Try Brave Search first, fall back to DuckDuckGo"""
        global brave_search_tool, duckduckgo_search_tool
        
        # Clean the query to remove any "Observation" text or JSON artifacts
        if query:
            import json
            import re
            # Remove "Observation" text if present
            query = query.replace("Observation", "").strip()
            # Try to extract JSON if it's wrapped in JSON
            json_match = re.search(r'\{[^}]*"query"[^}]*\}', query)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    query = parsed.get("query", query)
                except:
                    pass
            # Clean any remaining quotes or braces
            query = query.strip('"').strip("'").strip('{').strip('}').strip()
            # Remove any trailing "Observation" repetitions
            while query.endswith("Observation"):
                query = query[:-10].strip()
        
        if not query:
            return "Error: Empty query provided to search tool."
        
        try:
            if brave_search_tool:
                result = brave_search_tool.run({"query": query})
                return result
        except Exception as e:
            try:
                if duckduckgo_search_tool:
                    result = duckduckgo_search_tool.run(query)
                    return f"Note: Used DuckDuckGo (Brave Search failed): {result}"
            except Exception as e2:
                return f"Both search tools failed. Brave error: {str(e)}, DuckDuckGo error: {str(e2)}"
        return "No search tools available"


def calculate_max_steps(question: str) -> int:
    """
    Dynamically calculate max_steps based on question complexity.
    
    Factors considered:
    - Question length
    - Number of required actions (keywords)
    - Complexity indicators
    """
    base_steps = 10
    
    # Count required actions/keywords
    required_actions = [
        "load", "dataset", "train", "predict", "analyze", "identify",
        "recommend", "estimate", "cost", "safety", "equipment", "engine",
        "rul", "maintenance", "risk", "list", "get", "set", "initialize"
    ]
    
    question_lower = question.lower()
    action_count = sum(1 for action in required_actions if action in question_lower)
    
    # Calculate complexity score
    word_count = len(question.split())
    complexity_score = word_count * 0.1 + action_count * 1.5
    
    # Determine steps based on complexity
    if complexity_score < 5:
        max_steps = base_steps + 5  # 15 steps
    elif complexity_score < 10:
        max_steps = base_steps + 10  # 20 steps
    elif complexity_score < 15:
        max_steps = base_steps + 15  # 25 steps
    else:
        max_steps = base_steps + 20  # 30 steps
    
    # Cap at reasonable maximum
    max_steps = min(max_steps, 30)
    
    return max_steps


def clean_outputs_directory(output_dir: Path):
    """Clean the outputs directory before running agent"""
    if output_dir.exists():
        import shutil
        try:
            # Remove all files in outputs directory
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            print(f"✅ Cleaned outputs directory: {output_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean outputs directory: {str(e)}")
    else:
        output_dir.mkdir(exist_ok=True)
        print(f"✅ Created outputs directory: {output_dir}")


def main():
    """Main function to run the agentic implementation."""
    
    print("="*70)
    print("🚀 AGENT-BASED INDUSTRIAL AUTOMATION DEMO")
    print("   Using HuggingFace PDMBench Datasets")
    print("="*70)
    
    # Step 1: Check and download datasets if needed
    print("\n📦 STEP 1: Checking Available Datasets")
    print("-"*70)
    
    available_datasets = list_available_datasets()
    
    if not available_datasets:
        print("❌ No datasets found. Downloading test datasets...")
        test_datasets = [
            "submission096/PlanetaryPdM",
            "submission096/CWRU",
            "submission096/Azure"
        ]
        
        for dataset_path in test_datasets:
            dataset_name = dataset_path.split("/")[-1]
            print(f"\nDownloading: {dataset_name}")
            result = download_and_save_dataset(dataset_path, force_download=False)
            print(f"Status: {result.get('status', 'unknown')}")
    else:
        print(f"✅ Found {len(available_datasets)} available dataset(s)")
        for name in available_datasets:
            print(f"   • {name}")
    
    # Step 2: Initialize search tools
    print("\n🔍 STEP 2: Initializing Search Tools")
    print("-"*70)
    
    global brave_search_tool, duckduckgo_search_tool
    
    try:
        brave_api_key = os.environ.get('BRAVE_API_KEY', '')
        if brave_api_key:
            brave_search_tool = BraveSearch.from_api_key(api_key=brave_api_key)
            print("✅ BraveSearch tool initialized successfully.")
        else:
            print("⚠️  BRAVE_API_KEY not found. BraveSearch will not be available.")
            brave_search_tool = None
    except Exception as e:
        print(f"⚠️  Error initializing BraveSearch: {str(e)}")
        brave_search_tool = None
    
    try:
        duckduckgo_search_tool = DuckDuckGoSearchRun()
        print("✅ DuckDuckGoSearch tool initialized successfully")
    except Exception as e:
        print(f"⚠️ Error initializing DuckDuckGoSearch: {str(e)}")
        duckduckgo_search_tool = None
    
    # Step 3: Create all tools
    print("\n🛠️  STEP 3: Creating Tools")
    print("-"*70)
    
    all_tools = []
    
    # Add dataset tools
    dataset_tools = create_dataset_tools()
    all_tools.extend(dataset_tools)
    print(f"✅ Added {len(dataset_tools)} dataset tools")
    
    # Add WatsonX tools
    watsonx_tools = create_watsonx_tools()
    all_tools.extend(watsonx_tools)
    print(f"✅ Added {len(watsonx_tools)} WatsonX tools")
    
    # Add HuggingFace tools
    hf_tools = create_huggingface_tools()
    all_tools.extend(hf_tools)
    print(f"✅ Added {len(hf_tools)} HuggingFace tools")
    
    # Add cost analysis tools
    cost_tools = create_cost_benefit_analysis_tools()
    all_tools.extend(cost_tools)
    print(f"✅ Added {len(cost_tools)} cost analysis tools")
    
    # Add RUL prediction tools
    rul_tools = create_rul_prediction_tools()
    all_tools.extend(rul_tools)
    print(f"✅ Added {len(rul_tools)} RUL prediction tools")
    
    # Add search tools
    if brave_search_tool:
        all_tools.append(brave_search_tool)
    if duckduckgo_search_tool:
        all_tools.append(duckduckgo_search_tool)
    all_tools.append(SearchToolWrapper())
    
    print(f"\n📊 Total tools available: {len(all_tools)}")
    
    # Step 4: Generate tool descriptions
    print("\n📝 STEP 4: Generating Tool Descriptions")
    print("-"*70)
    
    tool_desc_parts = []
    for i, tool in enumerate(all_tools):
        try:
            if tool.args_schema is not None:
                # Handle both Pydantic v1 (schema) and v2 (model_json_schema)
                if hasattr(tool.args_schema, 'model_json_schema'):
                    schema = tool.args_schema.model_json_schema()
                else:
                    schema = tool.args_schema.schema()
                props = schema.get('properties', {})
                params = list(props.keys()) if props else []
                tool_desc_parts.append(f"({i+1}) {tool.name}[{', '.join(params)}]: {tool.description}")
            else:
                tool_desc_parts.append(f"({i+1}) {tool.name}[]: {tool.description}")
        except Exception as e:
            # Fallback if schema access fails
            tool_desc_parts.append(f"({i+1}) {tool.name}[]: {tool.description}")
    
    tool_desc = "\n".join(tool_desc_parts)
    
    tool_names = [tool.name for tool in all_tools]
    
    print(f"✅ Generated descriptions for {len(all_tools)} tools")
    
    # Step 5: Create agent prompts
    print("\n💭 STEP 5: Creating Agent Prompts")
    print("-"*70)
    
    # Escape JSON braces for Jinja2 - use {% raw %} blocks or double braces
    agent_prompt = PromptTemplate(
        input_variables=["question", "tool_desc", "tool_names", "scratchpad", "examples"],
        template="""You are an AI Root Agent specialized in Intent-Based Industrial Automation for predictive maintenance and RUL (Remaining Useful Life) prediction using HuggingFace datasets.

## TASK INTERPRETATION:
Your goal is to:
1. List available datasets using list_datasets tool
2. Load a dataset using load_dataset tool
3. Initialize WatsonX API using initialize_watsonx_api tool
4. Get available models using get_chat_models_list tool
5. Train a model to predict RUL
6. Identify engines/equipment with low RUL (≤ 20 cycles) that are at risk
7. Provide safety recommendations and cost estimates

## WORKFLOW (YOU MUST FOLLOW THIS ORDER):
Step 1: List available datasets using list_datasets tool with empty JSON object (no parameters needed - use empty braces)
Step 2: Load a dataset using load_dataset tool with JSON containing dataset_name parameter: {% raw %}{{"dataset_name": "CWRU"}}{% endraw %}
Step 3: Initialize WatsonX API using initialize_watsonx_api tool with empty JSON object (no parameters, uses env vars - use empty braces)
Step 4: Get available models using get_chat_models_list tool with empty JSON object (no parameters - use empty braces)
Step 5: Set model ID using set_model_id tool with JSON containing model_id parameter: {% raw %}{{"model_id": "ibm/granite-3-2-8b-instruct"}}{% endraw %}
Step 6: Train model using train_agentic_model tool with JSON containing model_type and task_description: {% raw %}{{"model_type": "traditional", "task_description": "RUL prediction"}}{% endraw %}
   - For traditional models, use model_type="traditional" (this uses scikit-learn models like random_forest, linear_regression, svr)
   - You do NOT need to retrieve ML models from HuggingFace for traditional approach - the tool handles this automatically
   - IMPORTANT: You MUST train a model before predicting RUL. The train_agentic_model tool will automatically train the model.
Step 7: After training is complete, predict RUL using predict_rul tool with JSON containing model_type: {% raw %}{{"model_type": "random_forest"}}{% endraw %}
   - IMPORTANT: You can only predict RUL AFTER training a model. The predict_rul tool requires a trained model.
   - If you get an error about missing test data, the tool will use training data for prediction.
Step 8: Identify equipment at risk using get_engines_at_risk tool with JSON containing threshold: {% raw %}{{"threshold": 20}}{% endraw %}
Step 9: For equipment at risk, use smart_search to get OSHA safety protocols: {% raw %}{{"query": "OSHA safety protocols"}}{% endraw %}
Step 10: Estimate costs using estimate_maintenance_cost for each piece of equipment
Step 11: Perform cost-benefit analysis using cost_benefit_analysis
Step 12: Format results in a table

## CRITICAL FORMAT RULES:
1. Action: Use ONLY the tool name (e.g., list_datasets, load_dataset, smart_search)
2. Action Input: 
   - For tools with NO parameters: Use empty JSON object with just opening and closing braces (no parameters needed)
   - For tools WITH parameters: Use proper JSON format with parameter names as keys and values
   - Example: {% raw %}{{"dataset_name": "CWRU"}}{% endraw %} or {% raw %}{{"query": "RUL prediction models"}}{% endraw %}
   - CRITICAL: Action Input must be ONLY valid JSON - DO NOT include the word "Observation" anywhere in Action Input
   - CRITICAL: Action Input must end immediately after the closing brace - nothing after it
   - If you see "Observation" in your Action Input, you have made an error - remove it completely
   - Action Input examples:
     * Correct: empty braces for no parameters
     * Correct: {% raw %}{{"dataset_name": "CWRU"}}{% endraw %}
     * WRONG: {% raw %}{{"dataset_name": "CWRU"}}{% endraw %} Observation
     * WRONG: {% raw %}{{"dataset_name": "CWRU"}}{% endraw %} Observation Observation

## SEARCH TOOL USAGE:
- Use smart_search tool which automatically falls back to DuckDuckGo if Brave Search fails
- Search queries should be in JSON format with query parameter: {% raw %}{{"query": "your search query"}}{% endraw %}
- Use search tools to find: OSHA safety protocols, maintenance costs, labor rates

## ERROR HANDLING:
- If a tool fails, try an alternative tool
- If you get an error, analyze it and try a different approach
- Do NOT repeat the same failed action multiple times
- If train_agentic_model fails, DO NOT retry with the same model_type - the tool has automatic fallback (scikit-learn → HuggingFace → WatsonX)
- If you see "All training methods failed", check data format and WatsonX initialization, then try model_type="agentic" directly

## TOOLS AVAILABLE:
{tool_desc}

## FORMAT:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be valid JSON format)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer in table format

## EXAMPLES:
{examples}

Begin!

Question: {question}

{scratchpad}"""
    )
    
    reflect_prompt = PromptTemplate(
        input_variables=["examples", "question", "scratchpad"],
        template="""You are an advanced reasoning agent specializing in predictive maintenance and RUL prediction. You will be given a previous reasoning trial where you attempted to answer a question about industrial equipment maintenance.

Your task is to:
1. Diagnose why the previous attempt may have failed or could be improved
2. Identify any missing steps in the reasoning process
3. Devise a new, concise plan that addresses the shortcomings

Common failure modes to check for:
- Missing dataset loading steps
- Incorrect tool usage or parameter mismatches
- Incomplete cost-benefit analysis
- Failure to use search tools for current information
- Missing model training or prediction steps

Here are some examples:
{examples}

Previous trial:
Question: {question}
{scratchpad}

Reflection:"""
    )
    
    print("✅ Agent prompts created successfully")
    
    # Step 6: Create and run agent
    print("\n🤖 STEP 6: Creating and Running Agent")
    print("-"*70)
    
    # Create output directory and clean it before running
    output_dir = Path(__file__).parent / "outputs"
    clean_outputs_directory(output_dir)
    
    # Clean up old log files in the main directory (not outputs)
    main_dir = Path(__file__).parent
    for log_file in main_dir.glob("*.log"):
        try:
            log_file.unlink()
            print(f"🗑️  Deleted old log file: {log_file.name}")
        except Exception as e:
            print(f"⚠️  Could not delete {log_file.name}: {e}")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"agent_execution_{timestamp}.log"
    step_by_step_file = output_dir / f"agent_steps_{timestamp}.txt"
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Execution log: {output_file}")
    print(f"📋 Step-by-step log: {step_by_step_file}")
    
    # Custom logging class to capture agent output
    class AgentOutputLogger:
        def __init__(self, log_file, step_file):
            self.log_file = log_file
            self.step_file = step_file
            self.step_count = 0
            self.log_file_handle = open(log_file, 'w', encoding='utf-8')
            self.step_file_handle = open(step_file, 'w', encoding='utf-8')
            self.any_success = False  # Track if any run was successful
            
        def log(self, message, step_info=False):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"[{timestamp}] {message}\n"
            
            # Write to both console and log file
            print(message)
            self.log_file_handle.write(formatted_msg)
            self.log_file_handle.flush()
            
            # Write step information separately
            if step_info:
                self.step_count += 1
                self.step_file_handle.write(f"\n{'='*70}\n")
                self.step_file_handle.write(f"STEP {self.step_count}: {timestamp}\n")
                self.step_file_handle.write(f"{'='*70}\n")
                self.step_file_handle.write(f"{message}\n")
                self.step_file_handle.flush()
        
        def mark_success(self):
            """Mark that at least one run was successful"""
            self.any_success = True
        
        def close(self):
            self.log_file_handle.close()
            self.step_file_handle.close()
            
            # Only keep logs if at least one run was successful
            if self.any_success:
                print(f"\n✅ Logs saved to:")
                print(f"   - {self.log_file}")
                print(f"   - {self.step_file}")
            else:
                # Delete log files if no successful runs
                try:
                    if self.log_file.exists():
                        self.log_file.unlink()
                    if self.step_file.exists():
                        self.step_file.unlink()
                    print(f"\n⚠️  No successful runs. Log files deleted.")
                except Exception as e:
                    print(f"\n⚠️  Could not delete log files: {e}")
    
    logger = AgentOutputLogger(output_file, step_by_step_file)
    
    # Define the question
    question = "We should focus on equipment that is running on fumes. Which equipment from the loaded dataset are likely to give out in the next 20 cycles? Provide me a list of equipment IDs with safety recommendations and cost estimates."
    
    # Dynamically calculate max_steps based on question complexity
    max_steps = calculate_max_steps(question)
    logger.log(f"📊 Calculated max_steps: {max_steps} (based on question complexity)")
    
    # Import benchmarking utilities
    from benchmark_utils import ModelBenchmark
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir)
    benchmark.start_benchmark()
    
    # Define models to test
    models_to_test = [
        {"name": "Granite-3.2-8B-Instruct", "type": "watsonx", "model_id": 15},
        # Add more models here for benchmarking
        # {"name": "Granite-3.0-8B-Instruct", "type": "watsonx", "model_id": 14},
        # {"name": "Llama-3-8B", "type": "huggingface", "model_id": "meta-llama/Llama-3-8b"},
    ]
    
    logger.log(f"\n🔬 Starting Benchmark: Testing {len(models_to_test)} model(s)")
    logger.log("-"*70)
    
    best_result = None
    all_results = []
    
    for model_config in models_to_test:
        model_name = model_config["name"]
        model_type = model_config["type"]
        model_id = model_config.get("model_id")
        
        logger.log(f"\n🧪 Testing Model: {model_name} ({model_type})")
        logger.log("-"*70)
        
        run_start_time = time.time()
        success = False
        error_message = None
        agent_result = None
        steps_taken = None
        final_answer = None
        
        try:
            agent_config = {
                "question": question,
                "key": f"comprehensive_rul_analytics_hf_{model_name.lower().replace(' ', '_')}",
                "max_steps": max_steps,
                "agent_prompt": agent_prompt,
                "reflect_prompt": reflect_prompt,
                "tool_names": tool_names,
                "tool_desc": tool_desc,
                "tools": all_tools,
                "react_llm_model_id": model_id if model_type == "watsonx" else 15,  # Default to Granite for now
                "reflect_llm_model_id": model_id if model_type == "watsonx" else 15,
                "actionstyle": "Text",
                "reactstyle": "thought_and_act_together",
                "max_retries": 1,
                "num_reflect_iteration": 3,
                "early_stop": False,
                "debug": True,
                "log_structured_messages": False,
            }
            
            logger.log(f"🚀 Creating agent with {model_name}...")
            root_agent = create_reactxen_agent(**agent_config)
            
            logger.log(f"🏃 Running agent...")
            logger.log("="*70)
            
            # Capture stdout during agent execution
            old_stdout = sys.stdout
            captured_output = StringIO()
            
            try:
                sys.stdout = captured_output
                result = root_agent.run()
                sys.stdout = old_stdout
                
                output_text = captured_output.getvalue()
                logger.log(output_text)
                
                # Extract final answer if available
                try:
                    if hasattr(root_agent, 'trajectory') and root_agent.trajectory:
                        # Safely access trajectory - handle empty or invalid indices
                        if len(root_agent.trajectory) > 0:
                            last_step = root_agent.trajectory[-1]
                            if isinstance(last_step, dict) and 'final_answer' in last_step:
                                final_answer = last_step['final_answer']
                except (IndexError, KeyError, TypeError) as e:
                    # Trajectory access failed, try alternative methods
                    pass
                
                # Get steps taken
                if hasattr(root_agent, 'step_n'):
                    steps_taken = root_agent.step_n
                elif hasattr(root_agent, 'export_benchmark_metric'):
                    try:
                        metrics = root_agent.export_benchmark_metric()
                        if metrics and 'per_round_info' in metrics and metrics['per_round_info']:
                            steps_taken = metrics['per_round_info'][-1].get('step', None)
                    except:
                        pass
                
                agent_result = str(result)
                
                # Determine success based on agent state and result
                if hasattr(root_agent, 'finished') and root_agent.finished:
                    success = True
                elif hasattr(root_agent, 'export_benchmark_metric'):
                    try:
                        metrics = root_agent.export_benchmark_metric()
                        if metrics and metrics.get('status') == 'Accomplished':
                            success = True
                        elif metrics and metrics.get('status') == 'Not Accomplished':
                            success = False
                        else:
                            # If we got a result and it's not empty, consider it successful
                            success = bool(agent_result and agent_result.strip() and agent_result != "-1")
                    except:
                        success = bool(agent_result and agent_result.strip() and agent_result != "-1")
                else:
                    # Default: if we got a result, consider it successful
                    success = bool(agent_result and agent_result.strip() and agent_result != "-1")
                
                logger.log("\n" + "="*70)
                logger.log(f"✅ Model {model_name} execution completed successfully")
                logger.log("="*70)
                if success:
                    logger.mark_success()  # Mark that this run was successful
                
            except Exception as e:
                sys.stdout = old_stdout
                error_message = str(e)
                logger.log(f"❌ Error during {model_name} execution: {error_message}")
                logger.log(traceback.format_exc())
                success = False
            
            execution_time = time.time() - run_start_time
            
            # Get metrics if available
            metrics = {}
            try:
                if hasattr(root_agent, 'export_benchmark_metric'):
                    metrics = root_agent.export_benchmark_metric()
            except:
                pass
            
            # Record benchmark result
            benchmark_result = benchmark.record_model_run(
                model_name=model_name,
                model_type=model_type,
                model_id=str(model_id) if model_id else None,
                metrics=metrics,
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                agent_result=agent_result,
                steps_taken=steps_taken,
                final_answer=final_answer,
            )
            
            all_results.append(benchmark_result)
            
            # Update best result
            if success and (best_result is None or benchmark_result["performance_score"] > best_result["performance_score"]):
                best_result = benchmark_result
            
        except Exception as e:
            execution_time = time.time() - run_start_time
            error_message = f"Failed to initialize or run model: {str(e)}"
            logger.log(f"❌ {error_message}")
            
            benchmark.record_model_run(
                model_name=model_name,
                model_type=model_type,
                model_id=str(model_id) if model_id else None,
                execution_time=execution_time,
                success=False,
                error_message=error_message,
            )
    
    # Export benchmark results
    logger.log("\n📊 STEP 7: Benchmark Results")
    logger.log("-"*70)
    
    best_model = benchmark.get_best_model()
    if best_model:
        logger.log(f"🏆 Best Model: {best_model['model_name']} (Score: {best_model['performance_score']:.2f}/100)")
        logger.log(f"   Execution Time: {best_model['execution_time']:.2f}s")
        logger.log(f"   Steps Taken: {best_model.get('steps_taken', 'N/A')}")
        logger.log(f"   Success: {'✅' if best_model['success'] else '❌'}")
    
    # Export in multiple formats
    json_file = benchmark.export_results(format="json")
    md_file = benchmark.export_results(format="markdown")
    txt_file = benchmark.export_results(format="txt")
    
    logger.log(f"✅ Benchmark results exported:")
    logger.log(f"   - JSON: {json_file}")
    logger.log(f"   - Markdown: {md_file}")
    logger.log(f"   - Text: {txt_file}")
    
    # Step 8: Export agent results (from best model run)
    logger.log("\n💾 STEP 8: Exporting Agent Results")
    logger.log("-"*70)
    
    import json
    
    # Export trajectory from last successful run
    try:
        if 'root_agent' in locals() and hasattr(root_agent, 'export_trajectory'):
            trajectory = root_agent.export_trajectory()
            traj_file = output_dir / f"agent_trajectory_{timestamp}.json"
            with open(traj_file, 'w') as f:
                json.dump(trajectory, f, indent=2)
            logger.log(f"✅ Trajectory exported to {traj_file}")
            
            metrics = root_agent.export_benchmark_metric()
            metrics_file = output_dir / f"agent_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.log(f"✅ Metrics exported to {metrics_file}")
            
            summary = root_agent.get_experiment_summary()
            summary_file = output_dir / f"agent_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.log(f"✅ Summary exported to {summary_file}")
    except Exception as e:
        logger.log(f"⚠️  Could not export agent results: {str(e)}")
    
    logger.log(f"\n📁 All outputs saved to: {output_dir}")
    
    # Only show success message if at least one run was successful
    if best_model and best_model.get('success', False) and best_model.get('performance_score', 0) >= 50:
        logger.mark_success()
        logger.close()
        print("\n" + "="*70)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"🏆 Best Model: {best_model['model_name']} (Score: {best_model['performance_score']:.2f}/100)")
        print("="*70)
    else:
        logger.close()
        print("\n" + "="*70)
        print("⚠️  DEMO COMPLETED WITH ISSUES")
        if best_model:
            print(f"⚠️  Best Model: {best_model['model_name']} (Score: {best_model['performance_score']:.2f}/100)")
            print("⚠️  Performance below 50 or task not completed successfully")
        else:
            print("⚠️  No successful model runs")
        print("="*70)


if __name__ == "__main__":
    main()

