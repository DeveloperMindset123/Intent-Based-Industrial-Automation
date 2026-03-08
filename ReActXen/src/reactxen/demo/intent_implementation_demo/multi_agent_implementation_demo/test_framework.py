"""Test script for multi-agent framework with proper path setup."""

import sys
from pathlib import Path

# Add reactxen to path
base_path = Path(__file__).parent.parent.parent.parent.parent.parent
reactxen_src = base_path / "ReActXen" / "src"
sys.path.insert(0, str(reactxen_src))

print("=== Multi-Agent Framework Test ===\n")

# Test 1: Dataset categorization
print("1. Testing dataset categorization...")
try:
    from utils.dataset_categorizer import categorize_dataset

    assert categorize_dataset("CMAPSS_FD001") == "rul"
    assert categorize_dataset("CWRU") == "fault"
    print("   ✓ Dataset categorization works\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 2: Data splitting
print("2. Testing data splitter...")
try:
    from utils.data_splitter import split_dataset

    print("   ✓ Data splitter imported\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 3: Model loader
print("3. Testing model loader...")
try:
    from models.model_loader import list_available_models

    models = list_available_models("huggingface")

    print(f"   ✓ Found {len(models)} WatsonX models\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 4: Tools
print("4. Testing tools...")
try:
    from tools.data_tools import LoadDatasetTool
    from tools.metric_tools import CalculateMAETool

    tool = LoadDatasetTool()
    print(f"   ✓ Tools imported: {tool.name}\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

print("=== Framework Test Complete ===")
