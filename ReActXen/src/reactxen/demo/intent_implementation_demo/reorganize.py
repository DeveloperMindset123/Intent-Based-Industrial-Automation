"""
Script to reorganize code into single_agent and multi_agent directories.
Run this once to set up the new structure.
"""
import shutil
from pathlib import Path

base_dir = Path(__file__).parent
single_dir = base_dir / "single_agent_implementation_demo"
multi_dir = base_dir / "multi_agent_implementation_demo"
shared_dir = base_dir / "shared"

# Create directories
single_dir.mkdir(exist_ok=True)
multi_dir.mkdir(exist_ok=True)
shared_dir.mkdir(exist_ok=True)

# Files for single agent implementation
single_agent_files = [
    "main.py",
    "agent_implementation.py",
    "agent_implementation_hf.py",
    "benchmark.py",
    "tools_logic.py",
]

# Files for multi agent implementation
multi_agent_files = [
    "main_hierarchical.py",
    "root_agent_implementation.py",
    "hierarchical_agents.py",
    "agent_tool_wrapper.py",
    "data_scientist_agent.py",
    "predictive_maintenance_agent.py",
    "cost_benefit_agent.py",
    "safety_policy_agent.py",
    "sub_agent_base.py",
    "dataset_tools.py",
    "ml_framework_tools.py",
    "ml_models_state.py",
    "ml_data_prep.py",
    "sklearn_tools.py",
    "pytorch_tools.py",
    "tensorflow_tools.py",
    "ml_prediction_tools.py",
]

# Shared files
shared_files = [
    "load_data.py",
    "benchmark_utils.py",
    "shared_utils.py",
    "input_schema.py",
]

# Copy files (don't move, keep originals for now)
print("Copying files to new structure...")
for file in single_agent_files:
    src = base_dir / file
    if src.exists():
        dst = single_dir / file
        shutil.copy2(src, dst)
        print(f"  Copied {file} -> single_agent_implementation_demo/")

for file in multi_agent_files:
    src = base_dir / file
    if src.exists():
        dst = multi_dir / file
        shutil.copy2(src, dst)
        print(f"  Copied {file} -> multi_agent_implementation_demo/")

for file in shared_files:
    src = base_dir / file
    if src.exists():
        dst = shared_dir / file
        shutil.copy2(src, dst)
        print(f"  Copied {file} -> shared/")

# Copy custom_tools directory
if (base_dir / "custom_tools").exists():
    shutil.copytree(base_dir / "custom_tools", multi_dir / "custom_tools", dirs_exist_ok=True)
    shutil.copytree(base_dir / "custom_tools", single_dir / "custom_tools", dirs_exist_ok=True)
    print("  Copied custom_tools/ to both directories")

print("\n✅ Reorganization complete!")
print(f"   Single agent: {single_dir}")
print(f"   Multi agent: {multi_dir}")
print(f"   Shared: {shared_dir}")

