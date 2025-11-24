"""
Export all scenarios from intent_implementation_demo and multi_agent_implementation_demo
to a centralized scenario directory.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add paths for imports
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "multi_agent_implementation_demo"))

def export_scenarios():
    """Export all scenarios from both directories to scenario/ directory."""
    
    # Create scenario directory in the base directory
    scenario_dir = base_dir / "scenario"
    scenario_dir.mkdir(exist_ok=True)
    
    all_scenarios = []
    
    # 1. Export scenarios from multi_agent_implementation_demo
    print("📋 Exporting scenarios from multi_agent_implementation_demo...")
    try:
        from multi_agent_implementation_demo.datasets_scenarios import (
            DatasetManager, ScenarioGenerator, save_scenarios
        )
        
        # Generate fresh scenarios
        dataset_manager = DatasetManager()
        scenario_generator = ScenarioGenerator(dataset_manager)
        multi_agent_scenarios = scenario_generator.generate_all_scenarios()
        
        # Add source identifier
        for scenario in multi_agent_scenarios:
            scenario['_source'] = 'multi_agent_implementation_demo'
        
        all_scenarios.extend(multi_agent_scenarios)
        print(f"   ✅ Generated {len(multi_agent_scenarios)} scenarios from multi_agent_implementation_demo")
        
        # Also check if there's an existing saved file
        existing_file = base_dir / "multi_agent_implementation_demo" / "outputs" / "scenarios" / "pdmbench_scenarios.json"
        if existing_file.exists():
            try:
                with open(existing_file, 'r') as f:
                    existing_scenarios = json.load(f)
                    # Save a copy
                    save_path = scenario_dir / "multi_agent_pdmbench_scenarios.json"
                    with open(save_path, 'w') as out_f:
                        json.dump(existing_scenarios, out_f, indent=2, default=str)
                    print(f"   ✅ Copied existing scenarios file to scenario/multi_agent_pdmbench_scenarios.json")
            except Exception as e:
                print(f"   ⚠️  Could not read existing scenarios file: {e}")
                
    except Exception as e:
        print(f"   ⚠️  Error exporting from multi_agent_implementation_demo: {e}")
    
    # 2. Check for scenarios in intent_implementation_demo (parent directory)
    print("\n📋 Checking for scenarios in intent_implementation_demo...")
    try:
        # Look for any scenario-related files or generate from benchmark.py if it has scenarios
        parent_scenario_files = list(base_dir.glob("*scenario*.json"))
        if parent_scenario_files:
            for scenario_file in parent_scenario_files:
                try:
                    with open(scenario_file, 'r') as f:
                        scenarios = json.load(f)
                        if isinstance(scenarios, list):
                            # Add source identifier
                            for scenario in scenarios:
                                if isinstance(scenario, dict):
                                    scenario['_source'] = 'intent_implementation_demo'
                            all_scenarios.extend(scenarios)
                            
                            # Copy file to scenario directory
                            copy_path = scenario_dir / scenario_file.name
                            with open(copy_path, 'w') as out_f:
                                json.dump(scenarios, out_f, indent=2, default=str)
                            print(f"   ✅ Exported {len(scenarios)} scenarios from {scenario_file.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not process {scenario_file.name}: {e}")
        else:
            print("   ℹ️  No scenario files found in intent_implementation_demo")
    except Exception as e:
        print(f"   ⚠️  Error checking intent_implementation_demo: {e}")
    
    # 3. Save all combined scenarios
    if all_scenarios:
        combined_path = scenario_dir / "all_scenarios.json"
        with open(combined_path, 'w') as f:
            json.dump(all_scenarios, f, indent=2, default=str)
        print(f"\n✅ Saved {len(all_scenarios)} total scenarios to scenario/all_scenarios.json")
        
        # Also save by type
        by_type = {}
        for scenario in all_scenarios:
            scenario_type = scenario.get('type', 'unknown')
            if scenario_type not in by_type:
                by_type[scenario_type] = []
            by_type[scenario_type].append(scenario)
        
        for scenario_type, scenarios in by_type.items():
            type_path = scenario_dir / f"scenarios_{scenario_type}.json"
            with open(type_path, 'w') as f:
                json.dump(scenarios, f, indent=2, default=str)
            print(f"   ✅ Saved {len(scenarios)} {scenario_type} scenarios to scenario/scenarios_{scenario_type}.json")
    
    # 4. Create a summary
    summary = {
        "total_scenarios": len(all_scenarios),
        "by_type": {},
        "by_source": {},
        "files_created": []
    }
    
    for scenario in all_scenarios:
        scenario_type = scenario.get('type', 'unknown')
        source = scenario.get('_source', 'unknown')
        summary["by_type"][scenario_type] = summary["by_type"].get(scenario_type, 0) + 1
        summary["by_source"][source] = summary["by_source"].get(source, 0) + 1
    
    for file in scenario_dir.glob("*.json"):
        summary["files_created"].append(file.name)
    
    summary_path = scenario_dir / "scenarios_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to scenario/scenarios_summary.json")
    print(f"\n{'='*60}")
    print(f"Scenarios exported successfully!")
    print(f"{'='*60}")
    print(f"Total scenarios: {len(all_scenarios)}")
    print(f"By type:")
    for scenario_type, count in summary["by_type"].items():
        print(f"  - {scenario_type}: {count}")
    print(f"By source:")
    for source, count in summary["by_source"].items():
        print(f"  - {source}: {count}")
    print(f"\nFiles created in scenario/ directory:")
    for file_name in summary["files_created"]:
        print(f"  - {file_name}")
    
    return scenario_dir, all_scenarios


if __name__ == "__main__":
    scenario_dir, all_scenarios = export_scenarios()

