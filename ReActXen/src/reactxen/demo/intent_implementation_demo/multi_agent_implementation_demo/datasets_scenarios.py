"""
Dataset and Scenario Preparation for PDMBench Benchmarking
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json


class DatasetManager:
    """Manages datasets for PDMBench benchmarking."""
    
    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            # Try multiple possible locations
            base_dir = Path(__file__).parent.parent.parent.parent.parent.parent
            possible_dirs = [
                base_dir / "data" / "CMAPSSData",
                Path("/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData"),
                Path(__file__).parent.parent.parent / "downloaded_datasets"
            ]
            for d in possible_dirs:
                if d.exists():
                    self.data_dir = d
                    break
            else:
                self.data_dir = base_dir / "data" / "CMAPSSData"
        else:
            self.data_dir = data_dir
    
    def load_cmapss_data(self, dataset: str = "FD001") -> Dict[str, pd.DataFrame]:
        """Load CMAPSS dataset."""
        data = {}
        for split in ['train', 'test']:
            file_path = self.data_dir / f"{split}_{dataset}.txt"
            if file_path.exists():
                # CMAPSS format: unit, time, op1, op2, op3, sensor1-21
                data[split] = pd.read_csv(
                    file_path, 
                    sep=r'\s+',  # Raw string to avoid escape sequence warning
                    header=None,
                    names=[f'col_{i}' for i in range(26)]
                )
                # Rename key columns
                data[split].rename(columns={
                    'col_0': 'unit',
                    'col_1': 'time',
                    'col_2': 'op_setting_1',
                    'col_3': 'op_setting_2',
                    'col_4': 'op_setting_3'
                }, inplace=True)
        return data
    
    def load_ground_truth(self, dataset: str = "FD001") -> pd.Series:
        """Load ground truth RUL values."""
        gt_path = self.data_dir / f"RUL_{dataset}.txt"
        if gt_path.exists():
            return pd.read_csv(gt_path, header=None, names=['rul']).squeeze()
        return None
    
    def prepare_scenario_data(self, dataset: str = "FD001") -> Dict[str, Any]:
        """Prepare scenario data for benchmarking."""
        data = self.load_cmapss_data(dataset)
        ground_truth = self.load_ground_truth(dataset)
        
        scenario = {
            'dataset': dataset,
            'train_data': data.get('train'),
            'test_data': data.get('test'),
            'ground_truth': ground_truth,
            'num_units': len(data['test']['unit'].unique()) if 'test' in data else 0,
            'num_sensors': 21,  # CMAPSS has 21 sensors
            'metadata': {
                'description': f'CMAPSS {dataset} dataset',
                'conditions': 'ONE' if dataset in ['FD001', 'FD003'] else 'SIX',
                'fault_modes': 'ONE' if dataset in ['FD001', 'FD002'] else 'TWO'
            }
        }
        
        return scenario


class ScenarioGenerator:
    """Generates benchmark scenarios for PDMBench tasks."""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
    
    def generate_rul_scenarios(self, dataset: str = "FD001", num_scenarios: int = 5) -> List[Dict[str, Any]]:
        """Generate RUL prediction scenarios."""
        scenario_data = self.dataset_manager.prepare_scenario_data(dataset)
        scenarios = []
        
        # Scenario 1: Equipment likely to fail soon
        scenarios.append({
            'type': 'rul_prediction',
            'question': 'We should focus on equipment that is running on fumes. Which equipment from the loaded dataset are likely to give out in the next 20 cycles? Provide me a list of equipment IDs with safety recommendations and cost estimates.',
            'ground_truth_file': f'RUL_{dataset}.txt',
            'threshold': 20,
            'expected_outputs': ['equipment_ids', 'rul_predictions', 'safety_recommendations', 'cost_estimates']
        })
        
        # Scenario 2: Critical equipment identification
        scenarios.append({
            'type': 'rul_prediction',
            'question': 'Identify the top 10 equipment units with the lowest remaining useful life. For each unit, predict the RUL and provide maintenance recommendations.',
            'ground_truth_file': f'RUL_{dataset}.txt',
            'threshold': None,
            'expected_outputs': ['top_10_units', 'rul_predictions', 'maintenance_recommendations']
        })
        
        # Scenario 3: Batch RUL prediction
        scenarios.append({
            'type': 'rul_prediction',
            'question': 'Predict the remaining useful life for all test equipment units. Verify your predictions against the ground truth data and report accuracy metrics.',
            'ground_truth_file': f'RUL_{dataset}.txt',
            'threshold': None,
            'expected_outputs': ['all_rul_predictions', 'ground_truth_verification', 'accuracy_metrics']
        })
        
        return scenarios[:num_scenarios]
    
    def generate_fault_classification_scenarios(self, num_scenarios: int = 3) -> List[Dict[str, Any]]:
        """Generate fault classification scenarios."""
        scenarios = [
            {
                'type': 'fault_classification',
                'question': 'Classify the fault types present in the equipment data. Identify all fault modes and their characteristics based on sensor patterns.',
                'expected_outputs': ['fault_types', 'fault_characteristics', 'sensor_patterns']
            },
            {
                'type': 'fault_classification',
                'question': 'Analyze the sensor data to detect anomalies and classify them into known fault categories. Provide confidence scores for each classification.',
                'expected_outputs': ['anomalies_detected', 'fault_categories', 'confidence_scores']
            },
            {
                'type': 'fault_classification',
                'question': 'For equipment showing degradation signs, classify the primary fault mode and identify contributing factors.',
                'expected_outputs': ['primary_fault_mode', 'contributing_factors', 'degradation_analysis']
            }
        ]
        return scenarios[:num_scenarios]
    
    def generate_cost_benefit_scenarios(self, num_scenarios: int = 2) -> List[Dict[str, Any]]:
        """Generate cost-benefit analysis scenarios."""
        scenarios = [
            {
                'type': 'cost_benefit',
                'question': 'Analyze the cost-benefit of performing preventive maintenance on equipment with low RUL. Compare maintenance costs vs failure costs, and recommend optimal maintenance schedule.',
                'expected_outputs': ['maintenance_costs', 'failure_costs', 'cost_comparison', 'maintenance_schedule']
            },
            {
                'type': 'cost_benefit',
                'question': 'Calculate the return on investment (ROI) for predictive maintenance vs reactive maintenance. Consider equipment downtime, repair costs, and safety implications.',
                'expected_outputs': ['roi_analysis', 'downtime_costs', 'repair_costs', 'safety_implications']
            }
        ]
        return scenarios[:num_scenarios]
    
    def generate_safety_policies_scenarios(self, num_scenarios: int = 2) -> List[Dict[str, Any]]:
        """Generate safety/policies scenarios."""
        scenarios = [
            {
                'type': 'safety_policies',
                'question': 'Evaluate safety risks for equipment operating near failure thresholds. Provide safety recommendations and assess policy compliance for continued operation.',
                'expected_outputs': ['safety_risks', 'safety_recommendations', 'policy_compliance']
            },
            {
                'type': 'safety_policies',
                'question': 'Assess the safety implications of delaying maintenance on critical equipment. Identify regulatory compliance issues and recommend action plans.',
                'expected_outputs': ['safety_implications', 'regulatory_compliance', 'action_plans']
            }
        ]
        return scenarios[:num_scenarios]
    
    def generate_all_scenarios(self) -> List[Dict[str, Any]]:
        """Generate all benchmark scenarios."""
        all_scenarios = []
        all_scenarios.extend(self.generate_rul_scenarios())
        all_scenarios.extend(self.generate_fault_classification_scenarios())
        all_scenarios.extend(self.generate_cost_benefit_scenarios())
        all_scenarios.extend(self.generate_safety_policies_scenarios())
        return all_scenarios


def save_scenarios(scenarios: List[Dict[str, Any]], output_path: Path):
    """Save scenarios to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(scenarios, f, indent=2, default=str)
    print(f"✅ Saved {len(scenarios)} scenarios to {output_path}")


if __name__ == "__main__":
    # Prepare datasets and scenarios
    dataset_manager = DatasetManager()
    scenario_generator = ScenarioGenerator(dataset_manager)
    
    scenarios = scenario_generator.generate_all_scenarios()
    
    output_dir = Path(__file__).parent / "outputs" / "scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_scenarios(scenarios, output_dir / "pdmbench_scenarios.json")
    print(f"\n📊 Generated {len(scenarios)} benchmark scenarios")

