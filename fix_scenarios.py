#!/usr/bin/env python3
"""
Script to fix scenario dataset assignments based on classification type
and extract correct ground truth from datasets.

RUL Prediction scenarios should only use: CMAPSS_FD001-FD004, FEMTO
Fault Classification scenarios can use: CWRU, IMS, XJTU, HUST, MFPT, Paderborn, Mendeley, etc.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import os

# Define workspace root
WORKSPACE_ROOT = Path("/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation")
SCENARIOS_FILE = WORKSPACE_ROOT / "ReActXen/src/reactxen/demo/intent_implementation_demo/scenario/my_scenarios.json"
DATA_DIR = WORKSPACE_ROOT / "ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/PDMBench_Data_Directory/submission096"

# Available datasets by type
RUL_DATASETS = ['CMAPSS_FD001', 'CMAPSS_FD002', 'CMAPSS_FD003', 'CMAPSS_FD004', 'FEMTO']
FAULT_CLASSIFICATION_DATASETS = ['CWRU', 'IMS', 'ElectricMotorVibrations', 'Padeborn']

# Change log
changes_log = []

def log_change(task_id: str, field: str, old_value: Any, new_value: Any, reason: str):
    """Log a change made to a scenario"""
    changes_log.append({
        'task_id': task_id,
        'field': field,
        'old_value': old_value,
        'new_value': new_value,
        'reason': reason
    })
    print(f"[{task_id}] {field}: {old_value} → {new_value} | {reason}")

def extract_rul_from_cmapss(dataset_name: str, top_n: int = 10) -> Tuple[List[int], float, float]:
    """
    Extract ground truth from CMAPSS dataset
    Returns: (list of unit IDs with lowest RUL, MAE range, RMSE range)
    """
    rul_file = DATA_DIR / f"RUL_{dataset_name.split('_')[1]}.txt"

    if not rul_file.exists():
        print(f"WARNING: RUL file not found: {rul_file}")
        return list(range(top_n)), 10.0, 15.0

    # Read RUL values
    with open(rul_file, 'r') as f:
        rul_values = [float(line.strip()) for line in f if line.strip()]

    # Create DataFrame with unit IDs and RUL
    df = pd.DataFrame({
        'unit_id': range(len(rul_values)),
        'rul': rul_values
    })

    # Get top N units with lowest RUL
    top_units = df.nsmallest(top_n, 'rul')['unit_id'].tolist()

    # Calculate MAE and RMSE ranges (simplified estimates based on RUL distribution)
    rul_std = df['rul'].std()
    mae_range = max(5.0, rul_std * 0.3)
    rmse_range = max(8.0, rul_std * 0.4)

    return top_units, mae_range, rmse_range

def extract_rul_from_femto(top_n: int = 10) -> Tuple[List[int], float, float]:
    """
    Extract ground truth from FEMTO dataset using rul_percentage
    Returns: (list of segment IDs with highest RUL percentage, MAE range, RMSE range)
    """
    test_file = DATA_DIR / "FEMTO_test_split.csv"

    if not test_file.exists():
        print(f"WARNING: FEMTO test file not found: {test_file}")
        return list(range(top_n)), 10.0, 15.0

    # Read FEMTO data
    df = pd.read_csv(test_file)

    if 'rul_percentage' not in df.columns:
        print(f"WARNING: rul_percentage column not found in FEMTO")
        return list(range(top_n)), 10.0, 15.0

    # Sort by RUL percentage descending and drop duplicates to get unique segments
    # Higher RUL percentage = closer to failure = critical
    df_sorted = df.sort_values('rul_percentage', ascending=False)
    df_unique = df_sorted.drop_duplicates(subset=['segment_idx'], keep='first')

    # Get top N unique segments
    top_segments = df_unique.head(top_n)['segment_idx'].tolist()

    # Calculate MAE and RMSE ranges (based on RUL percentage as a proxy)
    # Convert to a scale more appropriate for RUL errors
    rul_std = df['rul_percentage'].std() * 100  # Scale to percentage points
    mae_range = max(5.0, rul_std * 0.3)
    rmse_range = max(8.0, rul_std * 0.4)

    return top_segments, mae_range, rmse_range

def extract_fault_classification_units(dataset_name: str, top_n: int = 10) -> List[int]:
    """
    Extract sample units from fault classification datasets
    Returns: list of segment IDs representing different fault types
    """
    test_file = DATA_DIR / f"{dataset_name}_test_split.csv"

    if not test_file.exists():
        print(f"WARNING: Test file not found: {test_file}")
        return list(range(top_n))

    # Read dataset
    df = pd.read_csv(test_file)

    if 'label' not in df.columns:
        print(f"WARNING: label column not found in {dataset_name}")
        return list(range(top_n))

    # Check if segment_idx exists, otherwise use row indices
    has_segment_idx = 'segment_idx' in df.columns

    # Get diverse samples from different fault types
    # Sample evenly across fault types
    samples = []
    unique_labels = df['label'].unique()

    samples_per_label = max(1, top_n // len(unique_labels))

    for label in unique_labels:
        label_df = df[df['label'] == label]

        if has_segment_idx:
            label_samples = label_df['segment_idx'].head(samples_per_label).tolist()
        else:
            # Use row indices if segment_idx doesn't exist
            label_samples = label_df.index[:samples_per_label].tolist()

        samples.extend(label_samples)

        if len(samples) >= top_n:
            break

    return samples[:top_n]

def reassign_dataset_for_rul(scenario_idx: int, rul_scenarios_count: Dict[str, int]) -> str:
    """
    Assign an appropriate RUL dataset for a RUL prediction scenario
    Distributes scenarios evenly across available RUL datasets
    """
    # Cycle through RUL datasets
    dataset_idx = scenario_idx % len(RUL_DATASETS)
    dataset = RUL_DATASETS[dataset_idx]

    # Track how many scenarios use each dataset
    rul_scenarios_count[dataset] = rul_scenarios_count.get(dataset, 0) + 1

    return dataset

def reassign_dataset_for_fault(scenario_idx: int, fault_scenarios_count: Dict[str, int]) -> str:
    """
    Assign an appropriate fault classification dataset
    Distributes scenarios evenly across available fault classification datasets
    """
    # Cycle through fault classification datasets
    dataset_idx = scenario_idx % len(FAULT_CLASSIFICATION_DATASETS)
    dataset = FAULT_CLASSIFICATION_DATASETS[dataset_idx]

    # Track how many scenarios use each dataset
    fault_scenarios_count[dataset] = fault_scenarios_count.get(dataset, 0) + 1

    return dataset

def update_scenario_dataset_and_ground_truth(scenario: Dict[str, Any],
                                             scenario_idx: int,
                                             rul_scenarios_count: Dict[str, int],
                                             fault_scenarios_count: Dict[str, int]) -> bool:
    """
    Update scenario with correct dataset assignment and ground truth
    Returns True if changes were made
    """
    task_id = scenario.get('task_id', 'unknown')
    classification_type = scenario.get('classification_type', '')
    current_dataset = scenario.get('dataset', '')

    changes_made = False

    # Handle RUL Prediction scenarios
    if classification_type == "RUL Prediction":
        # Check if current dataset is appropriate for RUL
        if current_dataset not in RUL_DATASETS:
            # Reassign to appropriate RUL dataset
            new_dataset = reassign_dataset_for_rul(scenario_idx, rul_scenarios_count)
            log_change(task_id, 'dataset', current_dataset, new_dataset,
                      f"RUL Prediction requires RUL-capable dataset (was fault classification dataset)")
            scenario['dataset'] = new_dataset
            current_dataset = new_dataset
            changes_made = True

        # Extract ground truth based on dataset
        gt = scenario.get('ground_truth', {})
        expected_top_count = gt.get('expected_top_units_count', 10)

        if current_dataset.startswith('CMAPSS'):
            # Extract from CMAPSS
            top_units, mae_range, rmse_range = extract_rul_from_cmapss(current_dataset, expected_top_count)

            old_units = gt.get('sample_correct_units', [])
            if old_units != top_units:
                log_change(task_id, 'sample_correct_units', old_units, top_units,
                          f"Extracted actual RUL ground truth from {current_dataset}")
                gt['sample_correct_units'] = top_units
                changes_made = True

            # Update MAE range
            old_mae = gt.get('mae_range', [])
            new_mae = [round(mae_range, 1), round(mae_range * 2, 1)]
            if old_mae != new_mae:
                log_change(task_id, 'mae_range', old_mae, new_mae,
                          f"Updated MAE range based on {current_dataset} RUL distribution")
                gt['mae_range'] = new_mae
                changes_made = True

            # Update RMSE range
            old_rmse = gt.get('rmse_range', [])
            new_rmse = [round(rmse_range, 1), round(rmse_range * 2, 1)]
            if old_rmse != new_rmse:
                log_change(task_id, 'rmse_range', old_rmse, new_rmse,
                          f"Updated RMSE range based on {current_dataset} RUL distribution")
                gt['rmse_range'] = new_rmse
                changes_made = True

        elif current_dataset == 'FEMTO':
            # Extract from FEMTO
            top_units, mae_range, rmse_range = extract_rul_from_femto(expected_top_count)

            old_units = gt.get('sample_correct_units', [])
            if old_units != top_units:
                log_change(task_id, 'sample_correct_units', old_units, top_units,
                          f"Extracted actual RUL ground truth from FEMTO rul_percentage")
                gt['sample_correct_units'] = top_units
                changes_made = True

            # Update MAE range
            old_mae = gt.get('mae_range', [])
            new_mae = [round(mae_range, 1), round(mae_range * 2, 1)]
            if old_mae != new_mae:
                log_change(task_id, 'mae_range', old_mae, new_mae,
                          f"Updated MAE range based on FEMTO RUL distribution")
                gt['mae_range'] = new_mae
                changes_made = True

            # Update RMSE range
            old_rmse = gt.get('rmse_range', [])
            new_rmse = [round(rmse_range, 1), round(rmse_range * 2, 1)]
            if old_rmse != new_rmse:
                log_change(task_id, 'rmse_range', old_rmse, new_rmse,
                          f"Updated RMSE range based on FEMTO RUL distribution")
                gt['rmse_range'] = new_rmse
                changes_made = True

        scenario['ground_truth'] = gt

    # Handle Fault Classification scenarios
    elif classification_type == "Fault Classification":
        # Check if current dataset is appropriate for fault classification
        if current_dataset in RUL_DATASETS and current_dataset != 'FEMTO':
            # CMAPSS is primarily for RUL, reassign to fault classification dataset
            new_dataset = reassign_dataset_for_fault(scenario_idx, fault_scenarios_count)
            log_change(task_id, 'dataset', current_dataset, new_dataset,
                      f"Fault Classification requires fault classification dataset (was RUL dataset)")
            scenario['dataset'] = new_dataset
            current_dataset = new_dataset
            changes_made = True

        # Extract ground truth
        gt = scenario.get('ground_truth', {})
        expected_top_count = gt.get('expected_top_units_count', 10)

        if current_dataset in FAULT_CLASSIFICATION_DATASETS:
            top_units = extract_fault_classification_units(current_dataset, expected_top_count)

            old_units = gt.get('sample_correct_units', [])
            if old_units != top_units:
                log_change(task_id, 'sample_correct_units', old_units, top_units,
                          f"Extracted actual fault classification samples from {current_dataset}")
                gt['sample_correct_units'] = top_units
                changes_made = True

        scenario['ground_truth'] = gt

    return changes_made

def main():
    """Main function to fix scenarios"""
    print("=" * 80)
    print("SCENARIO DATASET CORRECTION AND GROUND TRUTH EXTRACTION")
    print("=" * 80)
    print()

    # Load scenarios
    print(f"Loading scenarios from: {SCENARIOS_FILE}")
    with open(SCENARIOS_FILE, 'r') as f:
        data = json.load(f)

    scenarios = data.get('pdm_scenarios', [])
    print(f"Found {len(scenarios)} scenarios")
    print()

    # Track dataset assignments
    rul_scenarios_count = {}
    fault_scenarios_count = {}

    # Process each scenario
    total_changes = 0
    scenarios_changed = 0

    for idx, scenario in enumerate(scenarios):
        task_id = scenario.get('task_id', f'scenario_{idx}')
        classification_type = scenario.get('classification_type', '')

        print(f"\n--- Processing {task_id} ({classification_type}) ---")

        changes_made = update_scenario_dataset_and_ground_truth(
            scenario, idx, rul_scenarios_count, fault_scenarios_count
        )

        if changes_made:
            scenarios_changed += 1

    # Save updated scenarios
    print("\n" + "=" * 80)
    print(f"SAVING UPDATED SCENARIOS")
    print("=" * 80)

    with open(SCENARIOS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved to: {SCENARIOS_FILE}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total scenarios processed: {len(scenarios)}")
    print(f"Scenarios modified: {scenarios_changed}")
    print(f"Total changes made: {len(changes_log)}")
    print()

    print("RUL Dataset Distribution:")
    for dataset, count in sorted(rul_scenarios_count.items()):
        print(f"  {dataset}: {count} scenarios")
    print()

    print("Fault Classification Dataset Distribution:")
    for dataset, count in sorted(fault_scenarios_count.items()):
        print(f"  {dataset}: {count} scenarios")
    print()

    # Print detailed change log
    if changes_log:
        print("=" * 80)
        print("DETAILED CHANGE LOG")
        print("=" * 80)

        # Group by change type
        dataset_changes = [c for c in changes_log if c['field'] == 'dataset']
        ground_truth_changes = [c for c in changes_log if c['field'] == 'sample_correct_units']
        mae_changes = [c for c in changes_log if c['field'] == 'mae_range']
        rmse_changes = [c for c in changes_log if c['field'] == 'rmse_range']

        print(f"\nDataset reassignments: {len(dataset_changes)}")
        for change in dataset_changes:
            print(f"  [{change['task_id']}] {change['old_value']} → {change['new_value']}")

        print(f"\nGround truth extractions: {len(ground_truth_changes)}")
        for change in ground_truth_changes[:10]:  # Show first 10
            old = change['old_value'][:5] if len(change['old_value']) > 5 else change['old_value']
            new = change['new_value'][:5] if len(change['new_value']) > 5 else change['new_value']
            print(f"  [{change['task_id']}] {old}... → {new}...")

        if len(ground_truth_changes) > 10:
            print(f"  ... and {len(ground_truth_changes) - 10} more")

        print(f"\nMAE range updates: {len(mae_changes)}")
        print(f"RMSE range updates: {len(rmse_changes)}")

if __name__ == "__main__":
    main()

