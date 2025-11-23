"""
Generate Performance Diagrams for Research Paper
Creates visualizations showing performance improvements.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_performance_comparison_chart(evaluations: List[Dict[str, Any]], output_dir: Path):
    """Generate performance comparison chart vs PDMBench baseline."""
    # Group by task type
    task_types = {}
    for eval_data in evaluations:
        task_type = eval_data.get('task_type', 'unknown')
        if task_type not in task_types:
            task_types[task_type] = []
        task_types[task_type].append(eval_data)
    
    # PDMBench baseline scores
    baselines = {
        'rul_prediction': 75.2,
        'fault_classification': 76.5,
        'cost_benefit': 70.0,
        'safety_policies': 72.0
    }
    
    # Calculate averages
    task_names = []
    our_scores = []
    baseline_scores = []
    improvements = []
    
    for task_type, evals in task_types.items():
        if task_type in baselines:
            task_names.append(task_type.replace('_', ' ').title())
            avg_score = sum(e['metrics']['overall_score'] for e in evals) / len(evals) if evals else 0.0
            our_scores.append(avg_score)
            baseline_scores.append(baselines[task_type])
            improvements.append(avg_score - baselines[task_type])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Score Comparison
    x = range(len(task_names))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], baseline_scores, width, label='PDMBench Baseline', color='#ff7f7f', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], our_scores, width, label='Our Multi-Agent Approach', color='#7fbf7f', alpha=0.8)
    
    ax1.set_xlabel('Task Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison: Our Approach vs PDMBench Baseline', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Improvement Percentage
    colors = ['#4caf50' if imp > 0 else '#f44336' for imp in improvements]
    bars3 = ax2.bar(task_names, improvements, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Task Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score Improvement', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(task_names, rotation=15, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'+{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_success_criteria_chart(evaluations: List[Dict[str, Any]], output_dir: Path):
    """Generate chart showing success criteria fulfillment."""
    total = len(evaluations)
    task_accomplished = sum(1 for e in evaluations if e.get('success_criteria_met', {}).get('task_accomplished', False))
    ground_truth_validated = sum(1 for e in evaluations if e.get('success_criteria_met', {}).get('ground_truth_validated', False))
    both_met = sum(1 for e in evaluations if e.get('success', False))
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Data
    sizes = [both_met, task_accomplished - both_met, total - task_accomplished]
    labels = [
        f'✅ Both Criteria Met\n({both_met} tasks, {both_met/total*100:.1f}%)',
        f'⚠️ Task Only\n({task_accomplished - both_met} tasks)',
        f'❌ Failed\n({total - task_accomplished} tasks)'
    ]
    colors = ['#4caf50', '#ff9800', '#f44336']
    explode = (0.1, 0, 0)  # Emphasize successful
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Success Criteria Fulfillment\n(Task Accomplished + Ground Truth Validated)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"success_criteria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_execution_time_chart(evaluations: List[Dict[str, Any]], output_dir: Path):
    """Generate execution time comparison chart."""
    times = [e['execution_time'] for e in evaluations]
    successful = [e['execution_time'] for e in evaluations if e.get('success', False)]
    failed = [e['execution_time'] for e in evaluations if not e.get('success', False)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    bins = range(0, 200, 10)
    ax.hist([successful, failed], bins=bins, label=['✅ Successful', '❌ Failed'], 
            color=['#4caf50', '#f44336'], alpha=0.7, edgecolor='black')
    
    # Add target lines
    ax.axvline(x=120, color='blue', linestyle='--', linewidth=2, label='Target: 120s')
    ax.axvline(x=180, color='red', linestyle='--', linewidth=2, label='Hard Limit: 180s')
    
    ax.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tasks', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    avg_time = sum(times) / len(times) if times else 0
    avg_success = sum(successful) / len(successful) if successful else 0
    ax.text(0.02, 0.98, f'Avg (All): {avg_time:.1f}s\nAvg (Successful): {avg_success:.1f}s',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"execution_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_all_diagrams(evaluation_report_file: Path):
    """Generate all performance diagrams from evaluation report."""
    with open(evaluation_report_file, 'r') as f:
        report = json.load(f)
    
    evaluations = report.get('evaluations', [])
    output_dir = evaluation_report_file.parent / 'diagrams'
    output_dir.mkdir(exist_ok=True)
    
    diagrams = {}
    
    try:
        diagrams['comparison'] = generate_performance_comparison_chart(evaluations, output_dir)
        diagrams['success_criteria'] = generate_success_criteria_chart(evaluations, output_dir)
        diagrams['execution_time'] = generate_execution_time_chart(evaluations, output_dir)
        
        print(f"✅ Generated {len(diagrams)} diagrams in {output_dir}")
        return diagrams
    except Exception as e:
        print(f"⚠️  Could not generate diagrams (matplotlib may not be available): {e}")
        return {}


if __name__ == "__main__":
    # Find latest evaluation report
    eval_dir = Path(__file__).parent / "outputs" / "evaluation"
    if eval_dir.exists():
        reports = sorted(eval_dir.glob("evaluation_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if reports:
            generate_all_diagrams(reports[0])

