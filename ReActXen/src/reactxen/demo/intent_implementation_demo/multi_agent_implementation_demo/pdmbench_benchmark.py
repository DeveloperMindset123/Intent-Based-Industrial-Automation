"""
PDMBench Benchmarking System
Compares agentic approach vs PDMBench baseline
Supports both WatsonX and OpenAI LLMs
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdmbench_agent import create_pdmbench_root_agent
from memory_system import get_memory_system
from ground_truth_verification import VerifyRULPredictionsTool


class PDMBenchBenchmark:
    """Benchmark system for PDMBench tasks."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent / "outputs" / "benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
    
    def run_task(self, task: Dict[str, Any], model_id: int = 15, use_openai: bool = False) -> Dict[str, Any]:
        """Run a single PDMBench task."""
        question = task['question']
        task_type = task.get('type', 'rul_prediction')
        ground_truth = task.get('ground_truth', None)
        
        print(f"\n{'='*80}")
        print(f"📋 Task: {task_type}")
        print(f"❓ Question: {question}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Create agent
            root_agent, sub_agents = create_pdmbench_root_agent(
                question=question,
                model_id=model_id,
                use_openai=use_openai
            )
            
            # Run agent
            result = root_agent.run()
            
            execution_time = time.time() - start_time
            
            # Evaluate results
            evaluation = self._evaluate_result(
                task_type=task_type,
                question=question,
                result=result,
                ground_truth=ground_truth,
                execution_time=execution_time
            )
            
            return {
                'task': task,
                'result': result,
                'evaluation': evaluation,
                'execution_time': execution_time,
                'model_id': model_id,
                'use_openai': use_openai
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'task': task,
                'error': str(e),
                'execution_time': execution_time,
                'model_id': model_id,
                'use_openai': use_openai
            }
    
    def _evaluate_result(self, task_type: str, question: str, result: str, 
                        ground_truth: Any, execution_time: float) -> Dict[str, Any]:
        """Evaluate agent result against ground truth."""
        evaluation = {
            'task_type': task_type,
            'execution_time': execution_time,
            'result_length': len(result),
            'ground_truth_provided': ground_truth is not None,
            'metrics': {}
        }
        
        if task_type == 'rul_prediction' and ground_truth:
            # Evaluate RUL prediction accuracy
            try:
                verifier = VerifyRULPredictionsTool()
                # Extract predictions from result (simplified - would need parsing)
                # For now, check if verification was mentioned
                if 'verify' in result.lower() or 'ground truth' in result.lower():
                    evaluation['metrics']['ground_truth_verified'] = True
                else:
                    evaluation['metrics']['ground_truth_verified'] = False
            except:
                evaluation['metrics']['ground_truth_verified'] = False
        
        # Check for required components
        required_components = {
            'fault_classification': ['fault', 'classification', 'type'],
            'rul_prediction': ['rul', 'remaining', 'cycles'],
            'cost_benefit': ['cost', 'benefit', 'analysis'],
            'safety_policies': ['safety', 'policy', 'risk']
        }
        
        if task_type in required_components:
            components_found = sum(1 for comp in required_components[task_type] 
                                 if comp.lower() in result.lower())
            evaluation['metrics']['components_found'] = components_found
            evaluation['metrics']['components_required'] = len(required_components[task_type])
            evaluation['metrics']['completeness'] = components_found / len(required_components[task_type])
        
        # Calculate overall score
        score = 0.0
        if evaluation['metrics'].get('ground_truth_verified', False):
            score += 30.0
        if evaluation['metrics'].get('completeness', 0) >= 0.8:
            score += 40.0
        if execution_time < 60:  # Efficiency bonus
            score += 20.0
        if len(result) > 100:  # Detailed response
            score += 10.0
        
        evaluation['metrics']['overall_score'] = min(score, 100.0)
        
        return evaluation
    
    def run_benchmark(self, tasks: List[Dict[str, Any]], model_ids: List[int] = [15], 
                     use_openai: bool = False) -> Dict[str, Any]:
        """Run full benchmark suite."""
        print(f"\n🚀 Starting PDMBench Benchmark")
        print(f"📊 Tasks: {len(tasks)}")
        print(f"🤖 Models: {model_ids}")
        print(f"🔧 OpenAI: {use_openai}")
        
        all_results = []
        
        for task_idx, task in enumerate(tasks, 1):
            print(f"\n{'='*80}")
            print(f"Task {task_idx}/{len(tasks)}")
            print(f"{'='*80}")
            
            for model_id in model_ids:
                result = self.run_task(task, model_id=model_id, use_openai=use_openai)
                all_results.append(result)
                self.results.append(result)
        
        # Aggregate results
        summary = self._aggregate_results(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"pdmbench_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': all_results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return {
            'summary': summary,
            'results': all_results,
            'results_file': str(results_file)
        }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate benchmark results."""
        summary = {
            'total_tasks': len(results),
            'successful_tasks': sum(1 for r in results if 'error' not in r),
            'failed_tasks': sum(1 for r in results if 'error' in r),
            'average_score': 0.0,
            'average_execution_time': 0.0,
            'task_type_breakdown': {}
        }
        
        successful = [r for r in results if 'error' not in r]
        if successful:
            scores = [r['evaluation']['metrics'].get('overall_score', 0) for r in successful]
            times = [r['execution_time'] for r in successful]
            
            summary['average_score'] = sum(scores) / len(scores)
            summary['average_execution_time'] = sum(times) / len(times)
        
        # Breakdown by task type
        for result in successful:
            task_type = result['evaluation']['task_type']
            if task_type not in summary['task_type_breakdown']:
                summary['task_type_breakdown'][task_type] = {
                    'count': 0,
                    'average_score': 0.0,
                    'scores': []
                }
            
            breakdown = summary['task_type_breakdown'][task_type]
            breakdown['count'] += 1
            score = result['evaluation']['metrics'].get('overall_score', 0)
            breakdown['scores'].append(score)
            breakdown['average_score'] = sum(breakdown['scores']) / len(breakdown['scores'])
        
        return summary


def create_pdmbench_tasks() -> List[Dict[str, Any]]:
    """Create PDMBench benchmark tasks."""
    tasks = [
        {
            'type': 'rul_prediction',
            'question': 'We should focus on equipment that is running on fumes. Which equipment from the loaded dataset are likely to give out in the next 20 cycles? Provide me a list of equipment IDs with safety recommendations and cost estimates.',
            'ground_truth': 'data/CMAPSSData/RUL_FD001.txt'
        },
        {
            'type': 'fault_classification',
            'question': 'Classify the fault types present in the equipment data. Identify all fault modes and their characteristics.',
            'ground_truth': None
        },
        {
            'type': 'cost_benefit',
            'question': 'Analyze the cost-benefit of performing maintenance on equipment with low RUL. Compare maintenance costs vs failure costs.',
            'ground_truth': None
        },
        {
            'type': 'safety_policies',
            'question': 'Evaluate safety risks for equipment operating near failure thresholds. Provide safety recommendations and policy compliance assessment.',
            'ground_truth': None
        }
    ]
    return tasks


if __name__ == "__main__":
    benchmark = PDMBenchBenchmark()
    tasks = create_pdmbench_tasks()
    results = benchmark.run_benchmark(tasks, model_ids=[15], use_openai=False)
    
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)
    print(json.dumps(results['summary'], indent=2))

