"""
Multi-LLM Benchmarking System
Supports both WatsonX and OpenAI LLMs
"""
import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdmbench_agent import create_pdmbench_root_agent
from datasets_scenarios import DatasetManager, ScenarioGenerator, save_scenarios
from memory_system import get_memory_system


class MultiLLMBenchmark:
    """Benchmark system supporting multiple LLM backends."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent / "outputs" / "benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
    
    def run_with_watsonx(self, task: Dict[str, Any], model_id: int = 15) -> Dict[str, Any]:
        """Run task with WatsonX LLM."""
        print(f"\n🤖 Running with WatsonX (model_id={model_id})...")
        return self._run_task(task, model_id=model_id, use_openai=False)
    
    def run_with_openai(self, task: Dict[str, Any], model_name: str = "gpt-4") -> Dict[str, Any]:
        """Run task with OpenAI LLM."""
        print(f"\n🤖 Running with OpenAI ({model_name})...")
        # Note: OpenAI integration would need to be implemented
        # For now, return placeholder
        return self._run_task(task, model_id=None, use_openai=True, openai_model=model_name)
    
    def _run_task(self, task: Dict[str, Any], model_id: int = None, 
                  use_openai: bool = False, openai_model: str = None) -> Dict[str, Any]:
        """Run a single task."""
        question = task['question']
        task_type = task.get('type', 'rul_prediction')
        
        start_time = time.time()
        
        try:
            # Create agent
            root_agent, sub_agents = create_pdmbench_root_agent(
                question=question,
                model_id=model_id or 15,
                use_openai=use_openai
            )
            
            # Run agent
            result = root_agent.run()
            
            execution_time = time.time() - start_time
            
            # Evaluate
            evaluation = self._evaluate_result(
                task_type=task_type,
                question=question,
                result=str(result) if result else "",
                ground_truth=task.get('ground_truth_file'),
                execution_time=execution_time,
                task=task
            )
            
            return {
                'task': task,
                'result': str(result) if result else "",
                'evaluation': evaluation,
                'execution_time': execution_time,
                'model_id': model_id,
                'use_openai': use_openai,
                'openai_model': openai_model,
                'success': True
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            import traceback
            return {
                'task': task,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'execution_time': execution_time,
                'model_id': model_id,
                'use_openai': use_openai,
                'openai_model': openai_model,
                'success': False
            }
    
    def _evaluate_result(self, task_type: str, question: str, result: str,
                        ground_truth: Any, execution_time: float, task: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent result."""
        evaluation = {
            'task_type': task_type,
            'execution_time': execution_time,
            'result_length': len(result),
            'ground_truth_provided': ground_truth is not None,
            'metrics': {}
        }
        
        # Check for required components
        expected_outputs = task.get('expected_outputs', [])
        components_found = sum(1 for comp in expected_outputs 
                              if comp.lower().replace('_', ' ') in result.lower())
        evaluation['metrics']['components_found'] = components_found
        evaluation['metrics']['components_required'] = len(expected_outputs)
        evaluation['metrics']['completeness'] = components_found / len(expected_outputs) if expected_outputs else 0.0
        
        # Task-specific checks
        if task_type == 'rul_prediction':
            if 'verify' in result.lower() or 'ground truth' in result.lower():
                evaluation['metrics']['ground_truth_verified'] = True
            else:
                evaluation['metrics']['ground_truth_verified'] = False
        
        # Calculate score
        score = 0.0
        if evaluation['metrics'].get('ground_truth_verified', False):
            score += 30.0
        if evaluation['metrics'].get('completeness', 0) >= 0.8:
            score += 40.0
        if execution_time < 120:  # Efficiency bonus
            score += 20.0
        if len(result) > 200:  # Detailed response
            score += 10.0
        
        evaluation['metrics']['overall_score'] = min(score, 100.0)
        
        return evaluation
    
    def run_comparative_benchmark(self, scenarios: List[Dict[str, Any]], 
                                 watsonx_models: List[int] = [15],
                                 openai_models: List[str] = []) -> Dict[str, Any]:
        """Run comparative benchmark across multiple LLMs."""
        print(f"\n🚀 Starting Multi-LLM Benchmark")
        print(f"📊 Scenarios: {len(scenarios)}")
        print(f"🤖 WatsonX Models: {watsonx_models}")
        print(f"🤖 OpenAI Models: {openai_models}")
        
        all_results = []
        
        for scenario_idx, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*80}")
            print(f"Scenario {scenario_idx}/{len(scenarios)}: {scenario['type']}")
            print(f"{'='*80}")
            
            # Run with WatsonX models
            for model_id in watsonx_models:
                result = self.run_with_watsonx(scenario, model_id=model_id)
                result['scenario_index'] = scenario_idx
                all_results.append(result)
                self.results.append(result)
            
            # Run with OpenAI models (if available)
            for model_name in openai_models:
                result = self.run_with_openai(scenario, model_name=model_name)
                result['scenario_index'] = scenario_idx
                all_results.append(result)
                self.results.append(result)
        
        # Aggregate results
        summary = self._aggregate_results(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"multi_llm_benchmark_{timestamp}.json"
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
            'total_runs': len(results),
            'successful_runs': sum(1 for r in results if r.get('success', False)),
            'failed_runs': sum(1 for r in results if not r.get('success', False)),
            'watsonx_results': [],
            'openai_results': [],
            'comparison': {}
        }
        
        successful = [r for r in results if r.get('success', False)]
        
        # Separate by LLM type
        watsonx = [r for r in successful if not r.get('use_openai', False)]
        openai = [r for r in successful if r.get('use_openai', False)]
        
        if watsonx:
            watsonx_scores = [r['evaluation']['metrics'].get('overall_score', 0) for r in watsonx]
            watsonx_times = [r['execution_time'] for r in watsonx]
            summary['watsonx_results'] = {
                'count': len(watsonx),
                'average_score': sum(watsonx_scores) / len(watsonx_scores),
                'average_time': sum(watsonx_times) / len(watsonx_times),
                'best_score': max(watsonx_scores),
                'worst_score': min(watsonx_scores)
            }
        
        if openai:
            openai_scores = [r['evaluation']['metrics'].get('overall_score', 0) for r in openai]
            openai_times = [r['execution_time'] for r in openai]
            summary['openai_results'] = {
                'count': len(openai),
                'average_score': sum(openai_scores) / len(openai_scores),
                'average_time': sum(openai_times) / len(openai_times),
                'best_score': max(openai_scores),
                'worst_score': min(openai_scores)
            }
        
        # Comparison
        if watsonx and openai:
            summary['comparison'] = {
                'score_difference': summary['watsonx_results']['average_score'] - summary['openai_results']['average_score'],
                'time_difference': summary['watsonx_results']['average_time'] - summary['openai_results']['average_time']
            }
        
        # Task type breakdown
        summary['task_type_breakdown'] = {}
        for result in successful:
            task_type = result['evaluation']['task_type']
            if task_type not in summary['task_type_breakdown']:
                summary['task_type_breakdown'][task_type] = {
                    'count': 0,
                    'scores': [],
                    'average_score': 0.0
                }
            breakdown = summary['task_type_breakdown'][task_type]
            breakdown['count'] += 1
            score = result['evaluation']['metrics'].get('overall_score', 0)
            breakdown['scores'].append(score)
            breakdown['average_score'] = sum(breakdown['scores']) / len(breakdown['scores'])
        
        return summary


def main():
    """Main benchmarking function."""
    # Prepare scenarios
    dataset_manager = DatasetManager()
    scenario_generator = ScenarioGenerator(dataset_manager)
    scenarios = scenario_generator.generate_all_scenarios()
    
    # Save scenarios
    scenario_dir = Path(__file__).parent / "outputs" / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    save_scenarios(scenarios, scenario_dir / "pdmbench_scenarios.json")
    
    # Run benchmark
    benchmark = MultiLLMBenchmark()
    
    # Run with WatsonX (model_id 15 = granite-3-2-8b-instruct)
    results = benchmark.run_comparative_benchmark(
        scenarios=scenarios[:3],  # Start with 3 scenarios for testing
        watsonx_models=[15],
        openai_models=[]  # Add OpenAI models when integration is ready
    )
    
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)
    print(json.dumps(results['summary'], indent=2))
    
    return results


if __name__ == "__main__":
    main()

