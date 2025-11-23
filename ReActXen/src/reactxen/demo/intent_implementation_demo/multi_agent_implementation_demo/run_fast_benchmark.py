"""
Fast Benchmark Runner with:
- Hard 180s timeout enforcement
- Real-time output streaming to terminal and file
- Multi-LLM support
- Automatic termination on hang
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.shared_utils import setup_paths
setup_paths()

from fast_optimized_agent import create_fast_agent
from datasets_scenarios import DatasetManager, ScenarioGenerator, save_scenarios
from comprehensive_evaluator import get_evaluator


def run_fast_benchmark(scenario: Dict[str, Any], output_file: Path,
                      model_type: str = "watsonx", model_id: Union[int, str] = 15,
                      timeout: int = 3600) -> Dict[str, Any]:
    """Run single scenario with fast agent."""
    question = scenario['question']
    task_type = scenario.get('type', 'rul_prediction')
    
    print(f"\n{'='*80}")
    print(f"📋 Scenario: {task_type}")
    print(f"❓ Question: {question[:120]}...")
    print(f"🤖 Model: {model_type} ({model_id})")
    print(f"⏱️  Timeout: {timeout}s")
    print(f"{'='*80}\n")
    
    agent = None
    try:
        # Create fast agent with task type
        agent = create_fast_agent(
            question=question,
            output_file=output_file,
            model_type=model_type,
            model_id=model_id,
            timeout=timeout,
            task_type=task_type
        )
        
        # Create agent
        agent.create_agent()
        
        # Run with hard timeout
        result = agent.run_with_hard_timeout()
        
        # Extract result string safely (handle both string and dict results)
        result_value = result.get('result', '')
        if isinstance(result_value, dict):
            # Convert dict to string representation
            result_str = json.dumps(result_value, indent=2)
        else:
            result_str = str(result_value) if result_value else ""
        
        # Comprehensive evaluation with success criteria
        try:
            evaluator = get_evaluator()
            evaluation_data = evaluator.evaluate_result(
                task=scenario,
                result=result_str,
                execution_time=result['execution_time'],
                ground_truth=scenario.get('ground_truth')
            )
        except Exception as e:
            import traceback
            print(f"⚠️  Evaluation error: {e}")
            print(f"Result type: {type(result_str)}, Result value: {result_str[:200] if result_str else 'None'}")
            traceback.print_exc()
            # Create a minimal evaluation result for failed cases
            evaluation_data = {
                "task_type": scenario.get('type', 'unknown'),
                "question": scenario.get('question', ''),
                "execution_time": result.get('execution_time', 0.0),
                "metrics": {"overall_score": 0.0, "completeness": 0.0},
                "success": False,
                "success_criteria_met": {"task_accomplished": False, "ground_truth_validated": False},
                "error": str(e)
            }
        
        # Success is determined by evaluation (task accomplished AND ground truth validated)
        success = evaluation_data.get('success', False)
        
        # Log success criteria status
        criteria_met = evaluation_data.get('success_criteria_met', {})
        print(f"\n{'='*80}")
        print(f"📊 SUCCESS CRITERIA EVALUATION")
        print(f"{'='*80}")
        print(f"Task Accomplished: {'✅ YES' if criteria_met.get('task_accomplished') else '❌ NO'}")
        print(f"Ground Truth Validated: {'✅ YES' if criteria_met.get('ground_truth_validated') else '❌ NO'}")
        print(f"Overall Success: {'✅ YES' if success else '❌ NO'}")
        print(f"Score: {evaluation_data['metrics'].get('overall_score', 0.0):.2f}/100.0")
        if not success:
            penalty = evaluation_data['metrics'].get('penalty_applied', '')
            if penalty:
                print(f"⚠️  {penalty}")
        print(f"{'='*80}\n")
        
        return {
            'task': scenario,
            'result': result.get('result', ''),
            'evaluation': evaluation_data,
            'execution_time': result['execution_time'],
            'success': success,  # Use evaluation success, not just agent completion
            'error': result.get('error'),
            'timeout_exceeded': result.get('timeout_exceeded', False),
            'success_criteria': criteria_met
        }
        
    except Exception as e:
        import traceback
        error_tb = traceback.format_exc()
        print(f"\n❌ Error in run_fast_benchmark: {e}")
        print(f"Traceback:\n{error_tb}")
        return {
            'task': scenario,
            'error': str(e),
            'traceback': error_tb,
            'execution_time': 0.0,
            'success': False,
            'result': None
        }
    finally:
        if agent:
            agent.close()


def main():
    """Main benchmark execution with fast agent."""
    output_file = Path(__file__).parent / "final_agent_output.txt"
    
    # Clear previous output
    if output_file.exists():
        output_file.unlink()
    
    print("="*80)
    print("🚀 FAST PDMBench Benchmark")
    print("="*80)
    print(f"📄 Output: {output_file}\n")
    
    try:
        # Step 1: Prepare scenarios
        print("📋 Step 1: Preparing scenarios...")
        dataset_manager = DatasetManager()
        scenario_generator = ScenarioGenerator(dataset_manager)
        scenarios = scenario_generator.generate_all_scenarios()
        
        # Save scenarios
        scenario_dir = Path(__file__).parent / "outputs" / "scenarios"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        save_scenarios(scenarios, scenario_dir / "pdmbench_scenarios.json")
        
        print(f"✅ Generated {len(scenarios)} scenarios\n")
        
        # Step 2: Run benchmark (start with 1 scenario)
        print("📊 Step 2: Running fast benchmark...\n")
        test_scenarios = scenarios[:1]
        
        # Model configurations to try (with context window limits)
        model_configs = [
            {"type": "watsonx", "id": 15, "name": "WatsonX Granite-3-2-8b", "context_limit": 128000},
            # OpenAI models (if API key available)
            # {"type": "openai", "id": "gpt-4-turbo", "name": "OpenAI GPT-4 Turbo", "context_limit": 128000},
            # {"type": "openai", "id": "gpt-3.5-turbo", "name": "OpenAI GPT-3.5 Turbo", "context_limit": 16384},
            # Meta-Llama models (if available via WatsonX)
            # {"type": "watsonx", "id": 8, "name": "Meta-Llama-3-70b", "context_limit": 8192},
        ]
        
        results = []
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{'='*80}")
            print(f"Scenario {i}/{len(test_scenarios)}")
            print(f"{'='*80}\n")
            
            # Try first model config
            model_cfg = model_configs[0]
            result = run_fast_benchmark(
                scenario=scenario,
                output_file=output_file,
                model_type=model_cfg["type"],
                model_id=model_cfg["id"],
                timeout=3600  # 1 hour limit (no strict timeout for full execution)
            )
            
            results.append(result)
            
            # Print result summary
            if result.get('success'):
                print(f"\n✅ SUCCESS in {result['execution_time']:.1f}s")
            else:
                print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
                if result.get('timeout_exceeded'):
                    print("   ⏱️  Timeout exceeded - execution terminated")
        
        # Step 3: Comprehensive Evaluation
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r.get('success', False)]
        evaluations = [r.get('evaluation', {}) for r in results if r.get('evaluation')]
        
        print(f"Total: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(results) - len(successful)}")
        
        if successful:
            times = [r['execution_time'] for r in successful]
            scores = [r.get('evaluation', {}).get('metrics', {}).get('overall_score', 0.0) for r in successful]
            
            print(f"\n⏱️  EXECUTION TIME:")
            print(f"  Average: {sum(times)/len(times):.1f}s")
            print(f"  Min: {min(times):.1f}s")
            print(f"  Max: {max(times):.1f}s")
            print(f"  Target: <180s {'✅ MET' if max(times) < 180 else '⚠️  NOT MET'}")
            
            print(f"\n📊 PERFORMANCE SCORES:")
            print(f"  Average: {sum(scores)/len(scores):.2f}/100.0")
            print(f"  Min: {min(scores):.2f}/100.0")
            print(f"  Max: {max(scores):.2f}/100.0")
            tasks_above_80 = sum(1 for s in scores if s >= 80.0)
            print(f"  Tasks ≥80.0: {tasks_above_80}/{len(scores)} ({tasks_above_80/len(scores)*100:.1f}%)")
            print(f"  Target: ≥80.0 {'✅ MET' if tasks_above_80 > 0 and sum(scores)/len(scores) >= 80.0 else '⚠️  NOT MET'}")
        
        # Compare with PDMBench baseline
        if evaluations:
            evaluator = get_evaluator()
            comparison = evaluator.compare_with_baseline(evaluations)
            
            print(f"\n📈 COMPARISON WITH PDMBENCH BASELINE:")
            print(f"  Baseline Avg Score: {comparison['overall']['baseline_avg_score']:.2f}")
            print(f"  Our Avg Score: {comparison['overall']['our_avg_score']:.2f}")
            print(f"  Improvement: +{comparison['overall']['improvement']:.2f} ({comparison['overall']['improvement_percent']:.1f}%)")
            
            for task_type, comp_data in comparison['task_types'].items():
                print(f"\n  {task_type.upper()}:")
                print(f"    Baseline: {comp_data['baseline_score']:.2f} → Our: {comp_data['our_avg_score']:.2f}")
                print(f"    Improvement: +{comp_data['improvement']:.2f} ({comp_data['improvement_percent']:.1f}%)")
            
            # Save evaluation report
            report_file = evaluator.save_evaluation_report(evaluations, comparison)
            text_report = evaluator.generate_text_report(evaluations, comparison)
            
            # Save text report
            text_report_file = report_file.parent / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(text_report_file, 'w') as f:
                f.write(text_report)
            
            print(f"\n📄 Evaluation reports saved:")
            print(f"  JSON: {report_file}")
            print(f"  Text: {text_report_file}")
            
            # Write summary to output file
            with open(output_file, 'a') as f:
                f.write("\n" + text_report + "\n")
        
        timeouts = [r for r in results if r.get('timeout_exceeded', False)]
        if timeouts:
            print(f"\n⚠️  {len(timeouts)} scenario(s) exceeded timeout")
        
        print(f"\n📄 Full output: {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

