"""
Final Benchmark Execution with Full Output Capture
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.shared_utils import setup_paths
setup_paths()

from multi_llm_benchmark import MultiLLMBenchmark
from datasets_scenarios import DatasetManager, ScenarioGenerator, save_scenarios


def main():
    """Run final benchmark with full output capture."""
    output_file = Path(__file__).parent / "final_agent_output.txt"
    
    print("="*80)
    print("🚀 PDMBench Agentic Implementation - Final Benchmark")
    print("="*80)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PDMBench Agentic Implementation - Final Benchmark\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.flush()
        
        try:
            # Step 1: Prepare scenarios
            print("\n📋 Step 1: Preparing datasets and scenarios...")
            f.write("="*80 + "\n")
            f.write("STEP 1: DATASET AND SCENARIO PREPARATION\n")
            f.write("="*80 + "\n\n")
            
            dataset_manager = DatasetManager()
            scenario_generator = ScenarioGenerator(dataset_manager)
            scenarios = scenario_generator.generate_all_scenarios()
            
            f.write(f"✅ Generated {len(scenarios)} benchmark scenarios\n\n")
            f.write("Scenario Summary:\n")
            for i, scenario in enumerate(scenarios, 1):
                f.write(f"{i}. Type: {scenario['type']}\n")
                f.write(f"   Question: {scenario['question'][:100]}...\n")
                f.write(f"   Expected Outputs: {', '.join(scenario.get('expected_outputs', []))}\n\n")
            f.flush()
            
            # Save scenarios
            scenario_dir = Path(__file__).parent / "outputs" / "scenarios"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            save_scenarios(scenarios, scenario_dir / "pdmbench_scenarios.json")
            f.write(f"✅ Saved scenarios to {scenario_dir / 'pdmbench_scenarios.json'}\n\n")
            f.flush()
            
            # Step 2: Run benchmark
            print("\n📊 Step 2: Running benchmark...")
            f.write("="*80 + "\n")
            f.write("STEP 2: BENCHMARK EXECUTION\n")
            f.write("="*80 + "\n\n")
            f.flush()
            
            benchmark = MultiLLMBenchmark()
            
            # Run with 2 scenarios for initial test (can be increased)
            test_scenarios = scenarios[:2]
            f.write(f"Running {len(test_scenarios)} test scenarios with WatsonX (model_id=15)...\n\n")
            f.flush()
            
            # Capture stdout/stderr during benchmark
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                start_time = time.time()
                results = benchmark.run_comparative_benchmark(
                    scenarios=test_scenarios,
                    watsonx_models=[15],
                    openai_models=[]
                )
                execution_time = time.time() - start_time
            
            # Write captured output
            f.write("=== Benchmark Execution Output ===\n")
            f.write(stdout_capture.getvalue())
            f.write("\n=== Error Output ===\n")
            f.write(stderr_capture.getvalue())
            f.write("\n")
            f.flush()
            
            # Step 3: Write results
            f.write("="*80 + "\n")
            f.write("STEP 3: BENCHMARK RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("Summary:\n")
            f.write(json.dumps(results['summary'], indent=2) + "\n\n")
            f.flush()
            
            f.write("Detailed Results:\n")
            for i, result in enumerate(results['results'], 1):
                f.write(f"\n--- Result {i} ---\n")
                f.write(f"Task Type: {result['task']['type']}\n")
                f.write(f"Question: {result['task']['question'][:150]}...\n")
                f.write(f"Success: {result.get('success', False)}\n")
                
                if result.get('success'):
                    eval_metrics = result['evaluation']['metrics']
                    f.write(f"Overall Score: {eval_metrics.get('overall_score', 0):.2f}/100\n")
                    f.write(f"Execution Time: {result['execution_time']:.2f}s\n")
                    f.write(f"Completeness: {eval_metrics.get('completeness', 0):.2%}\n")
                    f.write(f"Ground Truth Verified: {eval_metrics.get('ground_truth_verified', False)}\n")
                    f.write(f"Result Length: {len(result.get('result', ''))} chars\n")
                    f.write(f"\nResult Preview:\n{result.get('result', '')[:500]}...\n")
                else:
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    if 'traceback' in result:
                        f.write(f"\nTraceback:\n{result['traceback'][:1000]}...\n")
                f.write("\n")
                f.flush()
            
            # Step 4: Performance analysis
            f.write("="*80 + "\n")
            f.write("STEP 4: PERFORMANCE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            successful = [r for r in results['results'] if r.get('success', False)]
            if successful:
                scores = [r['evaluation']['metrics'].get('overall_score', 0) for r in successful]
                avg_score = sum(scores) / len(scores)
                f.write(f"Average Score: {avg_score:.2f}/100\n")
                f.write(f"Best Score: {max(scores):.2f}/100\n")
                f.write(f"Worst Score: {min(scores):.2f}/100\n")
                f.write(f"Success Rate: {len(successful)}/{len(results['results'])} ({len(successful)/len(results['results'])*100:.1f}%)\n")
                
                if avg_score >= 80.0:
                    f.write("\n✅ PERFORMANCE TARGET MET: Average score >= 80.0\n")
                else:
                    f.write(f"\n⚠️  PERFORMANCE TARGET NOT MET: Average score {avg_score:.2f} < 80.0\n")
            else:
                f.write("❌ No successful runs to analyze\n")
            
            f.write(f"\nTotal Execution Time: {execution_time:.2f}s\n")
            f.write(f"Results saved to: {results.get('results_file', 'N/A')}\n")
            f.flush()
            
            f.write("\n" + "="*80 + "\n")
            f.write("BENCHMARK COMPLETE\n")
            f.write("="*80 + "\n")
            f.write(f"Completed at: {datetime.now().isoformat()}\n")
            
            print(f"\n✅ Benchmark complete! Results saved to {output_file}")
            if successful:
                avg_score = sum([r['evaluation']['metrics'].get('overall_score', 0) for r in successful]) / len(successful)
                print(f"📊 Average Score: {avg_score:.2f}/100")
                print(f"✅ Success Rate: {len(successful)}/{len(results['results'])}")
            
        except Exception as e:
            import traceback
            error_msg = f"\n❌ ERROR: {str(e)}\n\n{traceback.format_exc()}\n"
            f.write(error_msg)
            print(error_msg)
            raise


if __name__ == "__main__":
    main()

