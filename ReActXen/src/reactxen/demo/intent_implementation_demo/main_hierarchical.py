"""
Main execution script for hierarchical agent system.
"""
from pathlib import Path
from datetime import datetime
from shared_utils import setup_paths
from root_agent_implementation import run_root_agent
from load_data import list_available_datasets

setup_paths()

# Try to import optional dependencies
try:
    from load_data import download_and_save_dataset
except ImportError:
    def download_and_save_dataset(*args, **kwargs):
        return {"status": "error", "message": "Download function not available"}

try:
    from benchmark_utils import ModelBenchmark
except ImportError:
    class ModelBenchmark:
        def __init__(self, *args, **kwargs):
            self.benchmark_results = []
        def record_model_run(self, **kwargs):
            result = {"performance_score": 0, "success": False, **kwargs}
            self.benchmark_results.append(result)
            return result
        def get_best_model(self):
            if not self.benchmark_results:
                return None
            return max(self.benchmark_results, key=lambda x: x.get("performance_score", 0))
        def export_results(self, **kwargs):
            return None


def main():
    """Main function to run the hierarchical agent system."""
    print("="*70)
    print("🏭 HIERARCHICAL AGENT SYSTEM - INTENT-BASED INDUSTRIAL AUTOMATION")
    print("="*70)
    print("\nThis system uses a hierarchical agent structure:")
    print("  • Root Agent - Orchestrates all sub-agents")
    print("  • Data Scientist Agent - ML model training and evaluation")
    print("  • Predictive Maintenance Agent - RUL prediction and risk assessment")
    print("  • Cost-Benefit Analysis Agent - Cost estimation and ROI analysis")
    print("  • Safety/Policy Agent - Safety protocols and compliance")
    print("="*70)
    
    # Step 1: Check datasets
    print("\n📦 STEP 1: Checking Available Datasets")
    print("-"*70)
    
    available_datasets = list_available_datasets()
    
    if not available_datasets:
        print("❌ No datasets found. Downloading test datasets...")
        test_datasets = [
            "submission096/PlanetaryPdM",
            "submission096/CWRU",
            "submission096/Azure"
        ]
        
        for dataset_path in test_datasets:
            dataset_name = dataset_path.split("/")[-1]
            print(f"\nDownloading: {dataset_name}")
            result = download_and_save_dataset(dataset_path, force_download=False)
            print(f"Status: {result.get('status', 'unknown')}")
        
        available_datasets = list_available_datasets()
    
    if available_datasets:
        print(f"✅ Found {len(available_datasets)} available dataset(s)")
        for name in available_datasets:
            print(f"   • {name}")
    else:
        print("⚠️  No datasets available. Please download datasets first.")
        return
    
    # Step 2: Setup output directory
    print("\n📁 STEP 2: Setting Up Output Directory")
    print("-"*70)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    print(f"✅ Output directory: {output_dir}")
    
    # Step 3: Initialize benchmark
    print("\n📊 STEP 3: Initializing Benchmark")
    print("-"*70)
    
    benchmark = ModelBenchmark(output_dir)
    
    # Step 4: Define question
    question = "We should focus on equipment that is running on fumes. Which equipment from the loaded dataset are likely to give out in the next 20 cycles? Provide me a list of equipment IDs with safety recommendations and cost estimates."
    
    print("\n❓ STEP 4: Defining Question")
    print("-"*70)
    print(f"Question: {question}")
    
    # Step 5: Run root agent
    print("\n🤖 STEP 5: Running Root Agent")
    print("-"*70)
    
    models_to_test = [
        {"name": "Granite-3.2-8B-Instruct", "model_id": 15},
    ]
    
    best_result = None
    
    for model_config in models_to_test:
        model_name = model_config["name"]
        model_id = model_config["model_id"]
        
        print(f"\n🧪 Testing Model: {model_name} (ID: {model_id})")
        print("-"*70)
        
        try:
            results = run_root_agent(
                question=question,
                react_llm_model_id=model_id,
                max_steps=35,  # Increased to allow for all steps
                output_dir=output_dir,
                debug=True
            )
            
            benchmark_result = benchmark.record_model_run(
                model_name=model_name,
                model_type="watsonx",
                model_id=str(model_id),
                metrics=results.get("metrics", {}),
                execution_time=results.get("execution_time", 0),
                success=results.get("success", False),
                error_message=results.get("error"),
                steps_taken=results.get("steps_taken"),
                final_answer=results.get("final_answer"),
            )
            
            if results.get("success") and (best_result is None or 
                benchmark_result["performance_score"] > best_result["performance_score"]):
                best_result = benchmark_result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"❌ Error with {model_name}: {error_msg}")
            print(f"Error details: {traceback.format_exc()}")
            benchmark.record_model_run(
                model_name=model_name,
                model_type="watsonx",
                model_id=str(model_id),
                execution_time=0,
                success=False,
                error_message=error_msg,
            )
    
    # Step 6: Export results
    print("\n📊 STEP 6: Benchmark Results")
    print("-"*70)
    
    best_model = benchmark.get_best_model()
    if best_model:
        print(f"🏆 Best Model: {best_model['model_name']} (Score: {best_model['performance_score']:.2f}/100)")
        print(f"   Execution Time: {best_model['execution_time']:.2f}s")
        print(f"   Steps Taken: {best_model.get('steps_taken', 'N/A')}")
        print(f"   Success: {'✅' if best_model['success'] else '❌'}")
    
    benchmark.export_results(format="json")
    benchmark.export_results(format="markdown")
    benchmark.export_results(format="txt")
    
    print("\n" + "="*70)
    print("🎉 HIERARCHICAL AGENT SYSTEM EXECUTION COMPLETED")
    print("="*70)
    print(f"\n📁 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
