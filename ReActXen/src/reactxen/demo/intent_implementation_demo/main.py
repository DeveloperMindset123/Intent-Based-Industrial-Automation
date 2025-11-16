"""Main entry point with adaptive retry (default) and optional benchmarking."""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from agent_implementation import agent_config
from benchmark import run_with_retry, benchmark_models

def main(benchmark: bool = False, model_ids: list = None):
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"agent_output_{timestamp}.txt"
    
    print("="*70)
    print("🚀 Simplified Agentic RUL Prediction (WatsonX Only)")
    print("="*70)
    print(f"📝 Output will be saved to: {output_file}\n")
    
    # Capture output
    class OutputLogger:
        def __init__(self, file_path):
            self.file = open(file_path, 'w', encoding='utf-8')
            self.stdout = sys.stdout
        
        def write(self, text):
            self.stdout.write(text)
            self.file.write(text)
            self.file.flush()
        
        def flush(self):
            self.stdout.flush()
            self.file.flush()
        
        def close(self):
            self.file.close()
    
    logger = OutputLogger(output_file)
    sys.stdout = logger
    
    try:
        if benchmark:
            # Benchmark multiple models (auto-detects from WatsonX)
            benchmark_file = output_dir / f"benchmark_{timestamp}.json"
            print("📊 Running benchmark across available models...\n")
            benchmark_result = benchmark_models(agent_config.copy(), model_ids)
            
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_result, f, indent=2, default=str)
            
            print(f"\n📄 Benchmark results saved to: {benchmark_file}")
            result = benchmark_result
        else:
            # Default: Run with retry logic to ensure task completion
            print("🔄 Running with adaptive retry logic (default)...\n")
            result = run_with_retry(agent_config.copy())
        
        print("\n" + "="*70)
        print("✅ Execution completed")
        print("="*70)
        
        return result
    finally:
        sys.stdout = logger.stdout
        logger.close()
        print(f"\n📄 Full output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agentic RUL prediction")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Benchmark available models from WatsonX (default: single model with retry)")
    parser.add_argument("--models", nargs="+", type=int, default=None,
                       help="Specific model IDs to benchmark (default: auto-detect from WatsonX)")
    
    args = parser.parse_args()
    main(benchmark=args.benchmark, model_ids=args.models)
