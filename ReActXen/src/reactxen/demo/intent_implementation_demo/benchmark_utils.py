"""Benchmarking utilities for comparing different models."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class ModelBenchmark:
    """Benchmark different models and store results"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.benchmark_results = []
    
    def record_model_run(self, model_name: str, model_type: str, model_id: Optional[str] = None,
                        metrics: Optional[Dict[str, Any]] = None, execution_time: Optional[float] = None,
                        success: bool = True, error_message: Optional[str] = None,
                        steps_taken: Optional[int] = None, final_answer: Optional[str] = None):
        """Record a model benchmark run"""
        result = {"model_name": model_name, "model_type": model_type, "model_id": model_id,
                 "timestamp": datetime.now().isoformat(), "execution_time": execution_time or 0.0,
                 "success": success, "error_message": error_message, "steps_taken": steps_taken,
                 "metrics": metrics or {}, "final_answer": final_answer}
        self.benchmark_results.append(result)
        return result
    
    def calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """Calculate a performance score for a model run"""
        score = 50.0 if result["success"] else 0.0
        if result["execution_time"] > 0:
            score += max(0, 20 * (1 - result["execution_time"] / 600))
        if result["success"] and result["steps_taken"]:
            score += max(0, 15 * (1 - result["steps_taken"] / 20))
        if result["metrics"]:
            m = result["metrics"]
            if "accuracy" in m:
                score += 15 * m["accuracy"]
            elif "mae" in m:
                score += max(0, 15 * (1 - m["mae"] / 100))
            elif "rmse" in m:
                score += max(0, 15 * (1 - m["rmse"] / 100))
        return min(100.0, score)
    
    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """Get the best performing model based on performance score"""
        if not self.benchmark_results:
            return None
        scored = [(r, self.calculate_performance_score(r)) for r in self.benchmark_results]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        best["performance_score"] = scored[0][1]
        return best
    
    def export_results(self, format: str = "json") -> Path:
        """Export benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for r in self.benchmark_results:
            r["performance_score"] = self.calculate_performance_score(r)
        sorted_results = sorted(self.benchmark_results, key=lambda x: x["performance_score"], reverse=True)
        best_model = self.get_best_model()
        
        if format == "json":
            output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump({"benchmark_timestamp": datetime.now().isoformat(), "total_runs": len(self.benchmark_results),
                          "best_model": best_model, "all_results": sorted_results}, f, indent=2)
        elif format == "markdown":
            output_file = self.output_dir / f"benchmark_results_{timestamp}.md"
            with open(output_file, 'w') as f:
                f.write(f"# Model Benchmark Results\n\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                if best_model:
                    f.write(f"## 🏆 Best Model: {best_model['model_name']} (Score: {best_model['performance_score']:.2f}/100)\n\n")
                f.write("| Rank | Model | Type | Score | Time (s) | Steps | Success |\n|------|-------|------|-------|----------|-------|----------|\n")
                for i, r in enumerate(sorted_results, 1):
                    f.write(f"| {i} | {r['model_name']} | {r['model_type']} | {r['performance_score']:.2f} | "
                           f"{r['execution_time']:.2f} | {r.get('steps_taken', 'N/A')} | {'✅' if r['success'] else '❌'} |\n")
        else:
            output_file = self.output_dir / f"benchmark_results_{timestamp}.txt"
            with open(output_file, 'w') as f:
                f.write("="*70 + "\nMODEL BENCHMARK RESULTS\n" + "="*70 + "\n\n")
                if best_model:
                    f.write(f"🏆 BEST: {best_model['model_name']} (Score: {best_model['performance_score']:.2f}/100)\n\n")
                for i, r in enumerate(sorted_results, 1):
                    f.write(f"{i}. {r['model_name']} - Score: {r['performance_score']:.2f}/100, "
                           f"Time: {r['execution_time']:.2f}s, Success: {'✅' if r['success'] else '❌'}\n")
        return output_file
