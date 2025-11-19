"""Simple benchmarking with adaptive retry logic."""
import os, json, sys, time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

reactxen_src = Path(__file__).parent.parent.parent.parent
if str(reactxen_src) not in sys.path:
    sys.path.insert(0, str(reactxen_src))

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from tools_logic import WatsonXAPIState, watsonx_api

load_dotenv(override=False)

# Sync OpenAI API key from .env to credentials.json
credentials_file = Path(__file__).parent / "credentials.json"
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key:
    with open(credentials_file, 'w') as f:
        json.dump({"openai_api_key": openai_key}, f, indent=2)
elif credentials_file.exists():
    with open(credentials_file, 'r') as f:
        creds = json.load(f)
        if creds.get("openai_api_key") and creds["openai_api_key"] != "your_openai_api_key_here":
            os.environ["OPENAI_API_KEY"] = creds["openai_api_key"]

def calculate_additional_steps(metrics: Dict, base_steps: int, actual_steps_used: Optional[int] = None) -> int:
    """Calculate additional steps needed based on agent's progress."""
    status = metrics.get("status", "Unknown")
    if actual_steps_used is None:
        pr_info = metrics.get("per_round_info", [{}])
        actual_steps_used = pr_info[-1].get("step", metrics.get("max_steps", base_steps)) if pr_info else base_steps
    if actual_steps_used < base_steps * 0.8:
        return max(5, base_steps - actual_steps_used)
    if "Partially" in status:
        return 10
    elif "Not Accomplished" in status:
        return 15 if metrics.get("max_steps", base_steps) >= base_steps else 10
    return 0

def run_with_retry(agent_config: Dict, max_retries: int = 3, timeout_seconds: int = 300) -> Dict:
    """Run agent with adaptive step increases if not accomplished. Includes timeout and error handling."""
    base_steps = agent_config.get("max_steps", 10)
    current_steps = base_steps
    results = []
    
    for attempt in range(max_retries):
        agent_config["max_steps"] = current_steps
        
        print(f"\n🔄 Attempt {attempt + 1}/{max_retries} with {current_steps} steps...")
        
        start_time = time.time()
        agent = None
        execution_time = 0
        error_occurred = False
        error_message = None
        
        try:
            agent = create_reactxen_agent(**agent_config)
            agent.run()
            execution_time = time.time() - start_time
            
            # Check if agent got stuck (execution time too long relative to steps)
            if execution_time > timeout_seconds:
                print(f"⚠️  Agent execution took {execution_time:.1f}s (exceeded timeout limit of {timeout_seconds}s).")
                error_occurred = True
                error_message = f"Execution time exceeded {timeout_seconds}s"
            
        except KeyboardInterrupt:
            execution_time = time.time() - start_time
            error_occurred = True
            error_message = "Interrupted by user"
            print(f"⚠️  {error_message}")
            raise  # Re-raise keyboard interrupt
        except Exception as e:
            execution_time = time.time() - start_time
            error_occurred = True
            error_message = f"Error: {str(e)}"
            print(f"❌ {error_message}")
        
        metrics, status, actual_steps_used = {}, "Error", None
        if agent:
            try:
                metrics = agent.export_benchmark_metric()
                status = metrics.get("status", "Error")
                actual_steps_used = agent.step_n if hasattr(agent, 'step_n') else (metrics.get("per_round_info", [{}])[-1].get("step", current_steps) if metrics.get("per_round_info") else current_steps)
                if actual_steps_used and actual_steps_used > current_steps * 0.5 and hasattr(agent, 'scratchpad') and agent.scratchpad:
                    sp = agent.scratchpad.lower()
                    if sp.count("final answer") >= 3 and (("action input" in sp and "none" in sp) or sp.count("incorrectly formatted") >= 2):
                        error_occurred, error_message = True, "Agent stuck: repeatedly trying Final Answer with incorrect formatting"
                        print(f"⚠️  {error_message}")
                    elif len(sp.split("Action")) >= 5 and len(set([a.strip()[:50] for a in sp.split("Action")[-10:] if a.strip()])) <= 2:
                        error_occurred, error_message = True, "Agent stuck in repetitive action loop"
                        print(f"⚠️  {error_message}")
            except Exception as e:
                print(f"⚠️  Could not extract metrics: {e}")
                status = "Error extracting metrics"
        else:
            status = "Failed to create agent"
        
        if error_occurred:
            result = {"attempt": attempt + 1, "steps": current_steps, "steps_used": actual_steps_used or 0,
                     "status": f"Error: {error_message}", "accomplished": False, "execution_time": execution_time,
                     "metrics": metrics, "error": True}
            results.append(result)
            print(f"❌ Model failed: {error_message}. Skipping remaining attempts.")
            return result
        
        accomplished = status == "Accomplished"
        if not accomplished and actual_steps_used and actual_steps_used >= current_steps * 0.9 and hasattr(agent, 'answer') and agent.answer:
            accomplished, status = True, "Accomplished (used most steps)"
        
        result = {"attempt": attempt + 1, "steps": current_steps, "steps_used": actual_steps_used or current_steps,
                 "status": status, "accomplished": accomplished, "execution_time": execution_time,
                 "metrics": metrics, "error": False}
        results.append(result)
        
        if accomplished:
            print(f"✅ Task accomplished in {attempt + 1} attempt(s) with {current_steps} steps (used {actual_steps_used or current_steps})")
            return result
        
        additional_steps = calculate_additional_steps(metrics, base_steps, actual_steps_used)
        current_steps += additional_steps
        print(f"⚠️  Not accomplished. Status: {status}. Used {actual_steps_used or current_steps}/{current_steps} steps. Adding {additional_steps} steps for next attempt.")
    
    print(f"❌ Task not accomplished after {max_retries} attempts")
    return results[-1]

def get_available_models() -> List[Dict]:
    """Get available models from WatsonX and filter by API key requirements."""
    try:
        instance = WatsonXAPIState.get_instance()
        if not instance:
            api_key = os.environ.get("WATSONX_APIKEY")
            project_id = os.environ.get("WATSONX_PROJECT_ID")
            project_url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com/")
            if api_key and project_id:
                instance = watsonx_api(api_key, project_id, project_url)
                WatsonXAPIState.set_instance(instance)
            else:
                return []
        models = instance.get_chat_models_list()
        if not isinstance(models, list):
            return []
        try:
            from reactxen.utils.model_inference import modelset
        except ImportError:
            modelset = []
        allowed = ["ibm/", "openai/", "meta-llama/", "mistralai/"]
        filtered = []
        for model_name in models:
            if any(model_name.startswith(p) for p in allowed) and modelset:
                try:
                    filtered.append({"name": model_name, "id": modelset.index(model_name)})
                except ValueError:
                    continue
        return filtered
    except Exception as e:
        print(f"⚠️  Error getting models: {e}")
        return []

def benchmark_models(agent_config: Dict, model_ids: Optional[List[int]] = None) -> Dict:
    """Benchmark multiple models. If model_ids not provided, uses available WatsonX models."""
    if model_ids is None:
        available_models = get_available_models()
        model_ids = [15] if not available_models else [m["id"] for m in available_models[:5]]
        if available_models:
            print(f"📋 Found {len(available_models)} available models. Benchmarking {len(model_ids)} models.")
    
    print(f"\n📊 Benchmarking {len(model_ids)} model(s)...\n" + "="*70)
    best_result, all_results = None, []
    
    for model_id in model_ids:
        print(f"\n🧪 Testing Model ID: {model_id}\n" + "-"*70)
        agent_config["react_llm_model_id"] = model_id
        agent_config["reflect_llm_model_id"] = model_id
        try:
            result = run_with_retry(agent_config.copy(), timeout_seconds=300)
            result["model_id"] = model_id
            all_results.append(result)
            if result.get("error"):
                print(f"⚠️  Model {model_id} failed or got stuck. Moving to next model.")
                continue
            if result["accomplished"] and (best_result is None or result["execution_time"] < best_result["execution_time"]):
                best_result = result
        except Exception as e:
            print(f"❌ Error testing model {model_id}: {e}")
            all_results.append({"model_id": model_id, "status": f"Error: {str(e)}", "accomplished": False,
                              "error": True, "execution_time": 0, "steps": 0, "steps_used": 0})
    
    print("\n" + "="*70 + "\n📊 BENCHMARK SUMMARY\n" + "="*70)
    for r in all_results:
        icon = "⚠️" if r.get("error") else ("✅" if r["accomplished"] else "❌")
        print(f"{icon} Model {r['model_id']}: {r['status']} ({r.get('steps', 0)} steps, {r.get('execution_time', 0):.1f}s)")
    if best_result:
        print(f"\n🏆 Best Model: {best_result['model_id']} ({best_result['steps']} steps, {best_result['execution_time']:.1f}s)")
    else:
        print("\n⚠️  No model accomplished the task")
    return {"best_result": best_result, "all_results": all_results}