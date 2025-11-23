"""
Optimized Agent System with:
- Context window detection and auto-switching
- Performance optimization (multithreading, Metal)
- Real-time output streaming
- Flexible nested sub-agent architecture
- Heuristics learning
- Step-by-step audit
- Architecture visualization
"""
import sys
import json
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.shared_utils import setup_paths
setup_paths()

from reactxen.utils.model_inference import get_context_length, count_tokens
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

# Try to import Metal Performance Shaders for macOS acceleration
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"


@dataclass
class ModelConfig:
    """Model configuration with context limits."""
    model_id: Union[int, str]
    context_limit: int
    provider: str  # "watsonx" or "openai"
    cost_per_1k_tokens: float = 0.0
    speed_rating: int = 5  # 1-10, higher is faster


# Model configurations with context limits
MODEL_CONFIGS = {
    15: ModelConfig(15, 128000, "watsonx", 0.0, 7),  # granite-3-2-8b-instruct
    "gpt-4": ModelConfig("gpt-4", 8192, "openai", 0.03, 8),
    "gpt-4-turbo": ModelConfig("gpt-4-turbo", 128000, "openai", 0.01, 9),
    "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", 16384, "openai", 0.0015, 10),
}


class ContextWindowManager:
    """Manages context window detection and model switching."""
    
    @staticmethod
    def estimate_tokens(text: str, model_id: Union[int, str] = 15) -> int:
        """Estimate token count for text."""
        try:
            if isinstance(model_id, int):
                model_name = f"model_{model_id}"
            else:
                model_name = model_id
            
            # Use approximate estimation (1 token ≈ 4 characters)
            estimated = len(text) // 4
            
            # Try to get actual count if possible
            try:
                actual = count_tokens(text, model_id=model_id)
                return actual
            except:
                return estimated
        except:
            return len(text) // 4
    
    @staticmethod
    def select_model(prompt_text: str, preferred_model_id: int = 15) -> ModelConfig:
        """Select appropriate model based on context size."""
        estimated_tokens = ContextWindowManager.estimate_tokens(prompt_text, preferred_model_id)
        
        preferred_config = MODEL_CONFIGS.get(preferred_model_id)
        if preferred_config and estimated_tokens < preferred_config.context_limit * 0.8:
            return preferred_config
        
        # Find best model for this context size
        for model_id, config in MODEL_CONFIGS.items():
            if estimated_tokens < config.context_limit * 0.8:
                if config.provider == "openai" and estimated_tokens > 10000:
                    # Prefer OpenAI for large contexts
                    return config
                elif config.provider == "watsonx":
                    return config
        
        # Default to OpenAI for very large contexts
        return MODEL_CONFIGS["gpt-4-turbo"]


class OutputStreamer:
    """Real-time output streaming to both file and terminal."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file = open(file_path, 'w', buffering=1)  # Line buffering
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()
    
    def write(self, text: str):
        """Write text to both file and terminal."""
        print(text, end='', flush=True)
        self.queue.put(text)
    
    def flush(self):
        """Flush buffers."""
        self.file.flush()
    
    def _stream_worker(self):
        """Background worker to write to file."""
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                self.file.write(text)
                self.file.flush()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Stream error: {e}", file=sys.stderr)
    
    def close(self):
        """Close the streamer."""
        self.running = False
        # Drain queue
        while not self.queue.empty():
            try:
                text = self.queue.get_nowait()
                self.file.write(text)
            except queue.Empty:
                break
        self.file.close()


class HeuristicsManager:
    """Manages heuristics learning and common patterns."""
    
    COMMON_HEURISTICS = {
        "tool_no_params": "For tools with no parameters, use empty JSON: {}",
        "tool_one_param": "For tools with one parameter, use: {\"param_name\": \"value\"}",
        "tool_multiple_params": "For tools with multiple parameters, use: {\"param1\": \"value1\", \"param2\": \"value2\"}",
        "reactxen_format": "ReActXen requires: Action: tool_name\\nAction Input: JSON string",
        "json_parsing": "Always ensure JSON is properly escaped in Action Input",
        "data_check": "Always verify data is loaded before processing: use list_datasets and load_dataset first",
        "ground_truth_required": "RUL predictions MUST be verified with verify_rul_predictions tool",
        "avoid_loops": "If same error repeats 3 times, use learning_analyze tool",
        "delegate_complex": "For complex tasks, delegate to specialized sub-agents",
        "check_memory": "Before starting, check memory for past mistakes using memory_context"
    }
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.learned_heuristics = []
    
    def get_heuristics_prompt(self) -> str:
        """Get formatted heuristics for prompt."""
        heuristics = list(self.COMMON_HEURISTICS.values())
        
        # Add learned heuristics
        if self.learned_heuristics:
            heuristics.extend(self.learned_heuristics)
        
        return "\n".join(f"- {h}" for h in heuristics)
    
    def learn_from_mistake(self, error: str, solution: str):
        """Learn new heuristic from mistake."""
        heuristic = f"If {error}, then {solution}"
        if heuristic not in self.learned_heuristics:
            self.learned_heuristics.append(heuristic)


class ArchitectureVisualizer:
    """Visualizes and manages agent architecture."""
    
    @staticmethod
    def create_architecture(question: str, available_tools: List[str], 
                          sub_agents: Dict[str, Any]) -> Dict[str, Any]:
        """Create architecture based on question analysis."""
        # Analyze question to determine needed components
        question_lower = question.lower()
        
        architecture = {
            "root_agent": {
                "tools": available_tools,
                "sub_agents": {}
            }
        }
        
        # Determine which sub-agents are needed
        if any(word in question_lower for word in ["rul", "remaining useful", "life", "cycles"]):
            architecture["root_agent"]["sub_agents"]["rul_predictor"] = {
                "tools": ["execute_python_code", "load_dataset", "verify_rul_predictions"],
                "required": True
            }
        
        if any(word in question_lower for word in ["fault", "classify", "anomaly", "degradation"]):
            architecture["root_agent"]["sub_agents"]["fault_classifier"] = {
                "tools": ["execute_python_code", "load_dataset"],
                "required": True
            }
        
        if any(word in question_lower for word in ["cost", "benefit", "roi", "maintenance cost"]):
            architecture["root_agent"]["sub_agents"]["cost_benefit_analyzer"] = {
                "tools": ["execute_python_code", "load_dataset"],
                "required": True
            }
        
        if any(word in question_lower for word in ["safety", "risk", "policy", "compliance"]):
            architecture["root_agent"]["sub_agents"]["safety_policies"] = {
                "tools": ["execute_python_code", "format_table"],
                "required": True
            }
        
        return architecture
    
    @staticmethod
    def format_architecture(arch: Dict[str, Any]) -> str:
        """Format architecture as readable string."""
        lines = ["Agent Architecture:"]
        lines.append("=" * 60)
        
        root = arch.get("root_agent", {})
        lines.append("Root Agent:")
        lines.append(f"  Tools: {', '.join(root.get('tools', [])[:5])}...")
        
        sub_agents = root.get("sub_agents", {})
        for name, config in sub_agents.items():
            lines.append(f"  └─ {name}:")
            lines.append(f"     Tools: {', '.join(config.get('tools', []))}")
        
        return "\n".join(lines)


class StepAuditor:
    """Audits agent steps for logic gaps and repetitive behavior."""
    
    def __init__(self):
        self.steps = []
        self.repetitions = {}
        self.gaps = []
    
    def add_step(self, step_type: str, action: str, result: str):
        """Add a step to audit."""
        step = {
            "type": step_type,
            "action": action,
            "result": result,
            "timestamp": time.time()
        }
        self.steps.append(step)
        
        # Check for repetitions
        key = f"{step_type}:{action}"
        if key in self.repetitions:
            self.repetitions[key] += 1
            if self.repetitions[key] >= 3:
                self.gaps.append(f"Repetitive action detected: {action} (repeated {self.repetitions[key]} times)")
        else:
            self.repetitions[key] = 1
    
    def get_audit_report(self) -> str:
        """Get audit report."""
        if not self.gaps:
            return "No issues detected in execution flow."
        
        return "Issues detected:\n" + "\n".join(f"- {gap}" for gap in self.gaps)
    
    def check_data_loaded(self) -> bool:
        """Check if data loading steps were performed."""
        data_actions = [s for s in self.steps if "load_dataset" in s.get("action", "").lower()]
        return len(data_actions) > 0


# Import existing modules will be done in optimized_pdmbench_agent.py

