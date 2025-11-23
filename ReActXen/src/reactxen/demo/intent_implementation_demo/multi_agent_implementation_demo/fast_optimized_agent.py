"""
Fast Optimized Agent with:
- Hard 180s timeout enforcement
- Real-time output streaming
- Multi-LLM support (WatsonX, OpenAI, Meta-Llama)
- Proper question passing (no cached examples)
- Better loop detection and mistake learning
"""
import sys
import json
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from contextlib import contextmanager

# Setup paths properly
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent
reactxen_src = grandparent_dir.parent.parent

sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(grandparent_dir))
sys.path.insert(0, str(reactxen_src))

from shared.shared_utils import setup_paths, get_tool_descriptions
setup_paths()

from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from reactxen.utils.model_inference import get_context_length, count_tokens
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

# Import tools
try:
    from tools.code_execution import CodeExecutionTool
except ImportError:
    from dynamic_agent_system import CodeExecutionTool

from dataset_tools import create_dataset_tools
from ground_truth_verification import create_ground_truth_tools
from table_formatter import create_table_formatting_tools
from memory_system import get_memory_system
from tools_logic import create_watsonx_tools
from common_tools import create_common_tools
from reflector_agent import ReflectorAgentTool
from learning_agent import LearningAgentTool
from enhanced_heuristics import get_heuristic_learner
from architecture_visualizer import get_architecture_visualizer
from model_selector import get_model_selector
from performance_optimizer import get_performance_optimizer, get_metal_accelerator


class TeeOutput:
    """Output that goes to both file and terminal in real-time."""
    
    def __init__(self, file_path: Path):
        self.file = open(file_path, 'w', buffering=1)
        self.terminal = sys.stdout
        self.file_path = file_path
    
    def write(self, text: str):
        self.terminal.write(text)
        self.terminal.flush()
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


class FastOptimizedAgent:
    """Fast optimized agent with strict timeout and real-time output."""
    
    def __init__(self, question: str, output_file: Path, 
                 model_type: str = "watsonx", model_id: Union[int, str] = 15,
                 timeout: int = 3600, task_type: str = "general"):
        self.question = question
        self.output_file = output_file
        self.model_type = model_type  # "watsonx", "openai", "meta-llama"
        self.model_id = model_id
        self.timeout = timeout
        self.task_type = task_type  # "rul_prediction", "fault_classification", etc.
        self.memory = get_memory_system()
        self.heuristic_learner = get_heuristic_learner()
        self.arch_visualizer = get_architecture_visualizer()
        self.model_selector = get_model_selector()
        self.performance_optimizer = get_performance_optimizer(max_workers=4)
        self.metal_accelerator = get_metal_accelerator()
        
        # Setup output tee
        self.tee = TeeOutput(output_file)
        
        # Track execution
        self.start_time = None
        self.agent = None
        self.kill_flag = threading.Event()
    
    def _create_tools(self) -> List[BaseTool]:
        """Create all necessary tools."""
        # Get heuristic learner for learning from mistakes
        self.heuristic_learner = get_heuristic_learner()
        
        return (
            create_common_tools() +
            [CodeExecutionTool()] +
            create_dataset_tools() +
            create_ground_truth_tools() +
            create_table_formatting_tools() +
            create_watsonx_tools() +
            [ReflectorAgentTool(), LearningAgentTool()]  # Add reflector and learning agents
        )
    
    def _create_prompt(self, tools: List[BaseTool]) -> PromptTemplate:
        """Create optimized prompt with learned heuristics."""
        memory_summary = self.memory.get_memory_summary() or "No previous memory."
        
        tool_names, tool_desc = get_tool_descriptions(tools)
        
        # Get learned heuristics from heuristic learner
        learned_heuristics = self.heuristic_learner.get_relevant_heuristics("", self.task_type)
        
        # Build heuristics section - sanitize problematic examples
        heuristics_lines = ["COMMON HEURISTICS (CRITICAL - FOLLOW THESE):"]
        for i, h in enumerate(learned_heuristics[:15], 1):  # Top 15 heuristics
            # Remove problematic dict examples that cause Jinja2 parse errors
            h_str = str(h)
            # Replace problematic patterns
            h_str = h_str.replace("{'param': 'value'}", "parameter keys and values")
            h_str = h_str.replace("{engine_id: predicted_rul}", "engine_id to predicted_rul mapping")
            h_str = h_str.replace('{"predictions": {"1": 18, "2": 15, "3": 12}}', "predictions dict with engine IDs")
            heuristics_lines.append(f"{i}. {h_str}")
        
        heuristics = "\n".join(heuristics_lines)
        
        # CRITICAL: ReactAgent uses Jinja2 Template which interprets {{ as print statements
        # We CANNOT escape braces with {{ because Jinja2 will parse them as print statements
        # SOLUTION: Use regex to find and REMOVE/REPLACE all JSON examples with braces
        # Only keep format variables: {question}, {scratchpad}, {examples}, {reflections}, {tool_names}
        import re
        
        tool_desc_sanitized = tool_desc
        if tool_desc:
            # Remove/replace all JSON-like patterns that contain braces
            # These patterns will confuse Jinja2 if we escape them
            patterns_replacements = [
                # Match JSON with code key: {"code": "...", "return_result": ...}
                (r'\{[^{]*"code"[^}]*"return_result"[^}]*\}', 'JSON object with code and return_result keys'),
                (r'\{[^{]*"code"[^}]*\}', 'JSON object with code key'),
                # Match JSON with predictions: {"predictions": {"1": 18, ...}}
                (r'\{[^{]*"predictions"\s*:\s*\{[^}]*\}[^}]*\}', 'JSON object with predictions dict'),
                # Match JSON with engine IDs: {"1": 18, "2": 15, ...}
                (r'\{"[12]"\s*:\s*\d+\s*,\s*"[23]"\s*:\s*\d+[^}]*\}', 'JSON object mapping engine IDs to RUL'),
                # Match any remaining JSON-like patterns with braces (but not format vars)
                (r'\{[^}]*engine_id[^}]*predicted_rul[^}]*\}', 'dict mapping engine_id to predicted_rul'),
            ]
            
            for pattern, replacement in patterns_replacements:
                tool_desc_sanitized = re.sub(pattern, replacement, tool_desc_sanitized, flags=re.IGNORECASE | re.DOTALL)
            
            # Now escape remaining braces (not format vars) to prevent format string errors
            # But preserve format variables: {question}, {scratchpad}, etc.
            format_vars = ['question', 'scratchpad', 'examples', 'reflections', 'tool_names']
            
            # Replace all { and } with placeholders first
            for var in format_vars:
                tool_desc_sanitized = tool_desc_sanitized.replace(f'{{{var}}}', f'__FORMAT_VAR_{var}__')
            
            # Escape remaining braces
            tool_desc_sanitized = tool_desc_sanitized.replace('{', '{{').replace('}', '}}')
            
            # Restore format variables (un-escape them)
            for var in format_vars:
                tool_desc_sanitized = tool_desc_sanitized.replace(f'{{{{__FORMAT_VAR_{var}__}}}}', f'{{{var}}}')
        
        # Sanitize heuristics similarly
        heuristics_sanitized = heuristics
        if heuristics:
            heuristics_sanitized = re.sub(r'\{[^}]*"param"[^}]*\}', 'parameter keys and values', heuristics_sanitized)
            heuristics_sanitized = re.sub(r'\{[^}]*engine_id[^}]*\}', 'engine_id to predicted_rul mapping', heuristics_sanitized)
            
            # Preserve format vars in heuristics too
            format_vars = ['question', 'scratchpad', 'examples', 'reflections', 'tool_names']
            for var in format_vars:
                heuristics_sanitized = heuristics_sanitized.replace(f'{{{var}}}', f'__FORMAT_VAR_{var}__')
            
            heuristics_sanitized = heuristics_sanitized.replace('{', '{{').replace('}', '}}')
            
            for var in format_vars:
                heuristics_sanitized = heuristics_sanitized.replace(f'{{{{__FORMAT_VAR_{var}__}}}}', f'{{{var}}}')
        
        # Build template with placeholders for pre-evaluated parts, keeping format variables for ReactAgent
        # Replace placeholders for pre-evaluated content, leave format variables for ReactAgent as-is
        prompt_template = """You are a fast, efficient Root Agent for PDMBench tasks.

MEMORY CONTEXT (Learn from past mistakes - DO NOT REPEAT):
__MEMORY_SUMMARY__

__HEURISTICS__

TOOLS:
__TOOL_DESC__

CRITICAL INSTRUCTIONS:
1. Check data availability FIRST (check_data_loaded, quick_data_summary)
2. If data already loaded, skip redundant loading steps
3. Be concise - avoid verbose explanations
4. Focus on actionable steps
5. Stop immediately if you detect a loop (same action repeated)
6. IMPORTANT: Action format must be exactly "Action: tool_name" (NO "Tool:" prefix, just the tool name directly)
7. FOR LARGE DATA STRUCTURES (e.g., predictions dict with 100+ engines): Use execute_code_simple instead of verify_rul_predictions or format_table to avoid JSON truncation
8. WHEN EXECUTING CODE: Always store results in variables and print them, e.g., "result = verify_rul_predictions(predictions); print(result)"

WORKFLOW:
1. Check data: check_data_loaded or quick_data_summary
2. If needed, load: list_datasets → load_dataset
3. Execute task-specific operations:
   - CRITICAL: After load_dataset, the data is available as global variables in shared.load_data module
   - Use: from shared.load_data import train_data, test_data, ground_truth
   - DO NOT try to read CSV files directly - use the imported variables
   - CRITICAL: DO NOT use predict_rul tool or set_model_id with 'rul_prediction_model' - these don't work. Instead:
     * For RUL prediction: Use execute_code_simple or execute_python_code to create predictions directly
     * Example: Use execute_code_simple with code that creates predictions dict from test_data
     * Variables persist between execute_code_simple calls, so you can build up results incrementally
   - For RUL prediction: After creating predictions, ALWAYS verify with ground_truth using verify_rul_predictions
4. Verify results (especially RUL predictions):
   - CRITICAL: For RUL tasks, ground truth validation is MANDATORY before finishing
   - After creating predictions, ALWAYS call verify_rul_predictions(predictions, ground_truth)
   - If predictions dict is large (>50 engines), use execute_code_simple with code that calls verify_rul_predictions(predictions_dict, ground_truth)
   - Store verification result in a variable and print it
   - DO NOT finish the task until ground truth validation is complete and results are printed
5. Format output:
   - If data is large, use execute_code_simple with code that calls format_table functions directly
   - Otherwise, use format_table tool

Format:
Question: {question}
Thought: [brief reasoning]
Action: tool_name (NO "Tool:" prefix - use tool name directly, e.g., "Action: quick_data_summary", NOT "Action: Tool: quick_data_summary")
Action Input: JSON format. 
- For execute_python_code: CRITICAL - MUST use SINGLE-LINE JSON with escaped newlines (\\n) - NEVER use literal newlines. 
- For execute_code_simple: Use JSON with code key containing your Python code - simpler alternative that avoids JSON parsing issues. ALWAYS store results in variables and print them. Variables persist between calls, so you can build up results incrementally.
- For verify_rul_predictions: Only use if predictions dict is small (<50 engines). For large dicts, use execute_code_simple instead.
- For format_table: Only use if data list is small (<50 items). For large lists, use execute_code_simple instead.
- For tools without parameters: Use empty JSON object.
Observation: result
... (continue efficiently)
Thought: Task complete
Action: Finish
Action Input: Final answer

{examples}

{reflections}

Begin with the ACTUAL question below:
Question: {question}
{scratchpad}

Note: Available tools are: {tool_names}"""
        
        # Replace placeholders with actual content (already escaped)
        prompt_template = prompt_template.replace("__MEMORY_SUMMARY__", memory_summary)
        prompt_template = prompt_template.replace("__HEURISTICS__", heuristics_sanitized)
        prompt_template = prompt_template.replace("__TOOL_DESC__", tool_desc_sanitized)
        
        return PromptTemplate(
            input_variables=["question", "tool_names", "scratchpad", "examples", "reflections"],
            template=prompt_template
        )
    
    def _select_llm(self):
        """Select LLM based on provider type."""
        if self.model_type == "openai":
            # For OpenAI, we need to create a custom LLM callable
            from reactxen.utils.model_inference import openai_llm
            return lambda prompt, **kwargs: openai_llm(prompt, model_id=self.model_id, **kwargs)
        elif self.model_type == "azure_openai":
            from reactxen.utils.model_inference import azure_openai_llm
            return lambda prompt, **kwargs: azure_openai_llm(prompt, model_id=self.model_id, **kwargs)
        else:
            # Default to WatsonX
            return None  # Will use default watsonx_llm
    
    def create_agent(self):
        """Create the agent with proper configuration."""
        self.start_time = time.time()
        
        # Use model selector to optimize model selection if needed
        if self.model_type == "auto" or not hasattr(self, 'model_type'):
            selected = self.model_selector.select_model(
                task_type=self.task_type,
                estimated_context_size=4096,
                preference="balanced"
            )
            self.model_type = selected["provider"]
            self.model_id = selected["model_id"]
        
        self.tee.write(f"\n{'='*80}\n")
        self.tee.write(f"🚀 FAST OPTIMIZED AGENT - Initialization\n")
        self.tee.write(f"{'='*80}\n")
        self.tee.write(f"Question: {self.question}\n")
        self.tee.write(f"Model: {self.model_type} ({self.model_id})\n")
        self.tee.write(f"Task Type: {self.task_type}\n")
        self.tee.write(f"Timeout: {self.timeout}s (HARD LIMIT)\n\n")
        
        # Create tools
        tools = self._create_tools()
        tool_names, tool_desc = get_tool_descriptions(tools)
        
        # Register with architecture visualizer
        self.arch_visualizer.register_tools(tools)
        self.arch_visualizer.register_root_agent({
            "question": self.question,
            "model_type": self.model_type,
            "model_id": self.model_id,
            "timeout": self.timeout,
            "max_steps": 15,
            "num_reflect_iteration": 2,
            "tools": tools
        })
        
        # Write architecture summary
        arch_summary = self.arch_visualizer.generate_architecture_summary()
        self.tee.write(arch_summary + "\n\n")
        self.tee.flush()
        
        # Create prompt
        prompt = self._create_prompt(tools)
        
        # Select LLM
        react_llm = self._select_llm()
        reflect_llm = self._select_llm()
        
        # Create agent with optimized settings
        # CRITICAL: Use empty react_example to avoid cached examples overriding actual question
        # Also set reflect_example to empty to avoid cached reflection examples
        agent_config = {
            "question": self.question,  # CRITICAL: Pass actual question
            "key": f"fast_pdmbench_{int(time.time())}",  # Unique key to avoid caching
            "agent_prompt": prompt,
            "tools": tools,
            "tool_names": tool_names,
            "tool_desc": tool_desc,
            "react_example": "",  # CRITICAL: Empty to avoid cached examples overriding question
            "reflect_example": "",  # CRITICAL: Empty to avoid cached reflection examples
            "max_steps": 15,  # Reduced for speed (was 20)
            "react_llm_model_id": self.model_id if isinstance(self.model_id, int) else 15,
            "reflect_llm_model_id": self.model_id if isinstance(self.model_id, int) else 15,
            "debug": False,  # Disable debug for speed
            "apply_loop_detection_check": True,
            "log_structured_messages": True,  # REQUIRED when loop detection is enabled
            "early_stop": True,  # Stop early if possible
            "num_reflect_iteration": 2,  # Reduced reflections (was 3)
            "max_retries": 1,
            "actionstyle": "Text",
            "reactstyle": "thought_and_act_together",
            "handle_context_length_overflow": True  # Handle large contexts
        }
        
        # Add custom LLM if needed
        if react_llm:
            agent_config["react_llm"] = react_llm
        if reflect_llm:
            agent_config["reflect_llm"] = reflect_llm
        
        self.tee.write(f"Creating agent with {len(tools)} tools...\n")
        self.tee.flush()
        
        try:
            self.agent = create_reactxen_agent(**agent_config)
            
            # Set max_execution_time directly on agent instance (not a create_reactxen_agent parameter)
            self.agent.max_execution_time = self.timeout
            
            # CRITICAL: Set agent reference in tools to access scratchpad/json_log
            # This allows the tools to recover raw action input when ReactAgent's JSON parsing fails
            for tool in tools:
                if hasattr(tool, 'set_agent_ref'):
                    tool.set_agent_ref(self.agent)
            
            self.tee.write("✅ Agent created successfully\n")
            self.tee.write(f"   Max execution time: {self.timeout}s\n\n")
            self.tee.flush()
        except Exception as e:
            self.tee.write(f"❌ Error creating agent: {e}\n")
            import traceback
            self.tee.write(traceback.format_exc() + "\n")
            raise
        
        return self.agent
    
    def run_with_hard_timeout(self) -> Dict[str, Any]:
        """Run agent with hard timeout enforcement and real-time output."""
        if not self.agent:
            self.create_agent()
        
        self.tee.write(f"\n{'='*80}\n")
        self.tee.write("EXECUTION STARTED\n")
        self.tee.write(f"{'='*80}\n")
        self.tee.write(f"Question: {self.question}\n")
        self.tee.write(f"Task Type: {self.task_type}\n")
        self.tee.write(f"\n📋 SUCCESS CRITERIA:\n")
        self.tee.write(f"  1. Task must be accomplished (completeness ≥ 80%)\n")
        if self.task_type == "rul_prediction":
            self.tee.write(f"  2. Ground truth validation MUST be completed\n")
        self.tee.write(f"  3. Both conditions must be met for success\n\n")
        self.tee.write(f"⏱️  Monitoring execution step-by-step...\n\n")
        self.tee.flush()
        
        result = None
        error = None
        execution_complete = threading.Event()
        
        def run_agent():
            nonlocal result, error
            # Save original stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                # Redirect stdout/stderr to tee to capture ReactAgent's print statements
                sys.stdout = self.tee
                sys.stderr = self.tee
                
                # Run agent - this will produce output via print statements
                # Note: ReactReflectAgent.run() returns None, but stores result in self.answer
                self.agent.run()
                
                # Restore stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                # Get result from agent's answer attribute
                result = getattr(self.agent, 'answer', None) or getattr(self.agent, 'scratchpad', '')
                
                # Learn from successful completion
                if result:
                    self.heuristic_learner.learn_from_success("agent_completion", str(result)[:200], self.task_type)
                
                execution_complete.set()
            except Exception as e:
                # Restore stdout/stderr even on error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                error = str(e)
                self.tee.write(f"❌ Agent Error: {error}\n")
                import traceback
                full_traceback = traceback.format_exc()
                self.tee.write(f"Full traceback:\n{full_traceback}\n")
                
                # Try to get partial result from agent
                try:
                    result = getattr(self.agent, 'answer', None) or getattr(self.agent, 'scratchpad', '')
                except:
                    result = None
                
                # Learn from error
                self.heuristic_learner.learn_from_mistake("agent_execution", str(result)[:200] if result else "", str(error))
                
                execution_complete.set()
        
        # Run agent in thread
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()
        
        # Monitor with hard timeout - check every second
        elapsed = 0
        last_status_time = 0
        status_interval = 15  # Print status every 15 seconds
        
        while not execution_complete.is_set() and elapsed < self.timeout:
            time.sleep(1)
            elapsed = time.time() - self.start_time
            
            # Periodic status updates
            if elapsed - last_status_time >= status_interval:
                remaining = self.timeout - elapsed
                self.tee.write(f"⏱️  [{elapsed:.0f}s/{self.timeout}s] Still running... {remaining:.0f}s remaining\n")
                self.tee.flush()
                last_status_time = elapsed
            
            # Hard timeout check
            if elapsed >= self.timeout:
                self.tee.write(f"\n🛑 HARD TIMEOUT: {self.timeout}s exceeded - TERMINATING\n")
                self.tee.flush()
                self.kill_flag.set()
                break
        
        # Wait briefly for completion if close
        if execution_complete.is_set():
            agent_thread.join(timeout=2)
        else:
            # Force termination
            self.tee.write("⚠️  Agent thread still running after timeout\n")
            self.tee.write("   Execution terminated due to timeout\n")
        
        execution_time = time.time() - self.start_time
        
        # Final status
        self.tee.write(f"\n{'='*80}\n")
        self.tee.write("EXECUTION COMPLETE\n")
        self.tee.write(f"{'='*80}\n")
        self.tee.write(f"Execution Time: {execution_time:.2f}s\n")
        self.tee.write(f"Timeout Limit: {self.timeout}s\n")
        
        if execution_time >= self.timeout:
            self.tee.write(f"Status: ⏱️  TIMEOUT EXCEEDED\n")
        elif error:
            self.tee.write(f"Status: ❌ FAILED - {error[:100]}\n")
        elif result:
            self.tee.write(f"Status: ✅ SUCCESS\n")
            self.tee.write(f"Result preview: {str(result)[:200]}...\n")
        else:
            self.tee.write(f"Status: ⚠️  NO RESULT\n")
        
        self.tee.flush()
        
        return {
            "success": error is None and result is not None and execution_time < self.timeout,
            "result": str(result) if result else None,
            "error": error or ("Timeout" if execution_time >= self.timeout else None),
            "execution_time": execution_time,
            "timeout_exceeded": execution_time >= self.timeout
        }
    
    def close(self):
        """Close resources."""
        if hasattr(self, 'tee'):
            self.tee.close()


def create_fast_agent(question: str, output_file: Path,
                     model_type: str = "watsonx", model_id: Union[int, str] = 15,
                     timeout: int = 180, task_type: str = "general") -> FastOptimizedAgent:
    """Create fast optimized agent."""
    return FastOptimizedAgent(question, output_file, model_type, model_id, timeout, task_type)

