"""
Optimized PDMBench Agent with all improvements:
- Context window detection and auto-switching
- Performance optimization
- Real-time streaming
- Flexible nested architecture
- Heuristics learning
- Step-by-step audit
"""
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.shared_utils import setup_paths, get_tool_descriptions
setup_paths()

from optimized_agent_system import (
    ContextWindowManager, OutputStreamer, HeuristicsManager,
    ArchitectureVisualizer, StepAuditor, MODEL_CONFIGS
)
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
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
from agent_tool_wrapper import AgentTool


class OptimizedPDMBenchAgent:
    """Optimized PDMBench agent with all improvements."""
    
    def __init__(self, question: str, output_file: Path, 
                 preferred_model_id: int = 15, timeout: int = 180):
        self.question = question
        self.output_file = output_file
        self.preferred_model_id = preferred_model_id
        self.timeout = timeout
        
        # Initialize systems
        self.memory = get_memory_system()
        self.heuristics = HeuristicsManager(self.memory)
        self.auditor = StepAuditor()
        self.streamer = OutputStreamer(output_file)
        
        # Model selection
        self.model_config = self._select_model()
        
        # Architecture
        self.architecture = None
        self.root_agent = None
        self.sub_agents = {}
        
        # Performance tracking
        self.start_time = None
        self.execution_time = 0.0
    
    def _select_model(self) -> Any:
        """Select appropriate model based on context."""
        # Estimate context size
        estimated_tokens = ContextWindowManager.estimate_tokens(
            self.question, self.preferred_model_id
        )
        
        self.streamer.write(f"📊 Estimated context: {estimated_tokens} tokens\n")
        
        # Select model
        config = ContextWindowManager.select_model(self.question, self.preferred_model_id)
        
        self.streamer.write(f"🤖 Selected model: {config.provider} ({config.model_id})\n")
        self.streamer.write(f"   Context limit: {config.context_limit}\n")
        
        return config
    
    def _create_architecture(self):
        """Create and visualize architecture."""
        self.streamer.write("\n" + "="*80 + "\n")
        self.streamer.write("ARCHITECTURE INITIALIZATION\n")
        self.streamer.write("="*80 + "\n\n")
        
        # Get available tools
        all_tools = (
            [CodeExecutionTool()] +
            create_common_tools() +
            create_dataset_tools() +
            create_ground_truth_tools() +
            create_table_formatting_tools()
        )
        tool_names = [t.name for t in all_tools]
        
        # Create architecture
        self.architecture = ArchitectureVisualizer.create_architecture(
            self.question, tool_names, {}
        )
        
        # Display architecture
        arch_str = ArchitectureVisualizer.format_architecture(self.architecture)
        self.streamer.write(arch_str + "\n\n")
        
        return all_tools
    
    def _create_enhanced_prompt(self, tools: List[BaseTool]) -> PromptTemplate:
        """Create enhanced prompt with heuristics and memory."""
        memory_summary = self.memory.get_memory_summary() or "No previous memory recorded."
        heuristics = self.heuristics.get_heuristics_prompt()
        
        prompt_template = f"""You are an optimized Root Agent for PDMBench tasks.

MEMORY CONTEXT (Learn from past mistakes):
{memory_summary}

COMMON HEURISTICS (Follow these patterns):
{heuristics}

AVAILABLE TOOLS:
{{tool_desc}}

ARCHITECTURE:
{json.dumps(self.architecture, indent=2)}

CRITICAL RULES:
1. ALWAYS check if data is loaded using check_data_loaded BEFORE loading
2. For RUL predictions, MUST verify with verify_rul_predictions tool
3. If same error repeats, use learning_analyze tool
4. Delegate complex tasks to appropriate sub-agents
5. Validate tool inputs using validate_tool_input before execution
6. Use quick_data_summary to check available data without full loading

WORKFLOW:
1. Check data availability (check_data_loaded, quick_data_summary)
2. Analyze question and determine needed sub-agents
3. Delegate to sub-agents or use tools directly
4. Verify results (especially RUL predictions)
5. Format and return comprehensive answer

Format:
Question: {{question}}
Thought: [reasoning with architecture consideration]
Action: tool_name or sub_agent_name
Action Input: {{"param": "value"}} (validated JSON)
Observation: result
... (continue with audit of each step)
Thought: I have completed the task
Action: Finish
Action Input: Final comprehensive answer

Begin!
Question: {{question}}
{{scratchpad}}"""
        
        return PromptTemplate(
            input_variables=["question", "tool_desc", "scratchpad"],
            template=prompt_template
        )
    
    def create_agent(self):
        """Create the optimized agent."""
        self.start_time = time.time()
        
        self.streamer.write(f"\n🚀 Creating Optimized PDMBench Agent\n")
        self.streamer.write(f"Question: {self.question[:100]}...\n\n")
        
        # Create architecture
        all_tools = self._create_architecture()
        
        # Add common tools
        all_tools = create_common_tools() + all_tools
        
        # Get tool descriptions
        tool_names, tool_desc = get_tool_descriptions(all_tools)
        
        # Create prompt
        root_prompt = self._create_enhanced_prompt(all_tools)
        
        # Create agent with selected model
        model_id = self.model_config.model_id
        
        self.streamer.write(f"🤖 Creating agent with {self.model_config.provider} model: {model_id}\n\n")
        
        self.root_agent = create_reactxen_agent(
            question=self.question,
            key="optimized_pdmbench_root",
            agent_prompt=root_prompt,
            tools=all_tools,
            tool_names=tool_names,
            tool_desc=tool_desc,
            react_llm_model_id=model_id if isinstance(model_id, int) else 15,
            reflect_llm_model_id=model_id if isinstance(model_id, int) else 15,
            max_steps=30,  # Reduced for performance
            debug=True,
            apply_loop_detection_check=True,
            log_structured_messages=True,
            early_stop=True  # Stop early if task completed
        )
        
        return self.root_agent
    
    def run_with_timeout(self) -> Dict[str, Any]:
        """Run agent with timeout and real-time streaming."""
        if not self.root_agent:
            self.create_agent()
        
        self.streamer.write("\n" + "="*80 + "\n")
        self.streamer.write("EXECUTION STARTED\n")
        self.streamer.write("="*80 + "\n\n")
        
        result = None
        error = None
        
        def run_agent():
            nonlocal result, error
            try:
                result = self.root_agent.run()
            except Exception as e:
                error = str(e)
                import traceback
                self.streamer.write(f"❌ Error: {error}\n")
                self.streamer.write(f"{traceback.format_exc()}\n")
        
        # Run in thread with timeout
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)
        
        if thread.is_alive():
            self.streamer.write(f"\n⏱️  Execution timeout after {self.timeout}s\n")
            return {
                "success": False,
                "error": f"Timeout after {self.timeout} seconds",
                "execution_time": self.timeout
            }
        
        self.execution_time = time.time() - self.start_time
        
        # Get audit report
        audit_report = self.auditor.get_audit_report()
        self.streamer.write(f"\n📊 Audit Report:\n{audit_report}\n")
        
        self.streamer.write(f"\n⏱️  Execution time: {self.execution_time:.2f}s\n")
        
        return {
            "success": error is None,
            "result": str(result) if result else None,
            "error": error,
            "execution_time": self.execution_time,
            "audit": audit_report
        }
    
    def close(self):
        """Close resources."""
        self.streamer.close()


def create_optimized_agent(question: str, output_file: Path,
                          preferred_model_id: int = 15, timeout: int = 180) -> OptimizedPDMBenchAgent:
    """Create optimized PDMBench agent."""
    return OptimizedPDMBenchAgent(question, output_file, preferred_model_id, timeout)

