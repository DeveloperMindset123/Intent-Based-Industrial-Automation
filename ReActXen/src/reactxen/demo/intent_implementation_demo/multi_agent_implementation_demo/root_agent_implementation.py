"""
Root Agent Implementation - Fully dynamic agent creation system.
Root agent creates sub-agents and tools dynamically based on task requirements.
"""
import time
import traceback
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from langchain_core.prompts import PromptTemplate

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.shared_utils import setup_paths, get_tool_descriptions
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from tools_logic import create_watsonx_tools
# Import from local directory to avoid parent directory version
from .dynamic_agent_system import create_dynamic_agent_tools
from ground_truth_verification import create_ground_truth_tools
from table_formatter import create_table_formatting_tools
from reactxen.agents.react.utils import extract_action_and_input as original_extract_action_and_input

setup_paths()


def normalize_action_name(action: str) -> str:
    """Normalize action name by removing 'Action:' prefix if present."""
    if not action:
        return action
    # Remove "Action:" prefix (case insensitive, with optional whitespace)
    action = re.sub(r'^Action\s*:\s*', '', action, flags=re.IGNORECASE)
    return action.strip()


def extract_action_and_input_with_normalization(step: str, stop):
    """Wrapper for extract_action_and_input that normalizes action names."""
    result = original_extract_action_and_input(step, stop)
    if result and "action" in result and result["action"]:
        result["action"] = normalize_action_name(result["action"])
    return result


# Monkey-patch the extract_action_and_input function to normalize actions
import reactxen.agents.react.utils as react_utils
react_utils.extract_action_and_input = extract_action_and_input_with_normalization


def create_root_agent(
    question: str,
    react_llm_model_id: int = 15,
    max_steps: int = 40,
    **kwargs
) -> Any:
    """Create the root agent with dynamic agent creation capabilities."""
    all_tools = []
    
    # Add initialization tools
    try:
        watsonx_tools = create_watsonx_tools()
        init_tools = [t for t in watsonx_tools if t.name == "initialize_watsonx_api"]
        all_tools.extend(init_tools)
    except:
        pass
    
    # Add dynamic agent creation tools (THIS IS THE KEY - no pre-defined agents!)
    # Note: We'll pass root_agent after it's created (see below)
    try:
        dynamic_tools = create_dynamic_agent_tools(parent_model_id=react_llm_model_id, root_agent=None)
        all_tools.extend(dynamic_tools)
    except Exception as e:
        if kwargs.get("debug", False):
            print(f"Warning: Could not load dynamic agent tools: {e}")
    
    # Add ground truth verification tools
    try:
        all_tools.extend(create_ground_truth_tools())
    except Exception as e:
        if kwargs.get("debug", False):
            print(f"Warning: Could not load ground truth tools: {e}")
    
    # Add table formatting tools
    try:
        all_tools.extend(create_table_formatting_tools())
    except Exception as e:
        if kwargs.get("debug", False):
            print(f"Warning: Could not load table formatting tools: {e}")
    
    # Add search tools for online lookup
    try:
        from shared.shared_utils import get_search_tools
        all_tools.extend(get_search_tools())
    except:
        pass
    
    # Debug: Print available tools
    if kwargs.get("debug", False):
        print(f"Root agent has {len(all_tools)} tools: {[t.name for t in all_tools]}")
    
    tool_names, tool_desc = get_tool_descriptions(all_tools)
    
    root_agent_prompt = PromptTemplate(
        input_variables=["question", "tool_desc", "scratchpad"],
        template="""You are a Root Agent with the ability to CREATE sub-agents and tools dynamically.

YOUR CAPABILITIES:
1. CREATE sub-agents dynamically using create_sub_agent tool
2. CREATE custom tools dynamically using create_dynamic_tool tool
3. EXECUTE Python code using execute_python_code tool
4. SEARCH online for information using search tools
5. VERIFY predictions against ground truth (for RUL tasks)
6. FORMAT results as tables

CRITICAL: You do NOT have pre-defined sub-agents. You MUST create them based on the task!

WORKFLOW:
1. Analyze the question to determine what sub-agents and tools are needed
2. Initialize WatsonX API if needed (use initialize_watsonx_api)
3. CREATE sub-agents dynamically:
   - Use create_sub_agent tool to create agents for:
     * Data loading and ML model training (if needed)
     * RUL prediction and risk assessment (if needed)
     * Cost estimation (if needed)
     * Safety protocols (if needed)
   - Assign appropriate model_id to each agent
   - Specify the role, workflow, and tools_description for each agent
4. CREATE custom tools if standard tools don't meet your needs (use create_dynamic_tool)
5. EXECUTE code if you need calculations or data processing (use execute_python_code)
6. Delegate tasks to the created sub-agents
7. For RUL prediction tasks: Verify predictions using verify_rul_predictions tool
8. Format final results using format_table tool
9. Synthesize comprehensive final answer

CRITICAL - TOOL NAME REQUIREMENTS:
- You MUST use the EXACT tool name from the TOOLS list below
- Tool names are case-sensitive and must match exactly
- Examples of CORRECT tool names: "execute_python_code", "create_sub_agent", "initialize_watsonx_api"
- Examples of INCORRECT: "Execute Python code", "Create sub-agent", "Initialize WatsonX API"
- The Action field must contain ONLY the exact tool name, nothing else

IMPORTANT:
- CREATE agents and tools as needed - don't assume they exist
- Use execute_python_code to write and run your own code
- Use create_dynamic_tool to create tools with custom logic
- Always verify RUL predictions against ground truth
- Always format results as tables

TOOLS:
{tool_desc}

FORMAT - CRITICAL: Follow this EXACT format:
Question: {question}
Thought: analyze what agents and tools are needed
Action: execute_python_code
Action Input: JSON with code key containing your Python code as string
Observation: result
... (create agents and tools as needed, then delegate tasks)
Thought: I have completed the task
Action: Finish
Action Input: Your comprehensive final answer (formatted as table if possible)

IMPORTANT FORMATTING RULES:
- After "Action:" write ONLY the tool name (e.g., execute_python_code)
- DO NOT write "Action: Action: execute_python_code" - that is WRONG
- DO NOT write "Action: Execute Python code" - that is WRONG
- Write exactly: "Action: execute_python_code" (lowercase with underscores)
- The Action line should be: "Action: " followed immediately by the tool name

Begin!
Question: {question}
{scratchpad}"""
    )
    
    reflect_prompt = PromptTemplate(
        input_variables=["question", "scratchpad"],
        template="""Review the previous attempt and identify what went wrong.

CRITICAL CHECKS:
1. Did you CREATE the necessary sub-agents? If not, use create_sub_agent tool
2. Did you CREATE custom tools if needed? If not, use create_dynamic_tool tool
3. Did you delegate tasks to created agents? If not, delegate now
4. Did you verify RUL predictions? (Required for RUL tasks)
5. Did you format results as table? (Required for readability)
6. Did you provide a comprehensive final answer?

COMMON ERRORS:
- Using descriptive action names instead of exact tool names - USE EXACT NAMES like "execute_python_code" not "Execute Python code"
- Assuming pre-defined agents exist - YOU MUST CREATE THEM
- Not creating tools when needed - USE create_dynamic_tool
- Not executing code for calculations - USE execute_python_code
- Not formatting results - USE format_table

TOOL NAME EXAMPLES:
- ✅ CORRECT: "execute_python_code"
- ❌ WRONG: "Execute Python code" or "execute python code" or "Execute_Python_Code"
- ✅ CORRECT: "create_sub_agent"
- ❌ WRONG: "Create sub-agent" or "create sub agent" or "CreateSubAgent"

Previous attempt:
Question: {question}
{scratchpad}

Reflection: What went wrong and how to fix it."""
    )
    
    agent_config = {
        "question": question,
        "key": "root_agent_dynamic",
        "max_steps": max_steps,
        "agent_prompt": root_agent_prompt,
        "reflect_prompt": reflect_prompt,
        "tools": all_tools,
        "tool_names": tool_names,
        "tool_desc": tool_desc,
        "react_llm_model_id": react_llm_model_id,
        "reflect_llm_model_id": react_llm_model_id,
        "actionstyle": "Text",
        "reactstyle": "thought_and_act_together",
        "max_retries": 2,
        "num_reflect_iteration": 3,
        "early_stop": True,
        "apply_loop_detection_check": True,
        "log_structured_messages": True,  # Required when loop detection is enabled
        "debug": kwargs.get("debug", False),
    }
    
    agent_config.update({k: v for k, v in kwargs.items() if k not in agent_config})
    agent = create_reactxen_agent(**agent_config)
    
    # Update root_agent reference in CreateSubAgentTool instances
    for tool in all_tools:
        if hasattr(tool, 'root_agent'):
            object.__setattr__(tool, 'root_agent', agent)
    
    return agent


def run_root_agent(
    question: str,
    react_llm_model_id: int = 15,
    max_steps: int = 40,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run the root agent and return results."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*70)
    print("🚀 ROOT AGENT - DYNAMIC AGENT CREATION SYSTEM")
    print("="*70)
    print(f"Question: {question}")
    print(f"Model ID: {react_llm_model_id}")
    print(f"Max Steps: {max_steps}")
    print("="*70)
    
    root_agent = create_root_agent(
        question=question,
        react_llm_model_id=react_llm_model_id,
        max_steps=max_steps,
        **kwargs
    )
    
    start_time = time.time()
    try:
        print("🚀 Starting root agent execution...")
        result = root_agent.run()
        execution_time = time.time() - start_time
        print(f"✅ Root agent execution completed in {execution_time:.2f}s")
        
        final_answer = root_agent.answer if hasattr(root_agent, 'answer') else str(result)
        steps_taken = root_agent.step_n if hasattr(root_agent, 'step_n') else None
        
        # Extract final answer from scratchpad if needed
        if not final_answer or len(final_answer.strip()) < 50:
            if hasattr(root_agent, 'scratchpad') and root_agent.scratchpad:
                scratchpad = str(root_agent.scratchpad)
                if "Final Answer" in scratchpad:
                    parts = scratchpad.split("Final Answer")
                    if len(parts) > 1:
                        extracted = parts[-1].split("Observation")[0].strip()
                        if len(extracted) > 50:
                            final_answer = f"Final Answer:\n{extracted}"
        
        metrics = {}
        try:
            if hasattr(root_agent, 'export_benchmark_metric'):
                metrics = root_agent.export_benchmark_metric()
        except:
            pass
        
        results = {
            "question": question,
            "final_answer": final_answer,
            "execution_time": execution_time,
            "steps_taken": steps_taken,
            "metrics": metrics,
            "success": metrics.get('status') == 'Accomplished' if metrics else True,
        }
        
        import json
        results_file = output_dir / f"root_agent_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "="*70)
        status_icon = "✅" if results.get("success") else "⚠️"
        print(f"{status_icon} ROOT AGENT EXECUTION COMPLETED")
        print("="*70)
        print(f"Execution Time: {execution_time:.2f}s")
        print(f"Steps Taken: {steps_taken}")
        print(f"Success: {results.get('success', False)}")
        if metrics and isinstance(metrics, dict):
            review_status = metrics.get('status', 'Unknown')
            print(f"Review Status: {review_status}")
        print(f"Results saved to: {results_file}")
        print("="*70)
        
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        traceback.print_exc()
        
        results = {
            "question": question,
            "final_answer": f"Error: {error_msg}",
            "execution_time": execution_time,
            "steps_taken": None,
            "metrics": {},
            "success": False,
            "error": error_msg,
        }
        
        import json
        results_file = output_dir / f"root_agent_results_{timestamp}_error.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
