"""
Root Agent Implementation - Orchestrates all sub-agents for end-to-end industrial automation.
"""
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from langchain_core.prompts import PromptTemplate

from shared_utils import setup_paths, get_dataset_tools, get_tool_descriptions
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from hierarchical_agents import create_sub_agent_tools
from tools_logic import create_watsonx_tools

setup_paths()


def create_root_agent(
    question: str,
    react_llm_model_id: int = 15,
    max_steps: int = 35,  # Increased default to allow complete workflow
    **kwargs
) -> Any:
    """Create the root agent that orchestrates all sub-agents."""
    all_tools = []
    # Note: Root agent delegates dataset operations to data_scientist_agent
    # Dataset tools are available to sub-agents, not root agent
    all_tools.extend(create_sub_agent_tools(react_llm_model_id=react_llm_model_id))
    
    # Add initialization tools
    try:
        watsonx_tools = create_watsonx_tools()
        init_tools = [t for t in watsonx_tools if t.name == "initialize_watsonx_api"]
        all_tools.extend(init_tools)
    except:
        pass
    
    # Debug: Print available tools
    if kwargs.get("debug", False):
        print(f"Root agent has {len(all_tools)} tools: {[t.name for t in all_tools]}")
    
    tool_names, tool_desc = get_tool_descriptions(all_tools)
    
    root_agent_prompt = PromptTemplate(
        input_variables=["question", "tool_desc", "tool_names", "scratchpad"],
        template="""You are the Root Agent for Intent-Based Industrial Automation, orchestrating specialized sub-agents.

YOUR ROLE:
You coordinate multiple specialized sub-agents to complete the task:
1. Data Scientist Agent - ML model training and evaluation
2. Predictive Maintenance Agent - RUL prediction and risk assessment
3. Cost-Benefit Analysis Agent - Cost estimation and ROI analysis
4. Safety/Policy Agent - Safety protocols and compliance

CRITICAL: You MUST complete ALL steps and provide a final answer with equipment IDs, RUL predictions, cost estimates, and safety recommendations.

WORKFLOW (FOLLOW IN ORDER):
Step 1: Initialize WatsonX API using initialize_watsonx_api tool (use empty JSON object - no parameters needed)
Step 2: Delegate to data_scientist_agent with query: "Load the CWRU dataset, train a model for RUL prediction, and evaluate it"
   - Action Input: JSON with query key containing the query string (format: JSON object with "query" key)
   - The data_scientist_agent has access to dataset loading tools and will handle loading internally
Step 3: Delegate to predictive_maintenance_agent with query: "Identify equipment with RUL <= 20 cycles that are at risk of failure"
   - Action Input: JSON with query key containing the query string
Step 4: Delegate to cost_benefit_agent with query: "Estimate maintenance costs for equipment at risk"
   - Action Input: JSON with query key containing the query string
Step 5: Delegate to safety_policy_agent with query: "Provide OSHA safety protocols and recommendations for equipment with low RUL"
   - Action Input: JSON with query key containing the query string
Step 6: Synthesize ALL results into a comprehensive final answer with:
   - List of equipment IDs at risk (RUL <= 20 cycles)
   - RUL predictions for each equipment
   - Cost estimates for maintenance
   - Safety recommendations and protocols

IMPORTANT FORMATTING RULES:
- For sub-agents: Action Input must be JSON with query key - query value must be a STRING, not a dict
- Format for sub-agents: JSON object with "query" key containing your question as a string
- For direct tools: Use appropriate JSON format based on tool requirements
- If a sub-agent call fails, try again with a simpler query or use direct tools instead
- NEVER pass context as a dict - only pass query as a string

TOOLS:
{tool_desc}

FORMAT:
Question: {question}
Thought: think about next step
Action: tool name
Action Input: JSON format (for sub-agents: JSON with "query" key, for direct tools: use tool's required format)
Observation: result
... (continue until ALL steps are complete)
Thought: I have completed all steps and gathered all information
Action: Finish
Action Input: Your comprehensive final answer here

CRITICAL: When providing the Final Answer, use Action: Finish and include in Action Input:
1. Equipment IDs at risk (RUL <= 20 cycles) - list each equipment ID
2. RUL predictions for each equipment - provide specific RUL values in cycles
3. Cost estimates for maintenance - provide dollar amounts for each equipment
4. Safety recommendations and protocols - provide specific OSHA guidelines for each equipment

Begin!
Question: {question}
{scratchpad}"""
    )
    
    reflect_prompt = PromptTemplate(
        input_variables=["question", "scratchpad"],
        template="""Review the previous attempt and identify what went wrong and how to fix it.

CRITICAL CHECKS:
1. Did you initialize WatsonX API? If not, use initialize_watsonx_api tool
2. Did you delegate to data_scientist_agent? If not, delegate with query: "Load the CWRU dataset, train a model for RUL prediction, and evaluate it"
3. Did you delegate to predictive_maintenance_agent? If not, delegate with query: "Identify equipment with RUL <= 20 cycles that are at risk of failure"
4. Did you delegate to cost_benefit_agent? If not, delegate with query: "Estimate maintenance costs for equipment at risk"
5. Did you delegate to safety_policy_agent? If not, delegate with query: "Provide OSHA safety protocols and recommendations for equipment with low RUL"
6. Did you provide a final answer with equipment IDs, RUL values, costs, and safety info? If not, synthesize all sub-agent results into a comprehensive final answer

COMMON ERRORS TO AVOID:
- Passing context as dict instead of string: Use JSON with only query key, not query and context keys together
- Not completing all steps: You MUST complete all 6 steps (initialize API, delegate to all 4 sub-agents, synthesize final answer)
- Not providing final answer: You MUST synthesize results into a final answer with equipment IDs, RUL predictions, costs, and safety recommendations
- Using wrong tool format: For sub-agents, Action Input must be valid JSON with "query" key containing your question as a string

Previous attempt:
Question: {question}
{scratchpad}

Reflection: What went wrong and how to fix it in the next attempt."""
    )
    
    agent_config = {
        "question": question,
        "key": "root_agent_hierarchical",
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
        "early_stop": False,
        "debug": kwargs.get("debug", False),
    }
    
    agent_config.update({k: v for k, v in kwargs.items() if k not in agent_config})
    return create_reactxen_agent(**agent_config)


def run_root_agent(
    question: str,
    react_llm_model_id: int = 15,
    max_steps: int = 35,  # Increased default to allow complete workflow
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run the root agent and return results."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*70)
    print("🚀 ROOT AGENT - HIERARCHICAL INDUSTRIAL AUTOMATION")
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
        
        # Check if final answer is meaningful and extract from various sources
        if not final_answer or len(final_answer.strip()) < 50:
            # Try to extract from scratchpad if available
            if hasattr(root_agent, 'scratchpad') and root_agent.scratchpad:
                scratchpad = str(root_agent.scratchpad)
                # Look for Final Answer in scratchpad
                if "Final Answer:" in scratchpad or "Final Answer" in scratchpad:
                    # Extract everything after "Final Answer"
                    parts = scratchpad.split("Final Answer")
                    if len(parts) > 1:
                        extracted = parts[-1].split("Observation")[0].strip()
                        if len(extracted) > 50:
                            final_answer = f"Final Answer:\n{extracted}"
                
                # If still no good answer, look for equipment/RUL info
                if not final_answer or len(final_answer.strip()) < 50:
                    if "equipment" in scratchpad.lower() or "rul" in scratchpad.lower() or "engine" in scratchpad.lower():
                        # Try to extract structured information
                        lines = scratchpad.split("\n")
                        relevant_lines = []
                        for line in lines:
                            if any(keyword in line.lower() for keyword in ["equipment", "engine", "rul", "cost", "safety", "maintenance"]):
                                relevant_lines.append(line)
                        if relevant_lines:
                            final_answer = "Final Answer:\n" + "\n".join(relevant_lines[-20:])  # Last 20 relevant lines
                        else:
                            final_answer = f"Based on the execution:\n\n{scratchpad[-800:]}\n\nNote: Please review the full execution log for complete details."
                    else:
                        final_answer = "Task execution completed but final answer was not properly formatted. Please review the execution log for details."
        
        metrics = {}
        try:
            if hasattr(root_agent, 'export_benchmark_metric'):
                metrics = root_agent.export_benchmark_metric()
                # Check if task was accomplished based on review agent
                if metrics and isinstance(metrics, dict):
                    review_status = metrics.get('status', '')
                    if review_status == 'Not Accomplished':
                        # Mark as not successful if review says not accomplished
                        results_success = False
                    else:
                        results_success = True
                else:
                    results_success = True
            else:
                results_success = True
        except:
            results_success = True
        
        results = {
            "question": question,
            "final_answer": final_answer,
            "execution_time": execution_time,
            "steps_taken": steps_taken,
            "metrics": metrics,
            "success": results_success,
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
        print(f"\n❌ Error: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "question": question,
            "error": str(e),
            "execution_time": execution_time,
            "success": False,
        }
