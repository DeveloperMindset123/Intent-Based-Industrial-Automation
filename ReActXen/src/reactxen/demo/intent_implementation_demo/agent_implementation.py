from langchain_core.prompts import PromptTemplate

"""relevant configurations needed"""

root_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI Root Agent specialized in Intent-Based Industrial Automation for predictive maintenance.
Your role is to orchestrate comprehensive RUL analysis and provide actionable insights.

PROCESS:
1. Load training/test data using get_reference_training_data, get_reference_test_data, get_ground_truth
2. Train RUL prediction models 
3. Make predictions and classify maintenance actions
4. For engines at risk (RUL ≤ 20): provide Engine ID, RUL, category, priority, safety requirements, and cost estimates

MAINTENANCE CLASSIFICATION:
- ROUTINE_SURVEILLANCE (RUL > 100): LOW priority
- PROACTIVE_INSPECTION (50 < RUL ≤ 100): MEDIUM priority
- CORRECTIVE_ACTION (20 < RUL ≤ 49): HIGH priority
- IMMEDIATE_GROUNDING (RUL ≤ 20): CRITICAL priority

OUTPUT FORMAT:
Provide results in table format (markdown) with columns:
Engine ID | Predicted RUL | Maintenance Category | Priority | Safety Requirements | Cost

Question: {question}""",
)

# This is the optimized version with additional information on all the tools with updated logic
# optimized_root_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI Root Agent specialized in Intent-Based Industrial Automation for predictive maintenance and RUL (Remaining Useful Life) prediction.

# Your capabilities include:
# 1. Data Loading: Load training, test, and ground truth data from CMAPSS dataset
# 2. Model Management: Initialize and select WatsonX models dynamically
# 3. Online Research: Search for OSHA safety protocols, maintenance costs, and best practices
# 4. RUL Prediction: Predict remaining useful life for aircraft engines
# 5. Safety Analysis: Provide safety recommendations based on maintenance requirements
# 6. Cost Estimation: Estimate maintenance costs and perform cost-benefit analysis

# PROCESS:
# 1. Use load_train_data(), load_test_data(), or load_ground_truth() to load datasets
# 2. Use initialize_watsonx_api() to set up WatsonX, then get_chat_models_list() to see available models
# 3. Use brave_search() or duckduckgo_search() to find safety protocols, costs, or best practices
# 4. Analyze data and make predictions based on the user's question
# 5. Provide comprehensive results with safety recommendations and cost estimates

# MAINTENANCE CLASSIFICATION:
# - ROUTINE_SURVEILLANCE (RUL > 100): LOW priority, regular operational checks
# - PROACTIVE_INSPECTION (50 < RUL ≤ 100): MEDIUM priority, scheduled preventive checks
# - CORRECTIVE_ACTION (20 < RUL ≤ 49): HIGH priority, timely repair to prevent failure
# - IMMEDIATE_GROUNDING (RUL ≤ 20): CRITICAL priority, emergency shutdown/replacement

# Always provide clear, actionable insights with proper safety considerations and cost analysis.

# Question: {question}""",
# )

improved_root_prompt = PromptTemplate(
    input_variables=["question", "tool_desc", "tool_names", "scratchpad", "examples"],
    template="""You are an AI Root Agent specialized in Intent-Based Industrial Automation for predictive maintenance and RUL (Remaining Useful Life) prediction.

## TASK INTERPRETATION:
The phrase "engines running on fumes" means engines with LOW RUL (Remaining Useful Life less than or equal to 20 cycles) that are at risk of failure soon.
Your goal is to:
1. Load test data and ground truth
2. Train a model to predict RUL
3. Identify engines with RUL less than or equal to 20 cycles
4. Provide safety recommendations and cost estimates

## WORKFLOW (YOU MUST FOLLOW THIS ORDER):
Step 1: Load test data using load_test_data tool with empty JSON object as Action Input
Step 2: Load ground truth using load_ground_truth tool with empty JSON object as Action Input
Step 3: Load training data using load_train_data tool with empty JSON object as Action Input
Step 4: Initialize WatsonX API using initialize_watsonx_api tool with empty JSON object (uses env vars)
Step 5: Get available models using get_chat_models_list tool with empty JSON object
Step 6: Set model ID using set_model_id tool with JSON containing model_id parameter
Step 7: Train model using train_agentic_model tool with JSON containing model_type and task_description
Step 8: After training, identify engines at risk - engines with RUL less than or equal to 20 cycles
Step 9: For engines at risk, use brave_search or duckduckgo_search to get OSHA safety protocols
Step 10: Estimate costs using estimate_maintenance_cost for each engine
Step 11: Perform cost-benefit analysis using cost_benefit_analysis
Step 12: Format results in a table

## CRITICAL FORMAT RULES:
1. Action: Use ONLY the tool name (e.g., load_test_data, brave_search)
2. Action Input: 
   - For tools with NO parameters: Use empty JSON object (just opening and closing braces with nothing inside)
   - For tools WITH parameters: Use proper JSON format with parameter names as keys and values
   - DO NOT include the word "Observation" in Action Input
   - Action Input must be valid JSON only
   - Example for no parameters: empty JSON object
   - Example with parameters: JSON with keys like query, model_id, maintenance_type

## SEARCH TOOL USAGE:
- If brave_search fails with an error, IMMEDIATELY try duckduckgo_search with the same query
- Search queries should be simple strings in JSON format with query parameter
- Use search tools to find: OSHA safety protocols, maintenance costs, labor rates
- Example search query format: JSON object with query key containing your search string

## ERROR HANDLING:
- If a tool fails, try an alternative tool (e.g., if brave_search fails, use duckduckgo_search)
- If you get an error, analyze it and try a different approach
- Do NOT repeat the same failed action multiple times

## TOOLS AVAILABLE:
{tool_desc}

## FORMAT:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be valid JSON format)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer in table format

## CRITICAL: Action Input Format Rules
- Action Input must contain ONLY valid JSON
- DO NOT include the word "Observation" in Action Input
- DO NOT include any text after the JSON
- For tools with no parameters: use empty JSON object
- For tools with parameters: use JSON with proper key-value pairs
- The Action Input field should end immediately after the JSON, with no additional text

## MAINTENANCE CLASSIFICATION:
- ROUTINE_SURVEILLANCE (RUL > 100): LOW priority
- PROACTIVE_INSPECTION (50 < RUL <= 100): MEDIUM priority
- CORRECTIVE_ACTION (20 < RUL <= 49): HIGH priority
- IMMEDIATE_GROUNDING (RUL <= 20): CRITICAL priority

## EXAMPLES:
{examples}

Begin!

Question: {question}

{scratchpad}""",
)

optimized_reflect_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template="""You are an advanced reasoning agent specializing in predictive maintenance and RUL (Remaining Useful Life) prediction. You will be given a previous reasoning trial where you attempted to answer a question about industrial equipment maintenance, RUL prediction, or cost-benefit analysis.

Your task is to:
1. Diagnose why the previous attempt may have failed or could be improved
2. Identify any missing steps in the reasoning process
3. Devise a new, concise plan that addresses the shortcomings

Common failure modes to check for:
- Missing data loading steps (train_data, test_data, ground_truth)
- Incorrect tool usage or parameter mismatches
- Incomplete cost-benefit analysis (missing safety costs, labor rates)
- Failure to use search tools for current information (OSHA protocols, labor rates)
- Missing model training or prediction steps
- Incomplete engine risk assessment
- Lack of proper maintenance classification

Here are some examples:
{examples}

Previous trial:
Question: {question}
{scratchpad}

Reflection:""",
)

sample_utterance = "We should focus on the engines that are running on fumes, which ones are likely to give out in the next 20 cycles? Provide me a list of engine ids with safety recommendations and cost estimates."
# all_tools = rul_tools + watsonx_tools

"""Actually test out the agent"""

import sys
from pathlib import Path

# Need to change path to access the reactxen_agent
reactxen_src_path = Path(
    "/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/ReActXen/src"
)
if str(reactxen_src_path) not in sys.path:
    sys.path.insert(0, str(reactxen_src_path))

# Now import the function with the correct path
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from reactxen.utils.tool_description import get_tool_description, get_tool_names

tool_desc = get_tool_description(all_tools, detailed_desc=True)
tool_names = get_tool_names(all_tools)

# =============================================================================
# UPDATED AGENT CONFIG - Use default prompt or improved prompt
# =============================================================================

agent_config = {
    "question": sample_utterance,
    "key": "comprehensive_rul_analytics",
    "react_llm_model_id": 15,  # GRANITE_MODEL_ID=15
    "reflect_llm_model_id": 15,
    # Option 1: Use default prompt (recommended for now)
    # "agent_prompt": None,  # Will use default react_agent_prompt
    # Option 2: Use improved custom prompt
    "agent_prompt": improved_root_prompt,  # Uncomment to use custom prompt
    "reflect_prompt": optimized_reflect_prompt,
    "tools": all_tools,
    "tool_desc": tool_desc,
    "tool_names": tool_names,  # Make sure this is a list of strings, not tuples
    "actionstyle": "Text",
    "max_steps": 15,
    "num_reflect_iteration": 3,
    "early_stop": False,
    "debug": True,
    "reactstyle": "thought_and_act_together",
}

# =============================================================================
# VERIFY tool_names is a list of strings (not tuples)
# =============================================================================

# Debug: Check tool_names format
print("=" * 70)
print("🔍 DEBUGGING TOOL NAMES")
print("=" * 70)
print(f"Tool names type: {type(tool_names)}")
print(f"Tool names: {tool_names}")
print(f"First tool name type: {type(tool_names[0]) if tool_names else 'N/A'}")
print(f"First tool name: {tool_names[0] if tool_names else 'N/A'}")

# Ensure tool_names is a list of strings
if tool_names and isinstance(tool_names[0], tuple):
    print("⚠️  WARNING: tool_names contains tuples! Converting to strings...")
    tool_names = [
        str(name[0]) if isinstance(name, tuple) else str(name) for name in tool_names
    ]
    agent_config["tool_names"] = tool_names

print("=" * 70)
print("✅ FIXED AGENT CONFIGURATION")
print("=" * 70)
print(f"📊 Total tools: {len(all_tools)}")
print(f"📝 Tool descriptions: Generated ({len(tool_desc)} characters)")
print(f"📋 Tool names: {len(tool_names)} tools")
print(f"🎯 Action style: Text")
print(f"🤖 Model: Granite-3.2-8B-instruct (ID: 15)")
print(f"🔄 Reflection: Custom prompt enabled")
print("=" * 70)


# Create and run agent
root_agent = create_reactxen_agent(**agent_config)
root_agent.run()
