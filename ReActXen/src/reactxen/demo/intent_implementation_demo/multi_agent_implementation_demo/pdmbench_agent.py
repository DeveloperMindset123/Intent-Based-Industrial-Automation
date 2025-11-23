"""
PDMBench Agentic Implementation
Pre-defined sub-agents for: Fault Classification, RUL Prediction, Cost-Benefit Analysis, Safety/Policies
Simplified architecture with dynamic tool creation capability
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

# Setup paths
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
from langchain_core.prompts import PromptTemplate

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


class PDMBenchSubAgent:
    """Base class for PDMBench sub-agents."""
    
    def __init__(self, name: str, role: str, tools: List[BaseTool], model_id: int = 15):
        self.name = name
        self.role = role
        self.tools = tools
        self.model_id = model_id
        self.agent = None
    
    def create_agent(self, question: str):
        """Create the agent instance."""
        tool_names, tool_desc = get_tool_descriptions(self.tools)
        
        prompt = PromptTemplate(
            input_variables=["question", "tool_desc", "scratchpad"],
            template=f"""You are a {self.role}.

YOUR ROLE: {self.role}

TOOLS:
{{tool_desc}}

Use the following format:
Question: {{question}}
Thought: think about what to do
Action: tool_name
Action Input: JSON with parameters
Observation: result
... (repeat as needed)
Thought: I have completed the task
Action: Finish
Action Input: Your final answer

Begin!
Question: {{question}}
{{scratchpad}}"""
        )
        
        self.agent = create_reactxen_agent(
            question=question,
            key=self.name,
            agent_prompt=prompt,
            tools=self.tools,  # Use 'tools' not 'cbm_tools'
            tool_names=tool_names,
            tool_desc=tool_desc,
            react_llm_model_id=self.model_id,
            reflect_llm_model_id=self.model_id,
            max_steps=20,
            debug=False,
            apply_loop_detection_check=True,
            log_structured_messages=True  # Required when loop detection is enabled
        )
        return self.agent


def create_fault_classification_agent(model_id: int = 15) -> PDMBenchSubAgent:
    """Create fault classification sub-agent."""
    tools = [
        CodeExecutionTool(),
        *create_dataset_tools(),
        *create_watsonx_tools()
    ]
    return PDMBenchSubAgent(
        name="fault_classifier",
        role="Fault Classification Agent - Identifies and classifies equipment faults",
        tools=tools,
        model_id=model_id
    )


def create_rul_prediction_agent(model_id: int = 15) -> PDMBenchSubAgent:
    """Create RUL prediction sub-agent."""
    tools = [
        CodeExecutionTool(),
        *create_dataset_tools(),
        *create_watsonx_tools(),
        *create_ground_truth_tools()
    ]
    return PDMBenchSubAgent(
        name="rul_predictor",
        role="RUL Prediction Agent - Predicts Remaining Useful Life of equipment",
        tools=tools,
        model_id=model_id
    )


def create_cost_benefit_agent(model_id: int = 15) -> PDMBenchSubAgent:
    """Create cost-benefit analysis sub-agent."""
    tools = [
        CodeExecutionTool(),
        *create_dataset_tools()
    ]
    return PDMBenchSubAgent(
        name="cost_benefit_analyzer",
        role="Cost-Benefit Analysis Agent - Analyzes maintenance costs vs benefits",
        tools=tools,
        model_id=model_id
    )


def create_safety_policies_agent(model_id: int = 15) -> PDMBenchSubAgent:
    """Create safety/policies sub-agent."""
    tools = [
        CodeExecutionTool(),
        *create_dataset_tools(),
        *create_table_formatting_tools()
    ]
    return PDMBenchSubAgent(
        name="safety_policies",
        role="Safety and Policies Agent - Evaluates safety risks and policy compliance",
        tools=tools,
        model_id=model_id
    )


def create_pdmbench_root_agent(question: str, model_id: int = 15, use_openai: bool = False, openai_model: str = None) -> Any:
    """Create root agent with all PDMBench sub-agents."""
    # Create sub-agents
    sub_agents = {
        'fault_classifier': create_fault_classification_agent(model_id),
        'rul_predictor': create_rul_prediction_agent(model_id),
        'cost_benefit_analyzer': create_cost_benefit_agent(model_id),
        'safety_policies': create_safety_policies_agent(model_id)
    }
    
    # Get memory system
    memory = get_memory_system()
    memory_summary = memory.get_memory_summary() or "No previous memory recorded."
    
    # Create root agent tools
    all_tools = [
        CodeExecutionTool(),
        *create_dataset_tools(),
        *create_ground_truth_tools(),
        *create_table_formatting_tools()
    ]
    
    tool_names, tool_desc = get_tool_descriptions(all_tools)
    
    # Create agent tool wrappers for sub-agents
    from agent_tool_wrapper import AgentTool
    agent_tools = []
    for name, sub_agent in sub_agents.items():
        # Create a question for the sub-agent
        sub_question = f"Execute {sub_agent.role} tasks"
        sub_agent.create_agent(sub_question)
        if sub_agent.agent:
            agent_tool = AgentTool(
                agent=sub_agent.agent,
                name=name,
                description=sub_agent.role
            )
            agent_tools.append(agent_tool)
            all_tools.append(agent_tool)
    
    tool_names, tool_desc = get_tool_descriptions(all_tools)
    
    # Root agent prompt - embed memory_context directly in template to avoid formatting issues
    root_prompt_template = f"""You are a Root Agent coordinating PDMBench tasks: Fault Classification, RUL Prediction, Cost-Benefit Analysis, and Safety/Policies.

MEMORY CONTEXT:
{memory_summary}

AVAILABLE SUB-AGENTS:
- fault_classifier: For fault classification tasks
- rul_predictor: For RUL prediction tasks (MUST verify with ground truth)
- cost_benefit_analyzer: For cost-benefit analysis
- safety_policies: For safety and policy evaluation

TOOLS:
{{tool_desc}}

WORKFLOW:
1. Analyze the question to determine which sub-agent(s) to use
2. Delegate to appropriate sub-agent(s)
3. For RUL predictions: MUST use verify_rul_predictions tool
4. Format results using format_table tool
5. Provide comprehensive final answer

Format:
Question: {{question}}
Thought: analyze and delegate
Action: sub_agent_name or tool_name
Action Input: JSON with parameters
Observation: result
... (continue)
Thought: I have completed the task
Action: Finish
Action Input: Final comprehensive answer

Begin!
Question: {{question}}
{{scratchpad}}"""
    
    root_prompt = PromptTemplate(
        input_variables=["question", "tool_desc", "scratchpad"],
        template=root_prompt_template
    )
    
    root_agent = create_reactxen_agent(
        question=question,
        key="pdmbench_root",
        agent_prompt=root_prompt,
        tools=all_tools,  # Use 'tools' not 'cbm_tools'
        tool_names=tool_names,
        tool_desc=tool_desc,
        react_llm_model_id=model_id,
        reflect_llm_model_id=model_id,
        max_steps=40,
        debug=True,
        apply_loop_detection_check=True,
        log_structured_messages=True  # Required when loop detection is enabled
    )
    
    return root_agent, sub_agents

