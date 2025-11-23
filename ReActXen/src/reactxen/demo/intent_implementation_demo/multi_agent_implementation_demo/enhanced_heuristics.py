"""
Enhanced Heuristics Learning System
Automatically learns and updates heuristics from mistakes and successes.
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
from memory_system import get_memory_system


class HeuristicLearner:
    """Automatically learns heuristics from agent behavior."""
    
    # Common initial heuristics
    INITIAL_HEURISTICS = [
        "Tool with NO parameters: Use {} as Action Input",
        "Tool with parameters: Use JSON format with parameter keys and values as dictionary",
        "ALWAYS check if data is loaded first using check_data_loaded tool",
        "Use quick_data_summary before loading datasets to avoid redundant operations",
        "For RUL predictions: MUST call verify_rul_predictions tool after predictions",
        "If same error repeats 3 times, stop and use learning_analyze",
        "Validate JSON input format before calling tools",
        "Keep reasoning concise - avoid repeating same thoughts",
        "Delegate to sub-agents for complex specialized tasks",
        "Use format_table tool for final results presentation",
        "Check data availability FIRST before attempting operations",
        "If data already loaded, skip redundant loading steps",
        "Be concise - avoid verbose explanations",
        "Focus on actionable steps",
        "Stop immediately if you detect a loop (same action repeated)",
        "For Python code execution: Ensure code key is present in JSON input",
        "When parsing fails: Check if input is valid JSON string",
        "For dataset operations: Use list_datasets before load_dataset",
        "For RUL predictions: Always include ground truth verification",
        "For cost analysis: Include both maintenance and failure costs",
        "CRITICAL: For large data structures (>50 items): Use execute_code_simple instead of verify_rul_predictions or format_table to avoid JSON truncation",
        "CRITICAL: When using execute_code_simple: Always store results in variables and print them, e.g., 'result = verify_rul_predictions(predictions); print(result)'",
        "If verify_rul_predictions receives truncated JSON: Use execute_code_simple with code that calls verify_rul_predictions(predictions_dict) directly",
        "If format_table receives truncated JSON: Use execute_code_simple with code that calls format_table functions directly",
        "CRITICAL: After load_dataset, access data using: from shared.load_data import train_data, test_data, ground_truth. DO NOT read CSV files directly.",
        "CRITICAL: For RUL prediction tasks, ground truth validation is REQUIRED. Use verify_rul_predictions tool or execute_code_simple to verify predictions against ground_truth.",
        "CRITICAL: DO NOT use 'rul_prediction_model' or 'set_model_id' with custom model names. These models don't exist. For RUL prediction, use execute_code_simple or execute_python_code to create predictions directly from the data.",
        "CRITICAL: If predict_rul tool fails with 'Model not supported', IMMEDIATELY switch to execute_code_simple to create RUL predictions using Python code. Do NOT try to set_model_id or train_model.",
        "For RUL prediction: Use execute_code_simple with code that: 1) imports train_data, test_data, ground_truth from shared.load_data, 2) Creates predictions dict, 3) Verifies with verify_rul_predictions(predictions, ground_truth), 4) Prints results"
    ]
    
    def __init__(self):
        self.memory = get_memory_system()
        self._initialize_heuristics()
    
    def _initialize_heuristics(self):
        """Initialize with common heuristics if memory is empty."""
        if not self.memory.memory.get("heuristics"):
            for heuristic in self.INITIAL_HEURISTICS:
                self.memory.add_heuristic(heuristic, priority=1)
    
    def learn_from_mistake(self, action: str, error: str, context: Optional[str] = None) -> List[str]:
        """Learn heuristics from a mistake."""
        learned = []
        error_lower = error.lower()
        action_lower = action.lower()
        
        # Pattern: execute_python_code with missing 'code' key
        if "execute_python_code" in action_lower:
            if "missing" in error_lower or "code" in error_lower or "empty" in error_lower:
                heuristic = "CRITICAL: If execute_python_code fails with JSON parsing errors, IMMEDIATELY switch to execute_code_simple tool. Do NOT retry execute_python_code."
                self.memory.add_heuristic(heuristic, priority=1)  # Highest priority
                learned.append(heuristic)
            
            if "not a dict" in error_lower or "parse" in error_lower or "json" in error_lower:
                heuristic = "CRITICAL: If execute_python_code fails with JSON parsing, use execute_code_simple instead. Do NOT retry the same action."
                self.memory.add_heuristic(heuristic, priority=1)  # Highest priority
                learned.append(heuristic)
        
        # Pattern: create_dynamic_tool errors
        if "create_dynamic_tool" in action_lower:
            if "missing" in error_lower:
                heuristic = "For create_dynamic_tool: Ensure JSON has 'tool_name', 'tool_description', and 'code' keys"
                self.memory.add_heuristic(heuristic, priority=2)
                learned.append(heuristic)
        
        # Pattern: Repeated actions (loops)
        recent_mistakes = [m for m in self.memory.memory.get("mistakes", [])[-5:] 
                          if m.get("action") == action]
        if len(recent_mistakes) >= 3:
            heuristic = f"If {action} fails repeatedly (3+ times), use learning_analyze tool to find alternative approach"
            self.memory.add_heuristic(heuristic, priority=3)
            learned.append(heuristic)
        
        # Pattern: Data loading without checking
        if "load_dataset" in action_lower:
            if "already" in error_lower or "exists" in error_lower:
                heuristic = "Before load_dataset: Use check_data_loaded to avoid redundant loading"
                self.memory.add_heuristic(heuristic, priority=2)
                learned.append(heuristic)
        
        # Pattern: RUL prediction without verification
        if "predict_rul" in action_lower and "verify" not in context.lower() if context else True:
            heuristic = "After predict_rul: Always call verify_rul_predictions to validate against ground truth"
            self.memory.add_heuristic(heuristic, priority=3)
            learned.append(heuristic)
        
        return learned
    
    def learn_from_success(self, action: str, result: str, context: Optional[str] = None) -> List[str]:
        """Learn heuristics from a successful action."""
        learned = []
        result_lower = result.lower()
        action_lower = action.lower()
        
        # Pattern: Successful data checking before loading
        if "check_data_loaded" in action_lower:
            if "loaded" in result_lower or "available" in result_lower:
                heuristic = "Checking data availability first (check_data_loaded) avoids redundant operations - GOOD PRACTICE"
                self.memory.add_heuristic(heuristic, priority=1)
                learned.append(heuristic)
        
        # Pattern: Successful RUL verification
        if "verify_rul" in action_lower:
            if "accuracy" in result_lower or "verified" in result_lower:
                heuristic = "Verifying RUL predictions against ground truth ensures accuracy - REQUIRED for RUL tasks"
                self.memory.add_heuristic(heuristic, priority=3)
                learned.append(heuristic)
        
        # Pattern: Successful table formatting
        if "format_table" in action_lower:
            heuristic = "Using format_table for results improves readability - GOOD PRACTICE for final outputs"
            self.memory.add_heuristic(heuristic, priority=1)
            learned.append(heuristic)
        
        return learned
    
    def get_relevant_heuristics(self, action: str, context: Optional[str] = None) -> List[str]:
        """Get relevant heuristics for an action."""
        action_lower = action.lower()
        relevant = []
        
        # Get all heuristics
        all_heuristics = self.memory.memory.get("heuristics", [])
        
        # Filter by action relevance
        for h in all_heuristics:
            # Extract heuristic string from dict if needed
            if isinstance(h, dict):
                h_str = h.get("heuristic", "")
            else:
                h_str = str(h)
            
            h_lower = h_str.lower()
            
            # Direct action match
            if action_lower in h_lower:
                relevant.append(h_str)
            # Tool category match
            elif ("execute_python_code" in action_lower and "python" in h_lower) or \
                 ("dataset" in action_lower and "dataset" in h_lower) or \
                 ("rul" in action_lower and "rul" in h_lower):
                relevant.append(h_str)
            # General heuristics (always relevant)
            elif any(keyword in h_lower for keyword in ["json", "validate", "check", "format"]):
                if h_str not in relevant:
                    relevant.append(h_str)
        
        # Sort by priority (heuristics with action names first, then general)
        relevant.sort(key=lambda x: (
            0 if action_lower in x.lower() else 1,
            -x.count(action_lower)
        ))
        
        return relevant[:10]  # Return top 10 most relevant


def get_heuristic_learner() -> HeuristicLearner:
    """Get or create heuristic learner instance."""
    if not hasattr(get_heuristic_learner, '_instance'):
        get_heuristic_learner._instance = HeuristicLearner()
    return get_heuristic_learner._instance

