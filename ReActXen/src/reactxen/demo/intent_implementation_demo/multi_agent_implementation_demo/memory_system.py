"""
Memory and Heuristic Learning System
Stores past mistakes, solutions, and builds heuristics to prevent repeated errors.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class MemorySystem:
    """Manages memory of past mistakes, solutions, and heuristics."""
    
    def __init__(self, memory_file: Optional[Path] = None):
        """Initialize memory system with persistent storage."""
        if memory_file is None:
            memory_file = Path(__file__).parent / "outputs" / "agent_memory.json"
        self.memory_file = memory_file
        self.memory_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Load existing memory
        self.memory = self._load_memory()
        
        # In-memory caches for quick access
        self.error_patterns = defaultdict[Any, int](int)  # Count of error patterns
        self.success_patterns = defaultdict[Any, int](int)  # Count of success patterns
        self.heuristics = []  # Learned heuristics
        self.tool_usage_stats = defaultdict[Any, dict[str, int]](lambda: {"success": 0, "failure": 0})
        
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                return self._default_memory()
        return self._default_memory()
    
    def _default_memory(self) -> Dict[str, Any]:
        """Return default memory structure."""
        return {
            "mistakes": [],
            "solutions": [],
            "heuristics": [],
            "tool_usage": {},
            "error_patterns": {},
            "success_patterns": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def _save_memory(self):
        """Save memory to file."""
        self.memory["metadata"]["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")
    
    def record_mistake(
        self,
        action: str,
        action_input: str,
        error: str,
        context: Optional[str] = None,
        step_number: Optional[int] = None
    ):
        """Record a mistake for learning."""
        mistake = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "action_input": action_input[:500],  # Truncate long inputs
            "error": error[:500],
            "context": context,
            "step_number": step_number
        }
        
        self.memory["mistakes"].append(mistake)
        
        # Track error patterns
        error_key = f"{action}:{error[:100]}"
        self.error_patterns[error_key] += 1
        self.memory["error_patterns"][error_key] = self.error_patterns[error_key]
        
        # Update tool usage stats
        self.tool_usage_stats[action]["failure"] += 1
        
        # Check for repeated mistakes (loop detection)
        recent_mistakes = [m for m in self.memory["mistakes"][-10:] if m.get("action") == action]
        if len(recent_mistakes) >= 3:
            # Add high-priority heuristic to avoid this loop
            heuristic = f"CRITICAL LOOP DETECTED: {action} has failed {len(recent_mistakes)} times. STOP using this action and find alternative approach."
            self.add_heuristic(heuristic, priority=0)  # Highest priority
        
        self._save_memory()
    
    def record_solution(
        self,
        action: str,
        action_input: str,
        result: str,
        context: Optional[str] = None,
        step_number: Optional[int] = None
    ):
        """Record a successful solution."""
        solution = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "action_input": action_input[:500],
            "result": result[:500],
            "context": context,
            "step_number": step_number
        }
        
        self.memory["solutions"].append(solution)
        
        # Track success patterns
        success_key = f"{action}:success"
        self.success_patterns[success_key] += 1
        self.memory["success_patterns"][success_key] = self.success_patterns[success_key]
        
        # Update tool usage stats
        self.tool_usage_stats[action]["success"] += 1
        
        self._save_memory()
    
    def add_heuristic(self, heuristic: str, priority: int = 1):
        """Add a learned heuristic."""
        heuristic_entry = {
            "heuristic": heuristic,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "usage_count": 0
        }
        
        self.memory["heuristics"].append(heuristic_entry)
        self.heuristics.append(heuristic_entry)
        
        # Sort by priority
        self.memory["heuristics"].sort(key=lambda x: x["priority"], reverse=True)
        self.heuristics.sort(key=lambda x: x["priority"], reverse=True)
        
        self._save_memory()
    
    def get_relevant_heuristics(self, action: str, error: Optional[str] = None) -> List[str]:
        """Get relevant heuristics for a given action/error."""
        relevant = []
        
        # Check for action-specific heuristics
        for h in self.heuristics:
            if action.lower() in h["heuristic"].lower():
                relevant.append(h["heuristic"])
        
        # Check for error-specific heuristics
        if error:
            for h in self.heuristics:
                if any(keyword in error.lower() for keyword in ["json", "parse", "missing", "key", "format"]):
                    if "json" in h["heuristic"].lower() or "format" in h["heuristic"].lower():
                        relevant.append(h["heuristic"])
        
        # Get top priority heuristics
        top_heuristics = [h["heuristic"] for h in self.heuristics[:5]]
        relevant.extend(top_heuristics)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_relevant = []
        for h in relevant:
            if h not in seen:
                seen.add(h)
                unique_relevant.append(h)
        
        return unique_relevant[:10]  # Limit to 10 most relevant
    
    def get_similar_mistakes(self, action: str, error: str) -> List[Dict[str, Any]]:
        """Find similar past mistakes."""
        similar = []
        error_lower = error.lower()
        
        for mistake in self.memory["mistakes"][-50:]:  # Check last 50 mistakes
            if mistake["action"] == action:
                mistake_error = mistake["error"].lower()
                # Simple similarity check
                if any(keyword in mistake_error for keyword in error_lower.split()[:3]):
                    similar.append(mistake)
        
        return similar[:5]  # Return top 5 similar mistakes
    
    def get_similar_solutions(self, action: str) -> List[Dict[str, Any]]:
        """Find similar past solutions."""
        similar = []
        
        for solution in self.memory["solutions"][-50:]:  # Check last 50 solutions
            if solution["action"] == action:
                similar.append(solution)
        
        return similar[:5]  # Return top 5 similar solutions
    
    def should_avoid_action(self, action: str, action_input: str) -> tuple[bool, str]:
        """Check if an action should be avoided based on past mistakes."""
        # Check if this exact action+input combination failed recently
        recent_mistakes = [m for m in self.memory["mistakes"][-10:] 
                          if m["action"] == action and m["action_input"] == action_input]
        
        if len(recent_mistakes) >= 3:
            return True, f"This exact action+input combination has failed {len(recent_mistakes)} times recently. Try a different approach."
        
        # Check error patterns
        error_key = f"{action}:"
        action_errors = {k: v for k, v in self.memory["error_patterns"].items() 
                        if k.startswith(error_key)}
        
        if action_errors:
            max_errors = max(action_errors.values())
            if max_errors >= 5:
                return True, f"Action '{action}' has failed {max_errors} times with similar errors. Consider alternative approach."
        
        return False, ""
    
    def get_tool_success_rate(self, tool_name: str) -> float:
        """Get success rate for a tool."""
        stats = self.tool_usage_stats.get(tool_name, {"success": 0, "failure": 0})
        total = stats["success"] + stats["failure"]
        if total == 0:
            return 0.5  # Default to 50% if no data
        return stats["success"] / total
    
    def get_memory_summary(self) -> str:
        """Get a summary of memory for agent context."""
        summary = []
        summary.append("MEMORY SUMMARY:")
        summary.append(f"- Total Mistakes Recorded: {len(self.memory['mistakes'])}")
        summary.append(f"- Total Solutions Recorded: {len(self.memory['solutions'])}")
        summary.append(f"- Learned Heuristics: {len(self.memory['heuristics'])}")
        
        # Top error patterns
        if self.memory["error_patterns"]:
            top_errors = sorted(
                self.memory["error_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            summary.append("\nTop Error Patterns:")
            for pattern, count in top_errors:
                summary.append(f"  - {pattern}: {count} occurrences")
        
        # Top heuristics
        if self.heuristics:
            summary.append("\nKey Heuristics:")
            for h in self.heuristics[:5]:
                summary.append(f"  - {h['heuristic']}")
        
        return "\n".join(summary)
    
    def clear_old_memory(self, days: int = 30):
        """Clear memory older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        # Filter mistakes
        self.memory["mistakes"] = [
            m for m in self.memory["mistakes"]
            if datetime.fromisoformat(m["timestamp"]).timestamp() > cutoff
        ]
        
        # Filter solutions
        self.memory["solutions"] = [
            s for s in self.memory["solutions"]
            if datetime.fromisoformat(s["timestamp"]).timestamp() > cutoff
        ]
        
        self._save_memory()


# Global memory instance
_global_memory: Optional[MemorySystem] = None


def get_memory_system() -> MemorySystem:
    """Get or create global memory system instance."""
    global _global_memory
    if _global_memory is None:
        _global_memory = MemorySystem()
    return _global_memory

