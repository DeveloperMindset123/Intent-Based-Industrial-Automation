"""
Learning Sub-Agent - Learns from mistakes and builds heuristics.
Develops cache-based approaches and optimization strategies.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from memory_system import get_memory_system


class LearningAgentTool(BaseTool):
    """Tool that wraps the learning agent for heuristic development."""
    
    name: str = "learning_analyze"
    description: str = """Analyze mistakes and build heuristics for improvement.
    
    This tool:
    - Analyzes recent mistakes and patterns
    - Suggests heuristics to prevent repeated errors
    - Provides optimization recommendations
    - Suggests alternative approaches
    
    Input: JSON with:
    - action: Action that failed
    - error: Error message
    - context: Additional context (optional)
    
    Returns: Analysis with heuristics and recommendations.
    """
    
    class LearningInput(BaseModel):
        action: str = Field(description="Action that failed or needs analysis")
        error: Optional[str] = Field(default=None, description="Error message if any")
        context: Optional[str] = Field(default=None, description="Additional context")
    
    args_schema: type[BaseModel] = LearningInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ for non-Pydantic attributes
        object.__setattr__(self, 'memory', get_memory_system())
    
    def _run(
        self,
        action: str,
        error: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """Run learning analysis."""
        try:
            analysis = {
                "action": action,
                "error": error,
                "similar_mistakes": [],
                "similar_solutions": [],
                "heuristics": [],
                "recommendations": [],
                "alternative_approaches": []
            }
            
            # Find similar mistakes
            if error:
                similar_mistakes = self.memory.get_similar_mistakes(action, error)
                analysis["similar_mistakes"] = similar_mistakes
            
            # Find similar solutions
            similar_solutions = self.memory.get_similar_solutions(action)
            analysis["similar_solutions"] = similar_solutions
            
            # Get relevant heuristics
            heuristics = self.memory.get_relevant_heuristics(action, error)
            analysis["heuristics"] = heuristics
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            # Suggest alternatives
            analysis["alternative_approaches"] = self._suggest_alternatives(action, error)
            
            # Format as readable report
            report_text = self._format_analysis(analysis)
            
            return report_text
            
        except Exception as e:
            return f"Error in learning analysis: {str(e)}"
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # If similar mistakes found, suggest avoiding the pattern
        if analysis["similar_mistakes"]:
            count = len(analysis["similar_mistakes"])
            recommendations.append(
                f"This action has failed {count} times with similar errors. "
                "Consider using a different approach or tool."
            )
        
        # If similar solutions found, suggest trying them
        if analysis["similar_solutions"]:
            solutions = analysis["similar_solutions"]
            recommendations.append(
                f"Found {len(solutions)} successful solutions for similar actions. "
                "Consider adapting one of these approaches."
            )
        
        # Action-specific recommendations
        action_lower = analysis["action"].lower()
        error_lower = (analysis["error"] or "").lower()
        
        if "create_dynamic_tool" in action_lower:
            if "missing" in error_lower and "key" in error_lower:
                recommendations.append(
                    "When creating dynamic tools, ensure JSON input has all required keys: "
                    "tool_name, tool_description, code. Use execute_python_code first to test code."
                )
        
        if "execute_python_code" in action_lower:
            if "axis" in error_lower or "=" in error_lower:
                recommendations.append(
                    "Avoid using axis=1 in pandas operations. Use loops or vectorization instead. "
                    "Example: for idx, row in df.iterrows(): instead of df.apply(..., axis=1)"
                )
        
        if "create_sub_agent" in action_lower:
            if "error" in error_lower:
                recommendations.append(
                    "When creating sub-agents, ensure all required parameters are provided: "
                    "agent_name, role, workflow, tools_description. Start with simple agents first."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "Start with simpler approaches and gradually increase complexity. "
                "Test each step before moving to the next."
            )
        
        return recommendations
    
    def _suggest_alternatives(self, action: str, error: Optional[str]) -> List[str]:
        """Suggest alternative approaches."""
        alternatives = []
        action_lower = action.lower()
        error_lower = (error or "").lower()
        
        # Alternatives for create_dynamic_tool
        if "create_dynamic_tool" in action_lower:
            alternatives.append(
                "Instead of create_dynamic_tool, try using execute_python_code first to test the logic, "
                "then create the tool once the code works."
            )
            alternatives.append(
                "Consider if a pre-existing tool can accomplish the task instead of creating a new one."
            )
        
        # Alternatives for execute_python_code
        if "execute_python_code" in action_lower:
            if "json" in error_lower or "parse" in error_lower:
                alternatives.append(
                    "Break complex code into smaller execute_python_code calls. "
                    "Ensure JSON strings are properly escaped (use \\n for newlines)."
                )
        
        # Alternatives for sub-agent creation
        if "create_sub_agent" in action_lower:
            alternatives.append(
                "Instead of creating a sub-agent, try using direct tools first. "
                "Only create sub-agents if the task is complex and requires specialized handling."
            )
        
        # General alternatives
        if not alternatives:
            alternatives.append(
                "Try a divide-and-conquer approach: break the task into smaller steps, "
                "complete each step successfully before moving to the next."
            )
            alternatives.append(
                "Use search tools to find examples or documentation for the task."
            )
        
        return alternatives
    
    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("LEARNING ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Action: {analysis['action']}")
        if analysis["error"]:
            lines.append(f"Error: {analysis['error']}")
        lines.append("")
        
        if analysis["similar_mistakes"]:
            lines.append("⚠️  SIMILAR PAST MISTAKES:")
            for mistake in analysis["similar_mistakes"][:3]:
                lines.append(f"  - {mistake.get('error', 'Unknown error')[:100]}")
            lines.append("")
        
        if analysis["similar_solutions"]:
            lines.append("✅ SIMILAR PAST SOLUTIONS:")
            for solution in analysis["similar_solutions"][:3]:
                lines.append(f"  - Action: {solution.get('action', 'Unknown')}")
                lines.append(f"    Result: {solution.get('result', 'Unknown')[:100]}")
            lines.append("")
        
        if analysis["heuristics"]:
            lines.append("💡 RELEVANT HEURISTICS:")
            for i, heuristic in enumerate(analysis["heuristics"][:5], 1):
                lines.append(f"  {i}. {heuristic}")
            lines.append("")
        
        if analysis["recommendations"]:
            lines.append("📋 RECOMMENDATIONS:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        if analysis["alternative_approaches"]:
            lines.append("🔄 ALTERNATIVE APPROACHES:")
            for i, alt in enumerate(analysis["alternative_approaches"], 1):
                lines.append(f"  {i}. {alt}")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def create_learning_agent_tool() -> LearningAgentTool:
    """Create a learning agent tool."""
    return LearningAgentTool()

