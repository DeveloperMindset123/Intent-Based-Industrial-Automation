"""
Reflector Sub-Agent - Conducts step-by-step audit of reasoning chains.
Identifies hallucinations, circular reasoning, and skipped steps.
"""
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from memory_system import get_memory_system


class ReflectorAgentTool(BaseTool):
    """Tool that wraps the reflector agent for reasoning chain audit."""
    
    name: str = "reflector_audit"
    description: str = """Conduct a step-by-step audit of the reasoning chain.
    
    This tool analyzes the agent's reasoning process to:
    - Identify hallucinations (unsupported claims)
    - Detect circular reasoning
    - Find skipped steps
    - Validate evidence sources
    - Check for logical gaps
    
    Input: JSON with:
    - reasoning_chain: List of steps with thought, action, input, observation
    - question: Original question
    - current_step: Current step number
    
    Returns: Audit report with findings and recommendations.
    """
    
    class ReflectorInput(BaseModel):
        reasoning_chain: str = Field(description="JSON string of reasoning chain steps")
        question: str = Field(description="Original question")
        current_step: Optional[int] = Field(default=None, description="Current step number")
    
    args_schema: type[BaseModel] = ReflectorInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ for non-Pydantic attributes
        object.__setattr__(self, 'memory', get_memory_system())
    
    def _run(
        self,
        reasoning_chain: str,
        question: str,
        current_step: Optional[int] = None
    ) -> str:
        """Run reflector audit."""
        try:
            # Parse reasoning chain
            if isinstance(reasoning_chain, str):
                chain_data = json.loads(reasoning_chain)
            else:
                chain_data = reasoning_chain
            
            # Create audit report
            audit_report = {
                "question": question,
                "current_step": current_step,
                "total_steps": len(chain_data) if isinstance(chain_data, list) else 0,
                "findings": [],
                "hallucinations": [],
                "circular_reasoning": [],
                "skipped_steps": [],
                "evidence_issues": [],
                "logical_gaps": [],
                "recommendations": []
            }
            
            # Analyze each step
            if isinstance(chain_data, list):
                for i, step in enumerate(chain_data):
                    step_analysis = self._analyze_step(step, i, question)
                    
                    if step_analysis["has_hallucination"]:
                        audit_report["hallucinations"].append({
                            "step": i,
                            "issue": step_analysis["hallucination_reason"]
                        })
                    
                    if step_analysis["is_circular"]:
                        audit_report["circular_reasoning"].append({
                            "step": i,
                            "issue": step_analysis["circular_reason"]
                        })
                    
                    if step_analysis["missing_evidence"]:
                        audit_report["evidence_issues"].append({
                            "step": i,
                            "issue": step_analysis["evidence_reason"]
                        })
            
            # Check for skipped steps
            audit_report["skipped_steps"] = self._detect_skipped_steps(chain_data, question)
            
            # Check for logical gaps
            audit_report["logical_gaps"] = self._detect_logical_gaps(chain_data)
            
            # Generate recommendations
            audit_report["recommendations"] = self._generate_recommendations(audit_report)
            
            # Format as readable report
            report_text = self._format_audit_report(audit_report)
            
            # Record findings in memory
            if audit_report["hallucinations"] or audit_report["circular_reasoning"]:
                self.memory.add_heuristic(
                    f"When reasoning, ensure each step has evidence from observations. Avoid circular reasoning.",
                    priority=2
                )
            
            return report_text
            
        except Exception as e:
            return f"Error in reflector audit: {str(e)}"
    
    def _analyze_step(self, step: Dict[str, Any], step_num: int, question: str) -> Dict[str, Any]:
        """Analyze a single reasoning step."""
        analysis = {
            "has_hallucination": False,
            "hallucination_reason": "",
            "is_circular": False,
            "circular_reason": "",
            "missing_evidence": False,
            "evidence_reason": ""
        }
        
        thought = step.get("thought", "").lower()
        action = step.get("action", "").lower()
        observation = step.get("observation", "").lower()
        
        # Check for hallucinations (claims without evidence)
        if "conclude" in thought or "determine" in thought:
            if not observation or "error" in observation:
                analysis["has_hallucination"] = True
                analysis["hallucination_reason"] = "Conclusion made without valid observation"
        
        # Check for circular reasoning
        if step_num > 0:
            prev_thought = step.get("previous_thought", "").lower()
            if thought == prev_thought and action == step.get("previous_action", "").lower():
                analysis["is_circular"] = True
                analysis["circular_reason"] = "Repeating same thought and action"
        
        # Check for missing evidence
        if "based on" in thought or "according to" in thought:
            if not observation or len(observation) < 20:
                analysis["missing_evidence"] = True
                analysis["evidence_reason"] = "Claim references evidence but observation is insufficient"
        
        return analysis
    
    def _detect_skipped_steps(self, chain: List[Dict], question: str) -> List[Dict[str, Any]]:
        """Detect potentially skipped steps."""
        skipped = []
        
        # Check for jumps in reasoning
        for i in range(len(chain) - 1):
            current = chain[i]
            next_step = chain[i + 1]
            
            current_action = current.get("action", "").lower()
            next_action = next_step.get("action", "").lower()
            
            # If action changes dramatically without intermediate step
            if current_action and next_action and current_action != next_action:
                # Check if there's a logical gap
                current_obs = current.get("observation", "").lower()
                next_thought = next_step.get("thought", "").lower()
                
                # If next thought doesn't reference current observation
                if current_obs and next_thought:
                    if not any(word in next_thought for word in current_obs.split()[:5]):
                        skipped.append({
                            "between_steps": f"{i} and {i+1}",
                            "issue": "Reasoning jump without connecting observation to next thought"
                        })
        
        return skipped
    
    def _detect_logical_gaps(self, chain: List[Dict]) -> List[Dict[str, Any]]:
        """Detect logical gaps in reasoning."""
        gaps = []
        
        # Check for missing validation steps
        has_predictions = any("predict" in str(step).lower() for step in chain)
        has_verification = any("verify" in str(step).lower() or "ground truth" in str(step).lower() for step in chain)
        
        if has_predictions and not has_verification:
            gaps.append({
                "issue": "Predictions made but no verification step found",
                "recommendation": "Add verification step using verify_rul_predictions tool"
            })
        
        # Check for missing formatting
        has_data = any("equipment" in str(step).lower() or "rul" in str(step).lower() for step in chain)
        has_formatting = any("format" in str(step).lower() or "table" in str(step).lower() for step in chain)
        
        if has_data and not has_formatting:
            gaps.append({
                "issue": "Data collected but no formatting step found",
                "recommendation": "Add formatting step using format_table tool"
            })
        
        return gaps
    
    def _generate_recommendations(self, audit_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on audit findings."""
        recommendations = []
        
        if audit_report["hallucinations"]:
            recommendations.append(
                "Ensure each conclusion is supported by valid observations. "
                "If an observation contains an error, address the error before proceeding."
            )
        
        if audit_report["circular_reasoning"]:
            recommendations.append(
                "Avoid repeating the same thought and action. Try a different approach or "
                "use a different tool if the current one is not working."
            )
        
        if audit_report["skipped_steps"]:
            recommendations.append(
                "Ensure each step logically follows from the previous observation. "
                "Don't skip intermediate reasoning steps."
            )
        
        if audit_report["logical_gaps"]:
            for gap in audit_report["logical_gaps"]:
                if "recommendation" in gap:
                    recommendations.append(gap["recommendation"])
        
        if audit_report["evidence_issues"]:
            recommendations.append(
                "When making claims, ensure you have sufficient evidence from observations. "
                "If evidence is missing, gather it first before making conclusions."
            )
        
        if not recommendations:
            recommendations.append("Reasoning chain appears sound. Continue with current approach.")
        
        return recommendations
    
    def _format_audit_report(self, report: Dict[str, Any]) -> str:
        """Format audit report as readable text."""
        lines = []
        lines.append("=" * 70)
        lines.append("REFLECTOR AUDIT REPORT")
        lines.append("=" * 70)
        lines.append(f"Question: {report['question']}")
        lines.append(f"Total Steps Analyzed: {report['total_steps']}")
        lines.append("")
        
        if report["hallucinations"]:
            lines.append("⚠️  HALLUCINATIONS DETECTED:")
            for h in report["hallucinations"]:
                lines.append(f"  Step {h['step']}: {h['issue']}")
            lines.append("")
        
        if report["circular_reasoning"]:
            lines.append("⚠️  CIRCULAR REASONING DETECTED:")
            for c in report["circular_reasoning"]:
                lines.append(f"  Step {c['step']}: {c['issue']}")
            lines.append("")
        
        if report["skipped_steps"]:
            lines.append("⚠️  SKIPPED STEPS DETECTED:")
            for s in report["skipped_steps"]:
                lines.append(f"  Between steps {s['between_steps']}: {s['issue']}")
            lines.append("")
        
        if report["evidence_issues"]:
            lines.append("⚠️  EVIDENCE ISSUES:")
            for e in report["evidence_issues"]:
                lines.append(f"  Step {e['step']}: {e['issue']}")
            lines.append("")
        
        if report["logical_gaps"]:
            lines.append("⚠️  LOGICAL GAPS:")
            for g in report["logical_gaps"]:
                lines.append(f"  - {g['issue']}")
            lines.append("")
        
        lines.append("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def create_reflector_agent_tool(parent_model_id: int = 15) -> ReflectorAgentTool:
    """Create a reflector agent tool."""
    return ReflectorAgentTool()

