"""
Architecture Visualization and Initialization
Provides visual representation of agent architecture and initialization tracking.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ArchitectureVisualizer:
    """Visualizes and tracks agent architecture initialization."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "outputs" / "architecture"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.architecture = {
            "root_agent": {},
            "sub_agents": [],
            "tools": [],
            "initialization_log": [],
            "timestamp": datetime.now().isoformat()
        }
    
    def log_initialization(self, component: str, details: Dict[str, Any]):
        """Log component initialization."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "details": details,
            "status": "initialized"
        }
        self.architecture["initialization_log"].append(entry)
        self._save_architecture()
    
    def register_root_agent(self, agent_config: Dict[str, Any]):
        """Register root agent configuration."""
        self.architecture["root_agent"] = {
            "question": agent_config.get("question", ""),
            "model": agent_config.get("model_type", "unknown"),
            "model_id": agent_config.get("model_id", None),
            "timeout": agent_config.get("timeout", 180),
            "max_steps": agent_config.get("max_steps", 15),
            "num_reflections": agent_config.get("num_reflect_iteration", 2),
            "tools_count": len(agent_config.get("tools", [])),
            "timestamp": datetime.now().isoformat()
        }
        self.log_initialization("root_agent", self.architecture["root_agent"])
    
    def register_sub_agent(self, agent_name: str, agent_type: str, tools: List[str]):
        """Register a sub-agent."""
        sub_agent = {
            "name": agent_name,
            "type": agent_type,
            "tools": tools,
            "timestamp": datetime.now().isoformat()
        }
        self.architecture["sub_agents"].append(sub_agent)
        self.log_initialization(f"sub_agent_{agent_name}", sub_agent)
    
    def register_tools(self, tools: List[Any]):
        """Register all tools."""
        tool_names = [getattr(tool, 'name', str(tool)) for tool in tools]
        self.architecture["tools"] = tool_names
        self.log_initialization("tools", {"count": len(tools), "tools": tool_names})
    
    def generate_architecture_summary(self) -> str:
        """Generate human-readable architecture summary."""
        lines = []
        lines.append("=" * 80)
        lines.append("AGENT ARCHITECTURE SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Root Agent
        root = self.architecture.get("root_agent", {})
        lines.append("ROOT AGENT:")
        lines.append(f"  Question: {root.get('question', 'N/A')[:80]}...")
        lines.append(f"  Model: {root.get('model', 'N/A')} ({root.get('model_id', 'N/A')})")
        lines.append(f"  Timeout: {root.get('timeout', 'N/A')}s")
        lines.append(f"  Max Steps: {root.get('max_steps', 'N/A')}")
        lines.append(f"  Reflections: {root.get('num_reflections', 'N/A')}")
        lines.append(f"  Tools: {root.get('tools_count', 'N/A')}")
        lines.append("")
        
        # Sub-Agents
        sub_agents = self.architecture.get("sub_agents", [])
        if sub_agents:
            lines.append("SUB-AGENTS:")
            for sub in sub_agents:
                lines.append(f"  - {sub.get('name', 'Unknown')} ({sub.get('type', 'Unknown')})")
                lines.append(f"    Tools: {', '.join(sub.get('tools', [])[:5])}")
            lines.append("")
        
        # Tools
        tools = self.architecture.get("tools", [])
        if tools:
            lines.append(f"TOOLS ({len(tools)}):")
            # Group tools by category
            tool_categories = {
                "Data": [t for t in tools if any(kw in t.lower() for kw in ["dataset", "data", "load"])],
                "RUL": [t for t in tools if any(kw in t.lower() for kw in ["rul", "predict", "verify"])],
                "Analysis": [t for t in tools if any(kw in t.lower() for kw in ["cost", "benefit", "analysis"])],
                "Code": [t for t in tools if any(kw in t.lower() for kw in ["code", "execute", "python"])],
                "Learning": [t for t in tools if any(kw in t.lower() for kw in ["learning", "reflector", "audit"])],
                "Other": []
            }
            
            # Add remaining tools to Other
            categorized = set()
            for cat_tools in tool_categories.values():
                categorized.update(cat_tools)
            tool_categories["Other"] = [t for t in tools if t not in categorized]
            
            for category, cat_tools in tool_categories.items():
                if cat_tools:
                    lines.append(f"  {category}: {', '.join(cat_tools[:8])}")
                    if len(cat_tools) > 8:
                        lines.append(f"    ... and {len(cat_tools) - 8} more")
            lines.append("")
        
        # Initialization Log
        init_log = self.architecture.get("initialization_log", [])
        if init_log:
            lines.append(f"INITIALIZATION LOG ({len(init_log)} entries):")
            for entry in init_log[-5:]:  # Show last 5
                lines.append(f"  [{entry.get('timestamp', '')[:19]}] {entry.get('component', 'Unknown')}: {entry.get('status', 'Unknown')}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _save_architecture(self):
        """Save architecture to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch_file = self.output_dir / f"architecture_{timestamp}.json"
        
        try:
            with open(arch_file, 'w') as f:
                json.dump(self.architecture, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save architecture: {e}")
    
    def save_visualization(self, format: str = "text") -> Path:
        """Save architecture visualization."""
        if format == "text":
            summary = self.generate_architecture_summary()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_file = self.output_dir / f"architecture_summary_{timestamp}.txt"
            with open(viz_file, 'w') as f:
                f.write(summary)
            return viz_file
        else:
            # Could add JSON, DOT (graphviz), or other formats
            return self._save_architecture()


def get_architecture_visualizer() -> ArchitectureVisualizer:
    """Get or create architecture visualizer instance."""
    if not hasattr(get_architecture_visualizer, '_instance'):
        get_architecture_visualizer._instance = ArchitectureVisualizer()
    return get_architecture_visualizer._instance

