#!/usr/bin/env python3
"""
PHMForge Architecture Diagram Generator
Generates a publication-quality architecture diagram for the introduction section.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set up the figure with high DPI for publication quality
fig, ax = plt.subplots(1, 1, figsize=(14, 8.5), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.set_aspect('equal')
ax.axis('off')

# Color palette (professional, accessible)
colors = {
    'query': '#E8F4FD',       # Light blue
    'query_border': '#1976D2', # Blue
    'agent': '#FFF3E0',        # Light orange
    'agent_border': '#F57C00', # Orange
    'mcp': '#E8F5E9',          # Light green
    'mcp_border': '#388E3C',   # Green
    'tools': '#F3E5F5',        # Light purple
    'tools_border': '#7B1FA2', # Purple
    'eval': '#FFEBEE',         # Light red
    'eval_border': '#D32F2F',  # Red
    'metrics': '#E0F7FA',      # Light cyan
    'metrics_border': '#0097A7', # Cyan
    'arrow': '#455A64',        # Dark gray
    'text': '#212121',         # Near black
}

def draw_rounded_box(ax, x, y, width, height, facecolor, edgecolor, label, sublabels=None, fontsize=10):
    """Draw a rounded rectangle with label and optional sublabels."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2,
        zorder=1
    )
    ax.add_patch(box)

    # Main label
    if sublabels:
        ax.text(x + width/2, y + height - 0.35, label,
                ha='center', va='top', fontsize=fontsize, fontweight='bold', color=colors['text'])
        # Sublabels
        sublabel_y = y + height - 0.7
        for i, sub in enumerate(sublabels):
            ax.text(x + width/2, sublabel_y - i*0.28, sub,
                    ha='center', va='top', fontsize=fontsize-2, color=colors['text'], style='italic')
    else:
        ax.text(x + width/2, y + height/2, label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold', color=colors['text'])

    return box

def draw_arrow(ax, start, end, color='#455A64', style='simple', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=15,
        color=color,
        linewidth=2,
        connectionstyle=connectionstyle,
        zorder=2
    )
    ax.add_patch(arrow)
    return arrow

# =============================================================================
# Adjusted Y positions - shifted everything up to make room for bottom boxes
# =============================================================================
main_y_offset = 0.8  # Reduced offset to bring bottom boxes closer

# =============================================================================
# Layer 1: Natural Language Query (Left side)
# =============================================================================
query_box = draw_rounded_box(ax, 0.3, 3.5 + main_y_offset, 2.4, 2.0,
                              colors['query'], colors['query_border'],
                              'Natural Language\nQuery',
                              sublabels=['"Which engines need', 'immediate maintenance?"'])

# =============================================================================
# Layer 2: Agentic Frameworks (Center-left)
# =============================================================================
# Main agent container - FIXED: increased height to 4.2 to contain LLM text
agent_container = FancyBboxPatch(
    (3.2, 2.3 + main_y_offset), 2.6, 4.2,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=colors['agent'],
    edgecolor=colors['agent_border'],
    linewidth=2,
    zorder=1
)
ax.add_patch(agent_container)
ax.text(4.5, 6.2 + main_y_offset, 'Agentic Frameworks', ha='center', va='center',
        fontsize=11, fontweight='bold', color=colors['text'])

# Individual agents - adjusted positions
agents = ['ReAct', 'Cursor Agent', 'Claude Code']
agent_y_positions = [5.5 + main_y_offset, 4.7 + main_y_offset, 3.9 + main_y_offset]
for agent, y_pos in zip(agents, agent_y_positions):
    small_box = FancyBboxPatch(
        (3.5, y_pos - 0.3), 2.0, 0.6,
        boxstyle="round,pad=0.01,rounding_size=0.1",
        facecolor='white',
        edgecolor=colors['agent_border'],
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(small_box)
    ax.text(4.5, y_pos, agent, ha='center', va='center', fontsize=9, color=colors['text'])

# LLM backbone annotation - FIXED: positioned inside the extended box
ax.text(4.5, 3.25 + main_y_offset, '+ LLM Backbone', ha='center', va='center',
        fontsize=8, color=colors['agent_border'], style='italic')
ax.text(4.5, 2.95 + main_y_offset, '(Sonnet 4.0, GPT-4o,', ha='center', va='center',
        fontsize=7, color=colors['text'])
ax.text(4.5, 2.75 + main_y_offset, 'Granite-3.0-8B)', ha='center', va='center',
        fontsize=7, color=colors['text'])

# =============================================================================
# Layer 3: MCP Servers (Center) - Aligned with Agentic Frameworks
# =============================================================================
# Main MCP container
mcp_container = FancyBboxPatch(
    (6.3, 2.3 + main_y_offset), 3.0, 4.2,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=colors['mcp'],
    edgecolor=colors['mcp_border'],
    linewidth=2,
    zorder=1
)
ax.add_patch(mcp_container)
ax.text(7.8, 6.2 + main_y_offset, 'MCP Servers', ha='center', va='center',
        fontsize=11, fontweight='bold', color=colors['text'])

# Prognostics Server
prog_box = FancyBboxPatch(
    (6.5, 4.5 + main_y_offset), 2.6, 1.6,
    boxstyle="round,pad=0.01,rounding_size=0.1",
    facecolor='white',
    edgecolor=colors['mcp_border'],
    linewidth=1.5,
    zorder=2
)
ax.add_patch(prog_box)
ax.text(7.8, 5.85 + main_y_offset, 'Prognostics Server', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['text'])
ax.text(7.8, 5.6 + main_y_offset, '(38 tools)', ha='center', va='center',
        fontsize=8, color=colors['mcp_border'])
prog_tools = ['load_dataset', 'train_rul_model', 'predict_rul', 'analyze_signal']
for i, tool in enumerate(prog_tools):
    ax.text(7.8, 5.3 + main_y_offset - i*0.2, f'• {tool}', ha='center', va='center',
            fontsize=7, color=colors['text'], family='monospace')

# Maintenance Server
maint_box = FancyBboxPatch(
    (6.5, 2.5 + main_y_offset), 2.6, 1.75,
    boxstyle="round,pad=0.01,rounding_size=0.1",
    facecolor='white',
    edgecolor=colors['mcp_border'],
    linewidth=1.5,
    zorder=2
)
ax.add_patch(maint_box)
ax.text(7.8, 4.0 + main_y_offset, 'Intelligent Maintenance', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['text'])
ax.text(7.8, 3.75 + main_y_offset, 'Server (27 tools)', ha='center', va='center',
        fontsize=8, color=colors['mcp_border'])
maint_tools = ['calc_maint_cost', 'assess_safety_risk', 'check_compliance', 'optimize_schedule']
for i, tool in enumerate(maint_tools):
    ax.text(7.8, 3.45 + main_y_offset - i*0.2, f'• {tool}', ha='center', va='center',
            fontsize=7, color=colors['text'], family='monospace')

# =============================================================================
# Layer 4: Evaluation Framework (Center-right)
# =============================================================================
eval_container = FancyBboxPatch(
    (9.8, 3.0 + main_y_offset), 2.0, 2.8,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=colors['eval'],
    edgecolor=colors['eval_border'],
    linewidth=2,
    zorder=1
)
ax.add_patch(eval_container)
ax.text(10.8, 5.55 + main_y_offset, 'Execution-Based', ha='center', va='center',
        fontsize=10, fontweight='bold', color=colors['text'])
ax.text(10.8, 5.25 + main_y_offset, 'Evaluation', ha='center', va='center',
        fontsize=10, fontweight='bold', color=colors['text'])

eval_items = ['Tool Validity', 'Schema Compliance', 'Dependency Order', 'Ground Truth Match']
for i, item in enumerate(eval_items):
    ax.text(10.8, 4.85 + main_y_offset - i*0.35, f'✓ {item}', ha='center', va='center',
            fontsize=8, color=colors['text'])

# =============================================================================
# Layer 5: Task-Commensurate Metrics (Right side)
# =============================================================================
metrics_container = FancyBboxPatch(
    (12.2, 2.8 + main_y_offset), 1.6, 3.7,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=colors['metrics'],
    edgecolor=colors['metrics_border'],
    linewidth=2,
    zorder=1
)
ax.add_patch(metrics_container)
ax.text(13.0, 6.2 + main_y_offset, 'Metrics', ha='center', va='center',
        fontsize=10, fontweight='bold', color=colors['text'])

metrics = [
    ('RUL', 'MAE/RMSE'),
    ('Fault', 'F1-Score'),
    ('Health', 'Categorical'),
    ('Cost', 'ROI Ratio'),
    ('Safety', 'Compliance')
]
for i, (task, metric) in enumerate(metrics):
    y_pos = 5.7 + main_y_offset - i*0.55
    ax.text(13.0, y_pos, task, ha='center', va='center',
            fontsize=8, fontweight='bold', color=colors['metrics_border'])
    ax.text(13.0, y_pos - 0.2, metric, ha='center', va='center',
            fontsize=7, color=colors['text'])

# =============================================================================
# Arrows connecting components - adjusted Y positions
# =============================================================================
arrow_y = 4.4 + main_y_offset

# Query -> Agent
draw_arrow(ax, (2.7, arrow_y), (3.2, arrow_y), colors['arrow'])

# Agent -> MCP
draw_arrow(ax, (5.8, arrow_y), (6.3, arrow_y), colors['arrow'])

# MCP -> Evaluation
draw_arrow(ax, (9.3, arrow_y), (9.8, arrow_y), colors['arrow'])

# Evaluation -> Metrics
draw_arrow(ax, (11.8, arrow_y), (12.2, arrow_y), colors['arrow'])

# =============================================================================
# Bottom: Industrial Assets and Scenarios - Moved up closer to main content
# =============================================================================
bottom_y = 1.7  # Moved up to be closer to the main flow

# Assets bar
assets_box = FancyBboxPatch(
    (0.3, bottom_y), 6.2, 1.2,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor='#ECEFF1',
    edgecolor='#607D8B',
    linewidth=1.5,
    zorder=1
)
ax.add_patch(assets_box)
ax.text(3.4, bottom_y + 0.95, '7 Industrial Asset Classes', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['text'])
assets = ['Turbofan', 'Bearings', 'Motors', 'Gearboxes', 'Aero-Eng.', '...']
for i, asset in enumerate(assets):
    ax.text(0.8 + i*1.0, bottom_y + 0.45, asset, ha='center', va='center',
            fontsize=7, color=colors['text'])

# Scenarios bar
scenarios_box = FancyBboxPatch(
    (6.8, bottom_y), 6.9, 1.2,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor='#ECEFF1',
    edgecolor='#607D8B',
    linewidth=1.5,
    zorder=1
)
ax.add_patch(scenarios_box)
ax.text(10.25, bottom_y + 0.95, '75 Expert-Curated Scenarios', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['text'])
scenarios = ['RUL (15)', 'Fault (15)', 'Health (30)', 'Cost (5)', 'Safety (10)']
for i, scenario in enumerate(scenarios):
    ax.text(7.4 + i*1.3, bottom_y + 0.45, scenario, ha='center', va='center',
            fontsize=7, color=colors['text'])

# Connecting lines from assets/scenarios to main flow
ax.plot([3.4, 3.4], [bottom_y + 1.2, 2.3 + main_y_offset], color='#607D8B', linewidth=1, linestyle='--', zorder=0)
ax.plot([10.25, 10.25], [bottom_y + 1.2, 2.8 + main_y_offset], color='#607D8B', linewidth=1, linestyle='--', zorder=0)

# =============================================================================
# Title
# =============================================================================
ax.text(7.0, 8.0, 'PHMForge: Benchmark Architecture Overview',
        ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'])

# Save the figure
plt.tight_layout()
plt.savefig('/Users/ayandas/Desktop/Development/AI_ML/research_ibm/Intent-Based-Industrial-Automation/ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/phmforge_architecture_diagram.pdf',
            format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('/Users/ayandas/Desktop/Development/AI_ML/research_ibm/Intent-Based-Industrial-Automation/ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/phmforge_architecture_diagram.png',
            format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)

print("Architecture diagram saved as:")
print("  - phmforge_architecture_diagram.pdf")
print("  - phmforge_architecture_diagram.png")

plt.show()
