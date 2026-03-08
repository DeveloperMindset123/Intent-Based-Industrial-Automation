import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up the figure with high DPI for publication quality
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

# Define the 5 stages with their data
stages = [
    {"count": 1, "label": "1\nScenario", "additions": "CMAPSS\nFD001", "timeline": "Week 1-2", "x": 0},
    {"count": 10, "label": "10\nScenarios", "additions": "+ FEMTO\n+ CWRU", "timeline": "Week 3-6", "x": 2.5},
    {"count": 20, "label": "20\nScenarios", "additions": "+ Cost-Benefit\n+ Safety", "timeline": "Week 7-12", "x": 5},
    {"count": 40, "label": "40\nScenarios", "additions": "+ 8 Bearing\nDatasets", "timeline": "Week 13-20", "x": 7.5},
    {"count": 75, "label": "75\nScenarios", "additions": "+ EngineMTQA\n+ Azure", "timeline": "Week 21-26", "x": 10}
]

# Colors for gradient effect (professional palette)
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

# Draw the horizontal funnel segments
for i, stage in enumerate(stages):
    # Calculate width based on scenario count (expanding)
    width_factor = stage['count'] / 75  # Normalize to max
    height = 0.3 + (width_factor * 2.2)  # Height ranges from 0.3 to 2.5

    # Draw trapezoid for each stage (expanding right)
    if i < len(stages) - 1:
        next_height = 0.3 + ((stages[i+1]['count'] / 75) * 2.2)
        x_start = stage['x']
        x_end = stages[i+1]['x']

        # Create trapezoid vertices (expanding funnel)
        vertices = [
            (x_start, -height/2),  # Bottom left
            (x_start, height/2),   # Top left
            (x_end, next_height/2),  # Top right
            (x_end, -next_height/2)  # Bottom right
        ]

        trapezoid = patches.Polygon(vertices, closed=True,
                                   facecolor=colors[i],
                                   edgecolor='black',
                                   linewidth=1.5,
                                   alpha=0.7)
        ax.add_patch(trapezoid)

    # Add scenario count label
    ax.text(stage['x'], 0, stage['label'],
           fontsize=11, fontweight='bold',
           ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

    # Add additions label (above)
    ax.text(stage['x'], height/2 + 0.4, stage['additions'],
           fontsize=8, ha='center', va='bottom',
           style='italic', color='#333333')

    # Add timeline label (below)
    ax.text(stage['x'], -height/2 - 0.4, stage['timeline'],
           fontsize=8, ha='center', va='top',
           fontweight='bold', color='#555555')

# Add arrows between stages
for i in range(len(stages) - 1):
    x_start = stages[i]['x'] + 0.3
    x_end = stages[i+1]['x'] - 0.3
    ax.annotate('', xy=(x_end, -1.8), xytext=(x_start, -1.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='#666666'))

# Add title
ax.text(5, 2.2, 'Progressive Scenario Expansion Strategy: From Proof-of-Concept to Comprehensive Benchmark',
       fontsize=12, fontweight='bold', ha='center', va='bottom')

# Add subtitle
ax.text(5, -2.2, 'Expert-Driven Curation Process Over 26 Weeks (396 Person-Hours)',
       fontsize=10, ha='center', va='top', style='italic', color='#555555')

# Set axis limits and remove axes
ax.set_xlim(-0.5, 11)
ax.set_ylim(-2.5, 2.5)
ax.axis('off')

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = '/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/reverse_funnel_horizontal.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Funnel diagram saved to: {output_path}")

# Also save as PDF for better quality in LaTeX
output_path_pdf = '/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/reverse_funnel_horizontal.pdf'
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
print(f"PDF version saved to: {output_path_pdf}")

plt.close()
