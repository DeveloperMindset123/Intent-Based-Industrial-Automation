import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with more space
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Data from my_scenarios.json
task_categories = {
    'Engine Health Analysis': 30,
    'Fault Classification': 15,
    'RUL Prediction': 15,
    'Safety/Policy Evaluation': 10,
    'Cost-Benefit Analysis': 5
}

# PHMBench categories mapping (inner ring)
phmbench_categories = {
    'Condition Monitoring': 30,  # Maps to Engine Health Analysis
    'Fault Diagnosis': 15,       # Maps to Fault Classification
    'Fault & RUL Detection': 15, # Maps to RUL Prediction
    'Maintenance Scheme': 15     # Maps to Cost-Benefit (5) + Safety (10)
}

# Colors for outer ring (your categories) - vibrant, professional palette
outer_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# Colors for inner ring (PHMBench categories) - matching darker shades
inner_colors = ['#c0392b', '#2980b9', '#27ae60', '#d68910']

# Outer ring data (your 5 categories)
outer_labels = list(task_categories.keys())
outer_sizes = list(task_categories.values())

# Inner ring data (PHMBench 4 categories)
inner_labels = list(phmbench_categories.keys())
inner_sizes = list(phmbench_categories.values())

# Create the donut chart with no labels initially
# Outer ring (your categories)
wedges_outer, texts_outer = ax.pie(
    outer_sizes,
    colors=outer_colors,
    startangle=90,
    radius=1.3,
    wedgeprops=dict(width=0.35, edgecolor='white', linewidth=3),
    labels=None
)

# Inner ring (PHMBench categories)
wedges_inner, texts_inner = ax.pie(
    inner_sizes,
    colors=inner_colors,
    startangle=90,
    radius=0.95,
    wedgeprops=dict(width=0.35, edgecolor='white', linewidth=3),
    labels=None
)

# Manually add labels for outer ring with better positioning
for i, (wedge, label, size) in enumerate(zip(wedges_outer, outer_labels, outer_sizes)):
    angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    x = 1.55 * np.cos(np.radians(angle))
    y = 1.55 * np.sin(np.radians(angle))

    # Horizontal alignment based on position
    ha = 'left' if x > 0 else 'right'

    # Add label with count
    percentage = (size / sum(outer_sizes)) * 100
    ax.text(x, y, f'{label}\n{size} scenarios ({percentage:.1f}%)',
            ha=ha, va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=outer_colors[i],
                     edgecolor='white', linewidth=2, alpha=0.9),
            color='white')

# Manually add labels for inner ring
for i, (wedge, label, size) in enumerate(zip(wedges_inner, inner_labels, inner_sizes)):
    angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    x = 0.55 * np.cos(np.radians(angle))
    y = 0.55 * np.sin(np.radians(angle))

    # Add label
    percentage = (size / sum(inner_sizes)) * 100
    ax.text(x, y, f'{label}\n({percentage:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=inner_colors[i],
                     edgecolor='white', linewidth=1.5, alpha=0.95))

# Add title
plt.title('Scenario Categorization: Task Distribution and PHMBench Mapping\n75 Expert-Vetted Scenarios Across 4 Core PHM Categories',
         fontsize=15, fontweight='bold', pad=30)

# Add center text
ax.text(0, 0, '75\nScenarios', ha='center', va='center',
       fontsize=20, fontweight='bold', color='#34495e',
       bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                edgecolor='#34495e', linewidth=2))

# Add legend with mapping information
legend_elements = [
    'PHMBench Mapping:',
    '',
    '• Inner Ring: PHMBench Core Categories (4)',
    '• Outer Ring: Our Task Categories (5)',
    '',
    'Category Alignment:',
    '• Condition Monitoring → Engine Health Analysis',
    '• Fault Diagnosis → Fault Classification',
    '• Fault & RUL Detection → RUL Prediction',
    '• Maintenance Scheme → Cost-Benefit + Safety'
]

plt.text(-2.3, -2.0, '\n'.join(legend_elements),
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1',
                 edgecolor='#34495e', linewidth=2, alpha=0.9))

# Equal aspect ratio ensures circular chart
ax.axis('equal')

# Remove axes
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = '/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/donut_chart_categorization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Donut chart saved to: {output_path}")

# Also save as PDF
output_path_pdf = '/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/ReActXen/src/reactxen/demo/intent_implementation_demo/multi_agent_implementation_demo/donut_chart_categorization.pdf'
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
print(f"PDF version saved to: {output_path_pdf}")

plt.close()
