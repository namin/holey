import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Haiku', 'Sonnet', 'Claude']
with_hint = [28, 23, 35]
without_hint = [31, 38, 40]
total_solved = [34, 39, 41]
total_puzzles = 47

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Set position of bars on x-axis
x = np.arange(len(models))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, with_hint, width, label='With Extrapolation Hint', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, without_hint, width, label='Without Hint', color='#2ecc71', alpha=0.8)

# Add total solved as a dotted line
ax.plot(x, total_solved, 'o--', color='#e74c3c', linewidth=2, markersize=8, label='Total Solved', zorder=5)

# Add a horizontal line for total puzzles
ax.axhline(y=total_puzzles, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'Total Puzzles ({total_puzzles})')

# Customize the plot
ax.set_ylabel('Number of Puzzles Solved', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Puzzle Solving Performance Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 50)

# Add value labels on bars with percentages
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_puzzles) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=8)

add_value_labels(bars1)
add_value_labels(bars2)

# Add value labels for the line plot
for i, (xi, yi) in enumerate(zip(x, total_solved)):
    percentage = (yi / total_puzzles) * 100
    ax.text(xi, yi + 1.5, f'{yi}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e74c3c')

plt.tight_layout()
fn = 'results_graph_claude_extrapolation.png'
plt.savefig(fn, dpi=300, bbox_inches='tight')
print(f"Graph saved as {fn}")
plt.show()
