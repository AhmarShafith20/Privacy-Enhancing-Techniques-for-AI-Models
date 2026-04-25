import matplotlib.pyplot as plt
import numpy as np

models = ['Baseline', 'Federated\nLearning', 'FL + DP\n(ε=1.0)']
accuracy = [98.77, 98.78, 87.75]
times = [87.41, 106.40, 362.02]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#2196F3', '#4CAF50', '#FF5722']

ax1.bar(models, accuracy, color=colors, width=0.5)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(80, 100)
for i, v in enumerate(accuracy):
    ax1.text(i, v + 0.1, f'{v}%', ha='center', fontweight='bold')

ax2.bar(models, times, color=colors, width=0.5)
ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Time (seconds)')
for i, v in enumerate(times):
    ax2.text(i, v + 5, f'{v}s', ha='center', fontweight='bold')

plt.suptitle('Privacy-Preserving AI: Accuracy vs Training Time', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved.")