import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


epochs = list(range(1, 11))

train_losses = [2.7437, 2.1831, 1.8895, 1.6549, 1.4100, 1.1478, 0.8927, 0.6599, 0.4813, 0.3582]
val_losses = [2.4379, 2.1556, 1.9533, 1.9505, 1.8292, 1.9599, 2.1616, 2.3937, 2.5083, 2.8860]

train_accs = [0.3397, 0.4394, 0.4994, 0.5488, 0.6038, 0.6692, 0.7300, 0.7972, 0.8474, 0.8862]
val_accs = [0.3823, 0.4508, 0.4921, 0.4910, 0.5214, 0.5217, 0.4903, 0.4925, 0.5142, 0.5029]

train_accs_pct = [acc * 100 for acc in train_accs]
val_accs_pct = [acc * 100 for acc in val_accs]


data = {
    'Country': ['United States', 'Japan', 'France', 'UK', 'Brazil', 'Australia', 
                'Russia', 'Canada', 'Spain', 'Argentina', 'Autres'],
    'Support': [2345, 802, 687, 509, 470, 345, 357, 302, 229, 136, 3818],
    'F1-Score': [0.73, 0.73, 0.44, 0.53, 0.53, 0.61, 0.45, 0.32, 0.23, 0.44, 0.15]
}

df = pd.DataFrame(data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.barh(df['Country'], df['Support'], color='steelblue')
ax1.set_xlabel('Number of Test Examples', fontsize=12)
ax1.set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

colors = ['green' if f1 > 0.5 else 'orange' if f1 > 0.3 else 'red' for f1 in df['F1-Score']]
ax2.barh(df['Country'], df['F1-Score'], color=colors)
ax2.set_xlabel('F1-Score', fontsize=12)
ax2.set_xlim(0, 1)
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (>0.5)')
ax2.set_title('Model Performance by Country', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('class_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))

metrics_data = [
    ('Train Acc', 88.6, 'Train performance'),
    ('Val Acc (best)', 52.2, 'Optimal point'),
    ('Macro F1', 19.0, 'Real performance'),
]

y_pos = range(len(metrics_data))
names = [m[0] for m in metrics_data]
values = [m[1] for m in metrics_data]
colors = ['lightgreen', 'orange', 'red']

bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=2)

# Ajouter les valeurs
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=12)
    
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel('Percentage (%)', fontsize=12)
ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.axvline(x=50, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target')
ax.grid(axis='x', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('summary_improved.png', dpi=300, bbox_inches='tight')