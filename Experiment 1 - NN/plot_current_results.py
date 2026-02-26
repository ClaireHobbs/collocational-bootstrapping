#!/usr/bin/env python3
"""
Visualize Model Accuracy vs Z Parameter matching the existing style.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime

# Read the CSV file
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = 'comprehensive_experiments/Z_0.0-3.0_step0.1_n10_20251026_215846/summary/comprehensive_results.csv'

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Could not find {csv_path}")
    sys.exit(1)

# Get unique Z values
z_values = sorted(df['c_value'].unique())
n_z_values = len(z_values)
n_iterations = 10  # from experiment parameters

# Color and marker scheme matching the reference chart
style_config = {
    'seen_match': {'color': '#3182bd', 'marker': 'o', 'label': 'Seen, Match'},
    'seen_mismatch': {'color': '#feb24c', 'marker': 's', 'label': 'Seen, Mismatch'},
    'unseen_match': {'color': '#de2d26', 'marker': '^', 'label': 'Unseen, Match'},
    'unseen_mismatch': {'color': '#756bb1', 'marker': 'v', 'label': 'Unseen, Mismatch'}
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each eval type
for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
    if eval_type not in df['eval_type'].values:
        continue

    data = df[df['eval_type'] == eval_type].sort_values('c_value')
    style = style_config[eval_type]

    # Shaded region for min/max range
    ax.fill_between(data['c_value'], data['min'], data['max'],
                     alpha=0.2, color=style['color'])

    # Solid line with markers and error bars (clipped at 0 and 100)
    # Calculate asymmetric error bars to prevent going above 100% or below 0%
    lower_error = np.minimum(data['std'], data['mean'])
    upper_error = np.minimum(data['std'], 100 - data['mean'])

    ax.errorbar(data['c_value'], data['mean'],
                yerr=[lower_error, upper_error],
                color=style['color'], marker=style['marker'],
                markersize=10, linewidth=2.5, capsize=5, capthick=2,
                label=style['label'], linestyle='-', zorder=3)

# Formatting
ax.set_xlabel(r'$\alpha$ Value', fontsize=14)
ax.set_ylabel('Mean Accuracy (%)', fontsize=14)
ax.set_title(f'Model Accuracy vs Zipfian Parameter $\\alpha$\nn={n_iterations}, step={z_values[1]-z_values[0]:.1f}, vocab=40, unseen=10, Run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             fontsize=14)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.set_ylim([0, 105])

# Set x-axis to show all Z values tested
ax.set_xlim([z_values[0] - 0.1, z_values[-1] + 0.1])

plt.tight_layout()
plt.savefig('accuracy_vs_Z.png', dpi=300, bbox_inches='tight')
print(f"\nChart saved as 'accuracy_vs_Z.png'")
print(f"Data includes {n_z_values} Z values: {z_values}")
print(f"\nSummary by eval type:")
for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
    if eval_type in df['eval_type'].values:
        data = df[df['eval_type'] == eval_type]
        print(f"  {eval_type:20s}: mean={data['mean'].mean():.1f}%, std={data['std'].mean():.1f}, range=[{data['min'].min():.1f}, {data['max'].max():.1f}]")
plt.show()
