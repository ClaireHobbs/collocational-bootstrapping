"""
Regenerate the age group plot using existing results (no reanalysis needed).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_z_by_age():
    """Create publication-ready plot showing z values across age groups."""
    # Load existing results
    df = pd.read_csv('output/age_groups_complete/age_group_summary_complete.csv')
    print("Loaded results:")
    print(df[['age_group', 'z', 'n_utterances']].to_string(index=False))

    # Set up publication-ready styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(df))
    z_values = df['z'].values
    age_labels = df['age_group'].values

    # Plot line connecting points
    ax.plot(x, z_values, color='#2C3E50', linewidth=2.5, zorder=1)

    # Plot large, readable points
    ax.scatter(x, z_values, s=200, color='#3498DB', edgecolor='#2C3E50',
               linewidth=2, zorder=2)

    # Add value labels above each point
    for i, (xi, z) in enumerate(zip(x, z_values)):
        ax.annotate(f'{z:.2f}',
                    xy=(xi, z),
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=13, fontweight='bold',
                    color='#2C3E50')

    # Title and subtitle with more space between them
    ax.set_title('Zipf Parameter Decreases with Child Age',
                 fontsize=20, fontweight='bold', pad=35, color='#2C3E50')

    # Subtitle showing trend (moved down with more space from title)
    z_start = z_values[0]
    z_end = z_values[-1]
    ax.text(0.5, 1.03,
            f'z = {z_start:.2f} (0-12mo) → z = {z_end:.2f} (60-100mo)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=14, style='italic', color='#7F8C8D')

    # Axis labels
    ax.set_xlabel('Child Age Group (Months)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Zipf Parameter (z)', fontsize=16, fontweight='bold', labelpad=10)

    # X-axis ticks - just numbers without "mo"
    ax.set_xticks(x)
    x_labels_short = [label.replace('mo', '') for label in age_labels]
    ax.set_xticklabels(x_labels_short, fontsize=13)

    # Y-axis range with some padding
    y_min = min(z_values) - 0.1
    y_max = max(z_values) + 0.15
    ax.set_ylim(y_min, y_max)

    # Y-axis ticks
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis='y', labelsize=12)

    # Add reference line for overall z value
    ax.axhline(y=1.42, color='#3498DB', linestyle=':', linewidth=2, alpha=0.7, zorder=0)
    ax.text(len(x) - 1, 1.42 + 0.015, 'Overall z = 1.42', fontsize=13,
            ha='right', va='bottom', color='#3498DB', style='italic')

    # Add subtle grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['left'].set_color('#2C3E50')
    ax.spines['bottom'].set_color('#2C3E50')

    plt.tight_layout()

    output_path = 'output/age_groups_complete/zipf_by_age_complete.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nPlot saved to {output_path}")
    plt.close()

    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == '__main__':
    plot_z_by_age()
