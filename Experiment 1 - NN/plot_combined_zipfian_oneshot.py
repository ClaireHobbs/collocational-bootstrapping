#!/usr/bin/env python3
"""
Plot combined results from Zipfian and One-Shot experiments.
This script creates a visualization comparing accuracy across:
- Zipfian distribution (Z=0 to Z=3)
- One-shot distribution (deterministic pairing)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_zipfian_results(zipfian_dir):
    """Load results from Zipfian experiment."""
    csv_path = Path(zipfian_dir) / "summary" / "comprehensive_results.csv"
    df = pd.read_csv(csv_path)
    return df

def load_oneshot_results(oneshot_dir):
    """Load results from One-Shot experiment."""
    csv_path = Path(oneshot_dir) / "summary" / "comprehensive_results.csv"
    df = pd.read_csv(csv_path)
    return df

def plot_combined_results(zipfian_df, oneshot_df, output_path):
    """Create combined visualization."""

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for each evaluation type
    colors = {
        'seen_match': '#3182bd',      # Blue
        'seen_mismatch': '#feb24c',    # Orange
        'unseen_match': '#de2d26',     # Red
        'unseen_mismatch': '#756bb1'   # Purple
    }

    labels = {
        'seen_match': 'Seen, Match',
        'seen_mismatch': 'Seen, Mismatch',
        'unseen_match': 'Unseen, Match',
        'unseen_mismatch': 'Unseen, Mismatch'
    }

    # Plot Zipfian results (Z = 0 to 3.0)
    for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
        subset = zipfian_df[zipfian_df['eval_type'] == eval_type]

        # Plot mean line
        ax.plot(subset['c_value'], subset['mean'],
                color=colors[eval_type],
                linewidth=2,
                label=labels[eval_type],
                marker='o',
                markersize=4)

        # Plot confidence interval (capped at 100%)
        upper_bound = np.minimum(subset['mean'] + subset['std'], 100)
        ax.fill_between(subset['c_value'],
                        subset['mean'] - subset['std'],
                        upper_bound,
                        color=colors[eval_type],
                        alpha=0.2)

    # Add oneshot result as a vertical line/marker at a special position
    # We'll place it at Z = 3.2 (to the right of Z=3.0) since one-shot is the limiting case as Z→∞
    oneshot_x = 3.2

    for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
        subset = oneshot_df[oneshot_df['eval_type'] == eval_type]
        mean_val = subset['mean'].values[0]
        std_val = subset['std'].values[0]

        # Calculate asymmetric error bars to cap at 100%
        lower_error = std_val
        upper_error = min(std_val, 100 - mean_val)  # Don't exceed 100%

        # Plot oneshot as a large marker with error bars
        ax.errorbar(oneshot_x, mean_val,
                   yerr=[[lower_error], [upper_error]],  # Asymmetric errors
                   marker='D',  # Diamond marker
                   markersize=10,
                   color=colors[eval_type],
                   capsize=5,
                   capthick=2,
                   linewidth=2,
                   linestyle='none')

    # Add vertical line to separate oneshot from Zipfian
    ax.axvline(x=3.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel('Distribution Parameter', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Accuracy: One-Shot vs Zipfian Distribution\n' +
                'Deterministic Pairing vs Varying Concentration\n' +
                'n=10, step=0.1, vocab=40, unseen=10',
                fontsize=16, fontweight='bold', pad=20)

    # Set x-axis limits and ticks
    ax.set_xlim(-0.1, 3.4)

    # Custom x-ticks
    xticks = list(np.arange(0, 3.1, 0.5)) + [oneshot_x]
    xticklabels = [rf'$\alpha$={z:.1f}' for z in np.arange(0, 3.1, 0.5)] + [r'One-Shot' + '\n' + r'($\alpha$→∞)']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')

    # Set y-axis
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(loc='lower left',
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      fontsize=11)
    legend.get_frame().set_alpha(0.9)

    # Add text annotation explaining oneshot
    ax.text(oneshot_x, 50, 'One-Shot:\nDeterministic\n1:1 pairing\n(Z→∞ limit)',
            ha='center', va='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {output_path}")

    return fig

def main():
    # Paths to experiment directories
    zipfian_dir = "comprehensive_experiments/Z_0.0-3.0_step0.1_n10_20251026_215846"
    oneshot_dir = "comprehensive_experiments/O_0.0-0.0_step0.1_n10_20251203_173458"
    output_path = "combined_oneshot_zipfian_comparison.png"

    # Check if directories exist
    if not Path(zipfian_dir).exists():
        print(f"Error: Zipfian directory not found: {zipfian_dir}")
        sys.exit(1)

    if not Path(oneshot_dir).exists():
        print(f"Error: One-shot directory not found: {oneshot_dir}")
        sys.exit(1)

    # Load data
    print("Loading Zipfian results...")
    zipfian_df = load_zipfian_results(zipfian_dir)

    print("Loading One-Shot results...")
    oneshot_df = load_oneshot_results(oneshot_dir)

    # Create plot
    print("Creating combined visualization...")
    plot_combined_results(zipfian_df, oneshot_df, output_path)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nOne-Shot Distribution (Deterministic 1:1 pairing):")
    for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
        subset = oneshot_df[oneshot_df['eval_type'] == eval_type]
        print(f"  {eval_type:20s}: {subset['mean'].values[0]:6.2f}% ± {subset['std'].values[0]:5.2f}%")

    print("\nZipfian Z=0 (Uniform - most similar to varied training):")
    z0_data = zipfian_df[zipfian_df['c_value'] == 0.0]
    for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
        subset = z0_data[z0_data['eval_type'] == eval_type]
        print(f"  {eval_type:20s}: {subset['mean'].values[0]:6.2f}% ± {subset['std'].values[0]:5.2f}%")

    print("\nZipfian Z=3 (Highly concentrated - most similar to deterministic):")
    z3_data = zipfian_df[zipfian_df['c_value'] == 3.0]
    for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
        subset = z3_data[z3_data['eval_type'] == eval_type]
        print(f"  {eval_type:20s}: {subset['mean'].values[0]:6.2f}% ± {subset['std'].values[0]:5.2f}%")

if __name__ == "__main__":
    main()
