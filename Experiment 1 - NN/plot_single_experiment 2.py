#!/usr/bin/env python3
"""
Simple plotting script for single experiment results.
Creates a bar chart showing accuracy across different evaluation types.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_single_experiment_results(experiment_dir, output_path=None):
    """Plot results from a single C experiment as a bar chart"""

    experiment_path = Path(experiment_dir)
    summary_file = experiment_path / 'summary.txt'

    if not summary_file.exists():
        print(f"Error: summary.txt not found in {experiment_dir}")
        return

    # Parse summary file for results
    results = {}
    c_value = None

    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if 'C=' in line and 'Experiment Summary' in line:
                # Extract C value from "Experiment Summary for C=0.5"
                c_value = line.split('C=')[1].split()[0]
            elif ': Model Accuracy:' in line:
                # Parse lines like "seen_match: Model Accuracy: 99.80% (998/1000)"
                parts = line.split(': Model Accuracy: ')
                eval_type = parts[0]
                accuracy_str = parts[1].split('%')[0]
                accuracy = float(accuracy_str)
                results[eval_type] = accuracy

    if not results:
        print("Error: No results found in summary.txt")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and labels for evaluation types
    eval_types = ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    labels = ['Seen, Match', 'Seen, Mismatch', 'Unseen, Match', 'Unseen, Mismatch']

    # Filter to only include evaluation types we have data for
    plot_types = []
    plot_colors = []
    plot_labels = []
    accuracies = []

    for i, eval_type in enumerate(eval_types):
        if eval_type in results:
            plot_types.append(eval_type)
            plot_colors.append(colors[i])
            plot_labels.append(labels[i])
            accuracies.append(results[eval_type])

    # Create bar chart
    bars = ax.bar(range(len(plot_types)), accuracies, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.8)

    # Customize the plot
    ax.set_xlabel('Evaluation Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Model Accuracy by Evaluation Type (C = {c_value})', fontsize=14)

    # Set x-axis labels
    ax.set_xticks(range(len(plot_types)))
    ax.set_xticklabels(plot_labels, rotation=45, ha='right')

    # Set y-axis limits
    ax.set_ylim(0, 105)

    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add grid for readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    # Print summary statistics
    print(f"\nResults Summary for C = {c_value}:")
    print("=" * 40)
    for eval_type, label, acc in zip(plot_types, plot_labels, accuracies):
        print(f"{label:20}: {acc:5.1f}%")

    if len(accuracies) > 1:
        print(f"\nMean accuracy: {np.mean(accuracies):.1f}%")
        print(f"Std deviation: {np.std(accuracies):.1f}%")
        print(f"Min accuracy:  {np.min(accuracies):.1f}%")
        print(f"Max accuracy:  {np.max(accuracies):.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Plot results from a single experiment')
    parser.add_argument('experiment_dir', type=str,
                        help='Path to experiment directory (e.g., experiments/C_50_20250914_145306)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: saves to experiment_dir/accuracy_plot.png)')

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        experiment_path = Path(args.experiment_dir)
        output_path = experiment_path / 'accuracy_plot.png'

    print(f"Plotting results from: {args.experiment_dir}")
    plot_single_experiment_results(args.experiment_dir, str(output_path))

if __name__ == "__main__":
    main()