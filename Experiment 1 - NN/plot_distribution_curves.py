#!/usr/bin/env python3
"""
Plot the probability distributions for different Z values.
Shows probability vs noun index for Zipfian distributions and one-shot.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import distribution_interface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distribution_interface import create_zipfian_distribution, create_oneshot_distribution

def plot_distribution_curves():
    """
    Plot probability distributions for different Z values.
    """
    # Z values to plot
    z_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Vocabulary parameters (from the experiment setup)
    vocab_size = 40
    unseen_count = 10

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette - use a colormap for the Z values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(z_values)))

    # Plot each Z value distribution
    for i, z in enumerate(z_values):
        distribution = create_zipfian_distribution(z, vocab_size, unseen_count)
        noun_indices = np.arange(len(distribution))

        ax.plot(noun_indices, distribution,
                marker='o',
                markersize=4,
                linewidth=2,
                label=rf'$\alpha$={z:.1f}',
                color=colors[i],
                alpha=0.8)

    # Plot one-shot distribution
    oneshot_dist = create_oneshot_distribution(0, vocab_size, unseen_count)
    noun_indices = np.arange(len(oneshot_dist))

    ax.plot(noun_indices, oneshot_dist,
            marker='s',
            markersize=4,
            linewidth=2,
            label=r'$\alpha \to \infty$',
            color='red',
            linestyle='--',
            alpha=0.8)

    # Formatting
    ax.set_xlabel('Noun Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
    ax.set_title(r'Noun Probability Distributions Across $\alpha$ Values', fontsize=16, fontweight='bold')

    # Set x-axis to show all noun indices
    ax.set_xlim(-1, vocab_size)
    ax.set_xticks(np.arange(0, vocab_size, 5))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add vertical line to show where unseen nouns start
    ax.axvline(x=vocab_size - unseen_count - 0.5,
               color='gray',
               linestyle=':',
               linewidth=2,
               alpha=0.5,
               label='_nolegend_')

    # Add text annotation for unseen region
    ax.text(vocab_size - unseen_count/2 - 0.5, ax.get_ylim()[1] * 0.5,
            'Unseen\nNouns',
            ha='center',
            va='center',
            fontsize=10,
            color='gray',
            style='italic')

    # Legend (placed after text so it doesn't overlap)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

    plt.tight_layout()

    # Save figure
    output_file = '/Users/clairesmall/Desktop/imagining-syntax/noun_distribution_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Distribution curves saved to: {output_file}")

    # Also create a log-scale version for better visibility of differences
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Plot each Z value distribution with log scale
    for i, z in enumerate(z_values):
        distribution = create_zipfian_distribution(z, vocab_size, unseen_count)
        # Only plot non-zero probabilities for log scale
        non_zero_indices = [idx for idx, p in enumerate(distribution) if p > 0]
        non_zero_probs = [distribution[idx] for idx in non_zero_indices]

        ax2.plot(non_zero_indices, non_zero_probs,
                marker='o',
                markersize=4,
                linewidth=2,
                label=rf'$\alpha$={z:.1f}',
                color=colors[i],
                alpha=0.8)

    # Plot one-shot distribution
    oneshot_dist = create_oneshot_distribution(0, vocab_size, unseen_count)
    non_zero_indices = [idx for idx, p in enumerate(oneshot_dist) if p > 0]
    non_zero_probs = [oneshot_dist[idx] for idx in non_zero_indices]

    ax2.plot(non_zero_indices, non_zero_probs,
            marker='s',
            markersize=4,
            linewidth=2,
            label=r'$\alpha \to \infty$',
            color='red',
            linestyle='--',
            alpha=0.8)

    # Formatting
    ax2.set_xlabel('Noun Index', fontsize=26, fontweight='bold')
    ax2.set_ylabel('Probability (log scale)', fontsize=26, fontweight='bold')
    # Title removed for paper layout
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=22)

    # Set x-axis to show all noun indices
    ax2.set_xlim(-1, vocab_size)
    ax2.set_xticks(np.arange(0, vocab_size, 5))

    # Grid
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')

    # Add vertical line to show where unseen nouns start
    ax2.axvline(x=vocab_size - unseen_count - 0.5,
               color='gray',
               linestyle=':',
               linewidth=2,
               alpha=0.5)

    # Add text annotation (positioned lower in log space to avoid legend)
    y_limits = ax2.get_ylim()
    # In log space, use geometric mean of limits for middle position
    text_y = np.sqrt(y_limits[0] * y_limits[1]) * 0.05
    ax2.text(vocab_size - unseen_count/2 - 0.5, text_y,
            'Unseen\nNouns',
            ha='center',
            va='center',
            fontsize=24,
            color='gray',
            style='italic')

    # Legend (placed after text so it doesn't overlap)
    ax2.legend(loc='upper right', fontsize=24, framealpha=0.95)

    plt.tight_layout()

    # Save log scale figure
    output_file_log = '/Users/clairesmall/Desktop/imagining-syntax/noun_distribution_curves_log.png'
    plt.savefig(output_file_log, dpi=300, bbox_inches='tight')
    print(f"Log-scale distribution curves saved to: {output_file_log}")

    # Also save to Plots for Paper
    paper_output = '/Users/clairesmall/Desktop/imagining-syntax/Plots for Paper/noun_distribution_curves_log_alpha.png'
    paper_output_pdf = '/Users/clairesmall/Desktop/imagining-syntax/Plots for Paper/noun_distribution_curves_log_alpha.pdf'
    plt.savefig(paper_output, dpi=300, bbox_inches='tight')
    plt.savefig(paper_output_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved to Plots for Paper: {paper_output}")

    plt.close('all')

    # Print summary statistics
    print("\n=== Distribution Statistics ===")
    for z in z_values:
        dist = create_zipfian_distribution(z, vocab_size, unseen_count)
        max_prob = max(dist)
        min_nonzero_prob = min([p for p in dist if p > 0])
        print(f"Z={z:.1f}: Max prob = {max_prob:.4f}, Min non-zero prob = {min_nonzero_prob:.6f}")

    oneshot_dist = create_oneshot_distribution(0, vocab_size, unseen_count)
    max_prob = max(oneshot_dist)
    min_nonzero_prob = min([p for p in oneshot_dist if p > 0])
    print(f"One-shot: Max prob = {max_prob:.4f}, Min non-zero prob = {min_nonzero_prob:.6f}")

if __name__ == '__main__':
    plot_distribution_curves()
