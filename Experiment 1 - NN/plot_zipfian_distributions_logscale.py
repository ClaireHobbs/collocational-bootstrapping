#!/usr/bin/env python3
"""
Visualize Zipfian distributions on log-log scale to verify power-law behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from distribution_interface import create_distribution

def plot_zipfian_distributions_logscale():
    """Plot Zipfian distributions for different Z values on log-log scale."""

    # Parameters
    z_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    vocab_size = 40
    unseen_count = 10
    seen_size = vocab_size - unseen_count  # 30

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color map
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(z_values)))

    # Plot 1: Regular scale
    for i, z in enumerate(z_values):
        dist = create_distribution('zipfian', z, vocab_size, unseen_count)
        # Only plot the seen pairs (first 30 items)
        seen_dist = dist[:seen_size]
        ranks = np.arange(1, seen_size + 1)

        ax1.plot(ranks, seen_dist,
                marker='o',
                markersize=4,
                linewidth=2,
                color=colors[i],
                label=rf'$\alpha$={z:.1f}')

    ax1.set_xlabel('Rank (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability P(k)', fontsize=12, fontweight='bold')
    ax1.set_title('Zipfian Distributions\nRegular Scale', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, seen_size + 1)

    # Plot 2: Log-log scale
    for i, z in enumerate(z_values):
        dist = create_distribution('zipfian', z, vocab_size, unseen_count)
        # Only plot the seen pairs (first 30 items)
        seen_dist = dist[:seen_size]
        ranks = np.arange(1, seen_size + 1)

        ax2.loglog(ranks, seen_dist,
                  marker='o',
                  markersize=4,
                  linewidth=2,
                  color=colors[i],
                  label=rf'$\alpha$={z:.1f}')

    ax2.set_xlabel('Rank (k) [log scale]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability P(k) [log scale]', fontsize=12, fontweight='bold')
    ax2.set_title('Zipfian Distributions\nLog-Log Scale (Linear = Power Law)',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')

    # Add annotation explaining linearity
    ax2.text(0.98, 0.98,
            r'Linear lines confirm' + '\n' + r'power-law behavior:' + '\n' + r'P(k) ∝ 1/k$^{\alpha}$',
            transform=ax2.transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='gray', alpha=0.9))

    plt.suptitle(r'Zipfian Distribution Analysis: $\alpha$=1.0 to $\alpha$=2.0 (step=0.2)' + '\n' +
                'vocab=40, unseen=10 (30 seen pairs shown)',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    output_path = 'zipfian_distributions_logscale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    # Print distribution statistics
    print("\n" + "="*80)
    print("DISTRIBUTION STATISTICS")
    print("="*80)

    for z in z_values:
        dist = create_distribution('zipfian', z, vocab_size, unseen_count)
        seen_dist = dist[:seen_size]

        print(f"\nZ = {z:.1f}:")
        print(f"  Top 5 probabilities: {[f'{p:.4f}' for p in seen_dist[:5]]}")
        print(f"  Bottom 5 probabilities: {[f'{p:.4f}' for p in seen_dist[-5:]]}")
        print(f"  Ratio (1st/30th): {seen_dist[0]/seen_dist[-1]:.2f}x")
        print(f"  Sum of probabilities: {sum(dist):.6f}")

        # Calculate concentration (% of probability in top 10%)
        top_10pct_count = max(1, seen_size // 10)
        concentration = sum(seen_dist[:top_10pct_count]) / sum(seen_dist)
        print(f"  Concentration (top {top_10pct_count} items): {concentration*100:.1f}%")

if __name__ == "__main__":
    plot_zipfian_distributions_logscale()
