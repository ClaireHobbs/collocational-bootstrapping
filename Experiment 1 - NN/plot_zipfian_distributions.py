#!/usr/bin/env python3
"""
Visualize Zipfian distributions at different Z values.
Shows how the distribution shape changes with the Zipfian parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from distribution_interface import create_zipfian_distribution

# Parameters
vocab_size = 40
unseen_count = 10
seen_vocab_size = vocab_size - unseen_count

# Z values to visualize (0 to 3 in steps of 0.2)
z_values = np.arange(0.0, 3.2, 0.2)

# Create figure with 4 rows and 4 columns
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

# Plot each distribution
for idx, Z in enumerate(z_values):
    ax = axes[idx]

    # Generate distribution
    dist = create_zipfian_distribution(Z, vocab_size=vocab_size, unseen_count=unseen_count)

    # Only plot seen pairs (non-zero probabilities)
    seen_dist = dist[:seen_vocab_size]
    ranks = np.arange(1, seen_vocab_size + 1)

    # Bar plot
    ax.bar(ranks, seen_dist, color='steelblue', alpha=0.7)
    ax.set_xlabel('Rank (Noun-Verb Offset)', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'Z = {Z:.1f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in seen_dist])
    max_entropy = np.log2(seen_vocab_size)
    ax.text(0.95, 0.95, f'H = {entropy:.2f} bits\n({entropy/max_entropy*100:.1f}% of max)',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Overall title
fig.suptitle(r'Zipfian Distributions at Different $\alpha$ Values (0 to 3, step 0.2)' + '\n(vocab_size=40, seen pairs=30)',
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('zipfian_distributions_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'zipfian_distributions_comparison.png'")
print(f"Showing {len(z_values)} distributions: Z = {list(z_values)}")
plt.show()
