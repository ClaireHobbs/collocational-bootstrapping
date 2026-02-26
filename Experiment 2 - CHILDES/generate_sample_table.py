"""
Generate a publication-quality table of sample utterances from each age group.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import textwrap

# Age groups
AGE_GROUPS = [
    (0, 12, '0-12'),
    (12, 24, '12-24'),
    (24, 36, '24-36'),
    (36, 48, '36-48'),
    (48, 60, '48-60'),
    (60, 100, '60-100')
]

def get_good_samples(df, age_min, age_max, n=5, seed=42):
    """Get clean, representative sample utterances."""
    subset = df[(df['target_child_age'] >= age_min) & (df['target_child_age'] < age_max)]

    # Common English words to check for (at least one should be present)
    common_words = r'\b(the|a|an|is|are|was|were|you|I|he|she|it|we|they|to|do|did|can|will|have|has|had|what|where|when|why|how|this|that|and|or|but|in|on|at|for|with|my|your|his|her|our|their)\b'

    # Filter for better quality utterances
    subset = subset[
        (subset['full_utterance'].str.len() >= 20) &  # At least 20 chars
        (subset['full_utterance'].str.len() <= 50) &  # Not too long
        (~subset['full_utterance'].str.contains('xxx', case=False, na=False)) &  # No unintelligible
        (~subset['full_utterance'].str.contains('_', na=False)) &  # No special markers
        (~subset['full_utterance'].str.contains('participant', case=False, na=False)) &  # No meta
        (subset['full_utterance'].str.match(r'^[a-zA-Z]', na=False)) &  # Starts with letter
        (subset['full_utterance'].str.match(r"^[a-zA-Z\s',\.!\?]+$", na=False)) &  # English chars only
        (subset['full_utterance'].str.contains(common_words, case=False, na=False, regex=True))  # Has common English words
    ]

    if len(subset) >= n:
        return subset.sample(n=n, random_state=seed)['full_utterance'].tolist()
    else:
        return subset['full_utterance'].tolist()[:n]

def create_table():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/childes_utterances.csv', low_memory=False)
    df = df.dropna(subset=['full_utterance', 'target_child_age'])
    df['full_utterance'] = df['full_utterance'].astype(str)

    # Collect samples
    print("Collecting samples...")
    all_samples = []
    for age_min, age_max, label in AGE_GROUPS:
        samples = get_good_samples(df, age_min, age_max, n=5, seed=123)
        for i, utterance in enumerate(samples):
            all_samples.append({
                'Age Group (months)': label,
                'Sample': i + 1,
                'Utterance': utterance
            })

    samples_df = pd.DataFrame(all_samples)

    # Save as CSV
    samples_df.to_csv('output/age_groups_complete/sample_utterances.csv', index=False)
    print("Saved to output/age_groups_complete/sample_utterances.csv")

    # Create publication-quality figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    ax.set_title('Sample Child-Directed Utterances by Age Group',
                 fontsize=18, fontweight='bold', pad=20, color='#2C3E50')

    # Prepare table data
    table_data = []
    for age_min, age_max, label in AGE_GROUPS:
        samples = get_good_samples(df, age_min, age_max, n=5, seed=123)
        for i, utterance in enumerate(samples):
            if i == 0:
                table_data.append([label, f'"{utterance}"'])
            else:
                table_data.append(['', f'"{utterance}"'])
        # Add empty row between age groups (except last)
        if label != '60-100':
            table_data.append(['', ''])

    # Create table
    col_labels = ['Age (months)', 'Utterance']
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
        colWidths=[0.15, 0.75]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(2):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#2C3E50')

    # Style data cells
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            cell.set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
            if j == 0 and table_data[i-1][0]:  # Age group cells
                cell.set_text_props(fontweight='bold', color='#3498DB')

    plt.tight_layout()
    plt.savefig('output/age_groups_complete/sample_utterances_table.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved to output/age_groups_complete/sample_utterances_table.png")
    plt.close()

    # Also create a cleaner text-based table
    print("\n" + "="*70)
    print("SAMPLE UTTERANCES BY AGE GROUP")
    print("="*70)

    for age_min, age_max, label in AGE_GROUPS:
        samples = get_good_samples(df, age_min, age_max, n=5, seed=123)
        print(f"\n{label} months:")
        print("-" * 50)
        for i, utterance in enumerate(samples, 1):
            print(f"  {i}. \"{utterance}\"")

    print("\n" + "="*70)

if __name__ == '__main__':
    create_table()
