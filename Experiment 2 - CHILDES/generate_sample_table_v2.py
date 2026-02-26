"""
Generate a publication-quality table of sample utterances from each age group.
Version 2 - Different samples, modern professional styling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    # Common English words to check for
    common_words = r'\b(?:the|a|an|is|are|was|were|you|I|he|she|it|we|they|to|do|did|can|will|have|has|had|what|where|when|why|how|this|that|and|or|but|in|on|at|for|with|my|your|his|her|our|their)\b'

    # Filter for better quality utterances
    subset = subset[
        (subset['full_utterance'].str.len() >= 20) &
        (subset['full_utterance'].str.len() <= 50) &
        (~subset['full_utterance'].str.contains('xxx', case=False, na=False)) &
        (~subset['full_utterance'].str.contains('_', na=False)) &
        (~subset['full_utterance'].str.contains('participant', case=False, na=False)) &
        (subset['full_utterance'].str.match(r'^[a-zA-Z]', na=False)) &
        (subset['full_utterance'].str.match(r"^[a-zA-Z\s',\.!\?]+$", na=False)) &
        (subset['full_utterance'].str.contains(common_words, case=False, na=False, regex=True))
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

    # Use different seed for v2
    SEED = 456

    # Collect samples
    print("Collecting samples (v2 with seed=456)...")
    all_samples = []
    for age_min, age_max, label in AGE_GROUPS:
        samples = get_good_samples(df, age_min, age_max, n=5, seed=SEED)
        for i, utterance in enumerate(samples):
            all_samples.append({
                'Age Group (months)': label,
                'Sample': i + 1,
                'Utterance': utterance
            })

    samples_df = pd.DataFrame(all_samples)

    # Save as CSV
    samples_df.to_csv('output/age_groups_complete/sample_utterances_v2.csv', index=False)
    print("Saved to output/age_groups_complete/sample_utterances_v2.csv")

    # Create modern, professional figure
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('off')

    # Collect data for table
    table_data = []
    for age_min, age_max, label in AGE_GROUPS:
        samples = get_good_samples(df, age_min, age_max, n=5, seed=SEED)
        row = [label] + [f'"{s}"' for s in samples]
        table_data.append(row)

    # Column headers
    col_labels = ['Age\n(months)', 'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.08] + [0.184] * 5
    )

    # Modern styling
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.2)

    # Style header row
    header_color = '#2C3E50'
    for j in range(6):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold', color='white', fontsize=10)
        cell.set_facecolor(header_color)
        cell.set_height(0.08)

    # Style data cells with alternating rows
    row_colors = ['#FFFFFF', '#F7F9FC']
    accent_color = '#3498DB'

    for i in range(1, len(table_data) + 1):
        row_color = row_colors[(i - 1) % 2]
        for j in range(6):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_text_props(fontsize=9)
            cell.set_edgecolor('#E5E8EC')

            # Style age column
            if j == 0:
                cell.set_text_props(fontweight='bold', color=accent_color, fontsize=11)
                cell.set_facecolor('#EBF5FB')

    # Add title
    fig.suptitle('Sample Child-Directed Utterances by Age Group',
                 fontsize=18, fontweight='bold', color='#2C3E50', y=0.95)

    # Add subtitle
    fig.text(0.5, 0.90, 'Representative examples from the CHILDES database',
             ha='center', fontsize=12, color='#7F8C8D', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig('output/age_groups_complete/sample_utterances_table_v2.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved to output/age_groups_complete/sample_utterances_table_v2.png")
    plt.close()

    # Print text version
    print("\n" + "="*80)
    print("SAMPLE UTTERANCES BY AGE GROUP (v2)")
    print("="*80)

    for age_min, age_max, label in AGE_GROUPS:
        samples = get_good_samples(df, age_min, age_max, n=5, seed=SEED)
        print(f"\n{label} months:")
        print("-" * 60)
        for i, utterance in enumerate(samples, 1):
            print(f"  {i}. \"{utterance}\"")

    print("\n" + "="*80)

if __name__ == '__main__':
    create_table()
