"""
Complete age-group analysis with age filtering: Estimate Zipf parameter z for 8 age groups.
Uses ALL utterances per age group (no sampling) for maximum accuracy.
FILTERS DATA to ages ≤ 96 months before analysis.

Age groups: 0-12mo, 12-24mo, 24-36mo, 36-48mo, 48-60mo, 60-72mo, 72-84mo, 84-96mo
"""

import pandas as pd
import numpy as np
import spacy
from collections import defaultdict
import os
import pickle
import re
import time
import matplotlib.pyplot as plt

TOP_VERBS = 100

# Age groups in months (min, max, label)
AGE_GROUPS = [
    (0, 12, '0-12mo'),
    (12, 24, '12-24mo'),
    (24, 36, '24-36mo'),
    (36, 48, '36-48mo'),
    (48, 60, '48-60mo'),
    (60, 72, '60-72mo'),
    (72, 84, '72-84mo'),
    (84, 96, '84-96mo')
]

def load_data(data_path='data/childes_utterances.csv', max_age=96):
    """Load all utterances and filter to ages ≤ max_age months."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total utterances loaded: {len(df):,}")

    # Filter out NaN and convert to string
    if 'full_utterance' in df.columns:
        df = df.dropna(subset=['full_utterance'])
        df['full_utterance'] = df['full_utterance'].astype(str)
        df = df.rename(columns={'full_utterance': 'gloss'})
    elif 'gloss' in df.columns:
        df = df.dropna(subset=['gloss'])
        df['gloss'] = df['gloss'].astype(str)

    print(f"Utterances after removing NaN: {len(df):,}")

    # Filter rows with valid age
    df = df.dropna(subset=['target_child_age'])
    print(f"Utterances with age info: {len(df):,}")

    # FILTER TO MAX AGE
    before_filter = len(df)
    df = df[df['target_child_age'] <= max_age].copy()
    after_filter = len(df)
    removed = before_filter - after_filter
    print(f"\nAGE FILTERING:")
    print(f"  Keeping ages ≤ {max_age} months")
    print(f"  Utterances before filter: {before_filter:,}")
    print(f"  Utterances after filter: {after_filter:,}")
    print(f"  Removed (age > {max_age}mo): {removed:,} ({removed/before_filter*100:.2f}%)")

    return df

def extract_subject_verb_pairs(utterances, nlp):
    """Extract subject-verb pairs using spaCy."""
    pairs = []
    total_pairs = 0

    start_time = time.time()
    batch_size = 1000
    utterance_list = utterances['gloss'].tolist()
    total_utterances = len(utterance_list)

    for i in range(0, total_utterances, batch_size):
        batch = utterance_list[i:i + batch_size]
        docs = nlp.pipe(batch, disable=['ner'])

        for doc in docs:
            for token in doc:
                if token.dep_ == 'nsubj' and token.head.pos_ in ('VERB', 'AUX'):
                    subject = token.lemma_.lower()
                    verb = token.head.lemma_.lower()
                    pairs.append((subject, verb))
                    total_pairs += 1

        if (i + batch_size) % 50000 == 0 and total_utterances > 50000:
            elapsed = time.time() - start_time
            rate = (i + batch_size) / elapsed
            remaining = (total_utterances - i - batch_size) / rate if rate > 0 else 0
            print(f"    Processed {i + batch_size:,}/{total_utterances:,} "
                  f"({(i + batch_size) / total_utterances * 100:.1f}%), "
                  f"{total_pairs:,} pairs, ~{remaining / 60:.1f} min left")

    return pairs

def filter_english_verbs(verb_counts):
    """Filter to keep only ASCII/English verbs."""
    pattern = re.compile(r"^[a-zA-Z'-]+$")
    return {v: c for v, c in verb_counts.items() if pattern.match(v)}

def get_top_verbs(pairs, n=TOP_VERBS):
    """Get top n verbs by total count."""
    verb_counts = defaultdict(int)
    for subj, verb in pairs:
        verb_counts[verb] += 1

    verb_counts = filter_english_verbs(dict(verb_counts))
    sorted_verbs = sorted(verb_counts.items(), key=lambda x: x[1], reverse=True)
    return [v for v, _ in sorted_verbs[:n]]

def calculate_rank_proportions(pairs, top_verbs):
    """Calculate average proportions at each rank position."""
    verb_subject_counts = defaultdict(lambda: defaultdict(int))
    for subj, verb in pairs:
        if verb in top_verbs:
            verb_subject_counts[verb][subj] += 1

    all_ranked_data = []

    for verb in top_verbs:
        subjects = verb_subject_counts[verb]
        if not subjects:
            continue

        total = sum(subjects.values())
        sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)

        for rank, (subj, freq) in enumerate(sorted_subjects, start=1):
            proportion = freq / total
            all_ranked_data.append({
                'verb': verb,
                'subject': subj,
                'frequency': freq,
                'rank': rank,
                'total': total,
                'proportion': proportion
            })

    if not all_ranked_data:
        return None, None

    combined_df = pd.DataFrame(all_ranked_data)

    rank_averages = combined_df.groupby('rank').agg(
        average_proportion=('proportion', 'mean'),
        num_verbs=('proportion', 'count'),
        sd_proportion=('proportion', 'std')
    ).reset_index()

    return rank_averages, combined_df

def predict_zipf(ranks, z):
    """Generate Zipf distribution."""
    predicted = 1.0 / np.power(ranks, z)
    return predicted / predicted.sum()

def find_optimal_z(rank_averages, z_min=0.1, z_max=3.0, z_step=0.01):
    """Find z that minimizes MSE."""
    if rank_averages is None or len(rank_averages) == 0:
        return None, None, None

    ranks = rank_averages['rank'].values
    actual = rank_averages['average_proportion'].values

    # Normalize actual to sum to 1
    actual_normalized = actual / actual.sum()

    best_z = z_min
    best_mse = float('inf')
    mse_results = []

    z_values = np.arange(z_min, z_max + z_step, z_step)
    for z in z_values:
        predicted = predict_zipf(ranks, z)
        mse = np.mean((actual_normalized - predicted) ** 2)
        mse_results.append({'z': z, 'mse': mse})

        if mse < best_mse:
            best_mse = mse
            best_z = z

    return best_z, best_mse, pd.DataFrame(mse_results)

def analyze_age_group(df, age_min, age_max, label, nlp, output_dir):
    """Analyze a single age group with ALL available data."""
    print(f"\n{'='*60}")
    print(f"Analyzing age group: {label} (ALL DATA, ≤96mo)")
    print(f"{'='*60}")

    start_time = time.time()

    # Filter by age
    age_df = df[(df['target_child_age'] >= age_min) & (df['target_child_age'] < age_max)].copy()
    print(f"Total utterances in age range: {len(age_df):,}")

    if len(age_df) == 0:
        print(f"No data for age group {label}")
        return None

    # Extract pairs
    print("  Extracting subject-verb pairs...")
    pairs = extract_subject_verb_pairs(age_df, nlp)
    n_pairs = len(pairs)
    print(f"  Subject-verb pairs: {n_pairs:,}")

    if n_pairs < 100:
        print(f"  Insufficient pairs for analysis")
        return None

    # Save pairs for this age group
    pairs_df = pd.DataFrame(pairs, columns=['subject_lemma', 'verb_lemma'])
    pairs_df.to_csv(f'{output_dir}/nsubj_verb_pairs_{label}.csv', index=False)

    # Get top verbs
    top_verbs = get_top_verbs(pairs, TOP_VERBS)
    n_verbs = len(top_verbs)
    print(f"  Top verbs found: {n_verbs}")

    if n_verbs < 10:
        print(f"  Insufficient verbs for analysis")
        return None

    # Calculate rank proportions
    rank_averages, combined_df = calculate_rank_proportions(pairs, top_verbs)

    if rank_averages is None or len(rank_averages) < 5:
        print(f"  Insufficient rank data")
        return None

    # Save intermediate data
    combined_df.to_csv(f'{output_dir}/all_verbs_ranked_{label}.csv', index=False)
    rank_averages.to_csv(f'{output_dir}/rank_averages_{label}.csv', index=False)

    # Find optimal z
    optimal_z, mse, mse_df = find_optimal_z(rank_averages)
    mse_df.to_csv(f'{output_dir}/mse_search_{label}.csv', index=False)

    elapsed = time.time() - start_time
    print(f"  Optimal z: {optimal_z:.2f} (MSE: {mse:.6f})")
    print(f"  Processing time: {elapsed / 60:.1f} minutes")

    # Create actual vs predicted comparison
    ranks = rank_averages['rank'].values
    actual_normalized = rank_averages['average_proportion'].values / rank_averages['average_proportion'].sum()
    predicted = predict_zipf(ranks, optimal_z)

    comparison = pd.DataFrame({
        'rank': ranks,
        'actual_frequency': actual_normalized,
        'predicted_frequency': predicted,
        'squared_error': (actual_normalized - predicted) ** 2,
        'num_verbs': rank_averages['num_verbs'].values
    })
    comparison.to_csv(f'{output_dir}/actual_vs_predicted_{label}.csv', index=False)

    return {
        'age_group': label,
        'age_min': age_min,
        'age_max': age_max,
        'z': optimal_z,
        'mse': mse,
        'n_utterances': len(age_df),
        'n_pairs': n_pairs,
        'n_unique_verbs': len(set(v for _, v in pairs)),
        'n_unique_subjects': len(set(s for s, _ in pairs)),
        'top_verbs': top_verbs[:10],
        'runtime_minutes': elapsed / 60
    }

def plot_z_by_age(results, output_dir):
    """Create publication-ready plot showing z values across age groups."""
    if not results:
        print("No results to plot")
        return

    df = pd.DataFrame([{
        'age_group': r['age_group'],
        'z': r['z'],
        'n_utterances': r['n_utterances'],
        'n_pairs': r['n_pairs']
    } for r in results])

    # Set up publication-ready styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(5, 3))

    x = np.arange(len(df))
    z_values = df['z'].values
    age_labels = df['age_group'].values

    # Strip "mo" from labels for cleaner x-axis (e.g., "0-12mo" -> "0-12")
    clean_labels = [l.replace('mo', '') for l in age_labels]

    # Plot line connecting points
    ax.plot(x, z_values, color='black', linewidth=1, zorder=1)

    # Plot large, readable points
    ax.scatter(x, z_values, s=60, color='#3182bd', edgecolor='#2C3E50',
               linewidth=1.5, zorder=2)

    # Add value labels above each point
    for i, (xi, zi) in enumerate(zip(x, z_values)):
        # Move 1.37 label below the point to avoid intersecting the red line
        if abs(zi - 1.37) < 0.005:
            ax.text(xi, zi - 0.02, f'{zi:.2f}',
                    ha='center', va='top', fontsize=10, fontweight='normal',
                    color='black')
        else:
            ax.text(xi, zi + 0.02, f'{zi:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='normal',
                    color='black')

    # Reference lines
    ax.axhline(y=1.43, color='#756bb1', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(len(x) - 0.5, 1.435, r'Overall $\alpha$ = 1.43',
            ha='right', va='bottom', fontsize=10, color='#756bb1')

    ax.axhline(y=1.4, color='#de2d26', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(len(x) - 0.5, 1.395, r'Model optimal $\alpha$ $\approx$ 1.4',
            ha='right', va='top', fontsize=10, color='#de2d26')

    # Axis labels
    ax.set_xlabel('Child Age Group (Months)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Zipf Parameter ($\alpha$)', fontsize=12, fontweight='bold')

    # X-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=10)

    # Y-axis range
    ax.set_ylim(1.15, 1.60)

    # Y-axis ticks
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/zipf_by_age_96mos_alpha.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nPlot saved to {output_dir}/zipf_by_age_96mos_alpha.png")
    plt.close()

    # Reset rcParams to defaults
    plt.rcParams.update(plt.rcParamsDefault)

def main():
    total_start = time.time()

    # Create output directory
    output_dir = 'output/age_groups_complete_96mos'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load spaCy
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')

    # Load ALL data with age filter
    df = load_data(max_age=96)

    # Analyze each age group
    results = []
    for age_min, age_max, label in AGE_GROUPS:
        result = analyze_age_group(df, age_min, age_max, label, nlp, output_dir)
        if result:
            results.append(result)

    # Create summary
    print("\n" + "="*70)
    print("SUMMARY: Zipf parameter z by age group (FILTERED TO ≤96 MONTHS)")
    print("="*70)

    if results:
        summary_df = pd.DataFrame([{
            'age_group': r['age_group'],
            'age_min_mo': r['age_min'],
            'age_max_mo': r['age_max'],
            'z': r['z'],
            'mse': r['mse'],
            'n_utterances': r['n_utterances'],
            'n_pairs': r['n_pairs'],
            'n_unique_verbs': r['n_unique_verbs'],
            'n_unique_subjects': r['n_unique_subjects']
        } for r in results])

        print(summary_df.to_string(index=False))
        print("="*70)

        # Save summary
        summary_df.to_csv(f'{output_dir}/summary_96mos.csv', index=False)
        print(f"\nSummary saved to {output_dir}/summary_96mos.csv")

        # Create plot
        plot_z_by_age(results, output_dir)

        # Save full results as pickle
        with open(f'{output_dir}/results_96mos.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Full results saved to {output_dir}/results_96mos.pkl")

    else:
        print("\nNo results generated!")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Total runtime: {total_elapsed / 60:.1f} minutes")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
