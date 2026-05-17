"""
Complete dataset analysis with age filtering: Extract subject-verb pairs and estimate Zipf parameter z.
Uses ALL utterances (no sampling) for maximum accuracy.
FILTERS DATA to ages ≤ 96 months before analysis.
"""

import pandas as pd
import numpy as np
import spacy
from collections import defaultdict
import os
import pickle
import re
import time

TOP_VERBS = 100

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
    before_age_filter = len(df)
    df = df.dropna(subset=['target_child_age'])
    print(f"Utterances with age info: {len(df):,}")

    # FILTER TO MAX AGE
    before_filter = len(df)
    df = df[df['target_child_age'] <= max_age].copy()
    after_filter = len(df)
    removed = before_filter - after_filter

    print(f"\n{'='*60}")
    print(f"AGE FILTERING:")
    print(f"{'='*60}")
    print(f"  Keeping ages ≤ {max_age} months")
    print(f"  Utterances before filter: {before_filter:,}")
    print(f"  Utterances after filter: {after_filter:,}")
    print(f"  Removed (age > {max_age}mo): {removed:,} ({removed/before_filter*100:.2f}%)")
    print(f"{'='*60}\n")

    return df

def extract_subject_verb_pairs(utterances, nlp):
    """
    Extract subject-verb pairs using spaCy.
    Returns list of (subject_lemma, verb_lemma) tuples.
    """
    pairs = []
    total_pairs = 0

    print("Extracting subject-verb pairs...")
    start_time = time.time()
    batch_size = 1000
    utterance_list = utterances['gloss'].tolist()
    total_utterances = len(utterance_list)

    for i in range(0, total_utterances, batch_size):
        batch = utterance_list[i:i + batch_size]
        docs = nlp.pipe(batch, disable=['ner'])

        for doc in docs:
            for token in doc:
                # Find nsubj relations where head is VERB or AUX
                if token.dep_ == 'nsubj' and token.head.pos_ in ('VERB', 'AUX'):
                    subject = token.lemma_.lower()
                    verb = token.head.lemma_.lower()
                    pairs.append((subject, verb))
                    total_pairs += 1

        if (i + batch_size) % 100000 == 0:
            elapsed = time.time() - start_time
            rate = (i + batch_size) / elapsed
            remaining = (total_utterances - i - batch_size) / rate if rate > 0 else 0
            print(f"  Processed {i + batch_size:,}/{total_utterances:,} utterances "
                  f"({(i + batch_size) / total_utterances * 100:.1f}%), "
                  f"{total_pairs:,} pairs found, "
                  f"~{remaining / 60:.1f} min remaining")

    elapsed = time.time() - start_time
    print(f"Total subject-verb pairs extracted: {total_pairs:,}")
    print(f"Extraction time: {elapsed / 60:.1f} minutes")

    return pairs

def filter_english_verbs(verb_counts):
    """Filter to keep only ASCII/English verbs (letters, hyphens, apostrophes)."""
    pattern = re.compile(r"^[a-zA-Z'-]+$")
    filtered = {v: c for v, c in verb_counts.items() if pattern.match(v)}
    removed = len(verb_counts) - len(filtered)
    print(f"Removed {removed} non-English verbs")
    return filtered

def get_top_verbs(pairs, n=TOP_VERBS):
    """Get top n verbs by total count."""
    verb_counts = defaultdict(int)
    for subj, verb in pairs:
        verb_counts[verb] += 1

    # Filter non-English verbs
    verb_counts = filter_english_verbs(dict(verb_counts))

    sorted_verbs = sorted(verb_counts.items(), key=lambda x: x[1], reverse=True)
    top = [v for v, _ in sorted_verbs[:n]]
    print(f"Top 10 verbs: {top[:10]}")
    return top

def calculate_rank_proportions(pairs, top_verbs):
    """
    For each top verb, rank subjects by frequency and calculate proportions.
    Returns DataFrame with average proportions at each rank position.
    """
    # Build verb -> subject counts
    verb_subject_counts = defaultdict(lambda: defaultdict(int))
    for subj, verb in pairs:
        if verb in top_verbs:
            verb_subject_counts[verb][subj] += 1

    # For each verb, calculate ranked proportions
    all_ranked_data = []

    for verb in top_verbs:
        subjects = verb_subject_counts[verb]
        if not subjects:
            continue

        total = sum(subjects.values())

        # Sort subjects by count (descending) and assign ranks
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

    combined_df = pd.DataFrame(all_ranked_data)

    # Calculate average proportion by rank (across all verbs)
    rank_averages = combined_df.groupby('rank').agg(
        average_proportion=('proportion', 'mean'),
        num_verbs=('proportion', 'count'),
        sd_proportion=('proportion', 'std')
    ).reset_index()

    print(f"Rank averages calculated for {len(rank_averages)} ranks")

    return rank_averages, combined_df

def predict_zipf(ranks, z):
    """Generate Zipf distribution: p(rank) = (1/rank^z) / sum(1/i^z)"""
    predicted = 1.0 / np.power(ranks, z)
    return predicted / predicted.sum()

def find_optimal_z(rank_averages, z_min=0.1, z_max=3.0, z_step=0.01):
    """
    Find z that minimizes MSE between actual and predicted distributions.
    Normalizes both actual and predicted to sum to 1 before comparison.
    """
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

def main():
    start_time = time.time()

    # Create output directory
    output_dir = 'output/complete_dataset_96mos'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')

    # Load ALL data with age filter
    df = load_data(max_age=96)

    # Extract subject-verb pairs
    pairs = extract_subject_verb_pairs(df, nlp)

    # Save pairs
    print(f"\nSaving {len(pairs):,} pairs...")
    pairs_df = pd.DataFrame(pairs, columns=['subject_lemma', 'verb_lemma'])
    pairs_df.to_csv(f'{output_dir}/nsubj_verb_pairs_96mos.csv', index=False)
    print(f"Saved to {output_dir}/nsubj_verb_pairs_96mos.csv")

    # Get top verbs
    top_verbs = get_top_verbs(pairs, TOP_VERBS)

    # Calculate average rank proportions
    rank_averages, combined_ranked = calculate_rank_proportions(pairs, top_verbs)

    # Save intermediate data
    combined_ranked.to_csv(f'{output_dir}/all_verbs_ranked_96mos.csv', index=False)
    rank_averages.to_csv(f'{output_dir}/rank_averages_96mos.csv', index=False)

    # Find optimal z
    print("\nFinding optimal Zipf parameter z...")
    optimal_z, mse, mse_df = find_optimal_z(rank_averages)
    mse_df.to_csv(f'{output_dir}/mse_search_96mos.csv', index=False)

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"COMPLETE DATASET RESULTS (≤96 MONTHS)")
    print(f"{'='*60}")
    print(f"Total utterances: {len(df):,}")
    print(f"Total subject-verb pairs: {len(pairs):,}")
    print(f"Unique verbs: {len(set(v for _, v in pairs)):,}")
    print(f"Unique subjects: {len(set(s for s, _ in pairs)):,}")
    print(f"{'='*60}")
    print(f"Optimal z: {optimal_z:.2f}")
    print(f"MSE: {mse:.6f}")
    print(f"{'='*60}")
    print(f"Total runtime: {total_time / 60:.1f} minutes")

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
    comparison.to_csv(f'{output_dir}/actual_vs_predicted_96mos.csv', index=False)

    # Save summary
    summary = {
        'dataset': 'complete_96mos',
        'max_age_months': 96,
        'n_utterances': len(df),
        'n_pairs': len(pairs),
        'n_unique_verbs': len(set(v for _, v in pairs)),
        'n_unique_subjects': len(set(s for s, _ in pairs)),
        'optimal_z': optimal_z,
        'mse': mse,
        'top_10_verbs': top_verbs[:10],
        'runtime_minutes': total_time / 60
    }

    with open(f'{output_dir}/summary_96mos.pkl', 'wb') as f:
        pickle.dump(summary, f)

    # Save summary as readable text
    with open(f'{output_dir}/summary_96mos.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPLETE DATASET ANALYSIS (≤96 MONTHS)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Max age filter: {summary['max_age_months']} months\n")
        f.write(f"Total utterances: {summary['n_utterances']:,}\n")
        f.write(f"Total subject-verb pairs: {summary['n_pairs']:,}\n")
        f.write(f"Unique verbs: {summary['n_unique_verbs']:,}\n")
        f.write(f"Unique subjects: {summary['n_unique_subjects']:,}\n\n")
        f.write(f"Optimal Zipf parameter (z): {summary['optimal_z']:.2f}\n")
        f.write(f"Mean squared error: {summary['mse']:.6f}\n\n")
        f.write(f"Top 10 verbs: {', '.join(summary['top_10_verbs'])}\n\n")
        f.write(f"Runtime: {summary['runtime_minutes']:.1f} minutes\n")
        f.write("="*60 + "\n")

    print(f"\nSummary saved to {output_dir}/summary_96mos.txt")
    print(f"Full results saved to {output_dir}/summary_96mos.pkl")

if __name__ == '__main__':
    main()
