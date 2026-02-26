#!/usr/bin/env python3
"""
Analyze age outliers in CHILDES utterances data (ages > 100 months).
"""

import pandas as pd
import numpy as np

def analyze_age_outliers(csv_path):
    """Analyze utterances with age > 100 months."""
    print("Reading CSV file...")
    df = pd.read_csv(csv_path, dtype={'speaker_name': str, 'target_child_name': str})

    # Get age column
    age_col = 'target_child_age'

    # Filter to valid ages
    valid_ages_df = df[df[age_col].notna()].copy()
    total_valid = len(valid_ages_df)

    print(f"\nTotal utterances with valid ages: {total_valid:,}")
    print(f"Age range: {valid_ages_df[age_col].min():.2f} - {valid_ages_df[age_col].max():.2f} months")

    # Analyze ages > 100 months
    outliers = valid_ages_df[valid_ages_df[age_col] > 100]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / total_valid) * 100

    print(f"\n{'='*70}")
    print("AGE OUTLIERS (> 100 months / > 8.3 years)")
    print(f"{'='*70}")
    print(f"Utterances with age > 100 months: {outlier_count:,}")
    print(f"Percentage of valid ages: {outlier_pct:.2f}%")
    print(f"Percentage of all utterances: {(outlier_count/len(df))*100:.2f}%")

    if outlier_count > 0:
        print(f"\nAge statistics for outliers:")
        print(f"  Minimum: {outliers[age_col].min():.2f} months ({outliers[age_col].min()/12:.2f} years)")
        print(f"  Maximum: {outliers[age_col].max():.2f} months ({outliers[age_col].max()/12:.2f} years)")
        print(f"  Mean: {outliers[age_col].mean():.2f} months ({outliers[age_col].mean()/12:.2f} years)")
        print(f"  Median: {outliers[age_col].median():.2f} months ({outliers[age_col].median()/12:.2f} years)")

        # Age distribution of outliers
        print(f"\nAge distribution (months):")
        age_bins_outliers = [100, 120, 150, 200, 300]
        for i in range(len(age_bins_outliers)-1):
            bin_low = age_bins_outliers[i]
            bin_high = age_bins_outliers[i+1]
            bin_count = len(outliers[(outliers[age_col] > bin_low) & (outliers[age_col] <= bin_high)])
            bin_pct = (bin_count / outlier_count) * 100 if outlier_count > 0 else 0
            print(f"  {bin_low}-{bin_high} months: {bin_count:,} ({bin_pct:.2f}%)")

        # Speaker role distribution for outliers
        print(f"\nSpeaker roles in outlier utterances:")
        speaker_counts = outliers['speaker_role'].value_counts().head(10)
        for role, count in speaker_counts.items():
            pct = (count / outlier_count) * 100
            print(f"  {role}: {count:,} ({pct:.2f}%)")

        # Corpus distribution for outliers
        print(f"\nCorpus distribution in outlier utterances:")
        corpus_counts = outliers['corpus_name'].value_counts().head(10)
        for corpus, count in corpus_counts.items():
            pct = (count / outlier_count) * 100
            print(f"  {corpus}: {count:,} ({pct:.2f}%)")

    # Compare with data excluding outliers
    print(f"\n{'='*70}")
    print("COMPARISON: WITH vs WITHOUT OUTLIERS")
    print(f"{'='*70}")

    without_outliers = valid_ages_df[valid_ages_df[age_col] <= 100]

    print(f"\nWith outliers (ages 3-284 months):")
    print(f"  Count: {total_valid:,}")
    print(f"  Mean age: {valid_ages_df[age_col].mean():.2f} months")
    print(f"  Median age: {valid_ages_df[age_col].median():.2f} months")
    print(f"  Std Dev: {valid_ages_df[age_col].std():.2f} months")

    print(f"\nWithout outliers (ages 3-100 months):")
    print(f"  Count: {len(without_outliers):,}")
    print(f"  Mean age: {without_outliers[age_col].mean():.2f} months")
    print(f"  Median age: {without_outliers[age_col].median():.2f} months")
    print(f"  Std Dev: {without_outliers[age_col].std():.2f} months")

    print(f"\nData loss if excluding outliers:")
    print(f"  Utterances lost: {outlier_count:,} ({outlier_pct:.2f}%)")
    print(f"  Utterances retained: {len(without_outliers):,} ({(len(without_outliers)/total_valid)*100:.2f}%)")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if outlier_pct < 1:
        print(f"The outliers represent only {outlier_pct:.2f}% of valid data.")
        print("RECOMMEND: Exclude outliers for typical child language acquisition studies.")
        print("Ages > 100 months (8.3 years) are beyond typical early childhood language acquisition.")
    elif outlier_pct < 5:
        print(f"The outliers represent {outlier_pct:.2f}% of valid data.")
        print("CONSIDER: Excluding outliers depending on your research focus.")
        print("If studying early language acquisition (0-8 years), exclude them.")
        print("If studying broader age ranges or longitudinal data, keep them.")
    else:
        print(f"The outliers represent {outlier_pct:.2f}% of valid data - a substantial amount.")
        print("CAUTION: Excluding outliers would lose significant data.")
        print("Review the corpus composition and research goals carefully before deciding.")

def main():
    csv_path = "/Users/clairesmall/Desktop/imagining-syntax/rdata/utterances/childes_full_utterances_20251211_120050.csv"
    analyze_age_outliers(csv_path)

if __name__ == '__main__':
    main()
