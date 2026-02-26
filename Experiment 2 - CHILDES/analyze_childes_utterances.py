#!/usr/bin/env python3
"""
Analyze CHILDES utterances CSV file and generate comprehensive data summary.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_childes_utterances(csv_path, output_path=None):
    """
    Analyze CHILDES utterances and generate summary report.

    Args:
        csv_path: Path to the utterances CSV file
        output_path: Path for output summary file (optional)
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Create output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(csv_path).parent / f"childes_utterances_summary_{timestamp}.txt"

    # Start building the summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CHILDES UTTERANCES DATA SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data file: {csv_path}")
    report_lines.append("")

    # === BASIC STATISTICS ===
    report_lines.append("=" * 80)
    report_lines.append("BASIC STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total utterances: {len(df):,}")
    report_lines.append(f"Total columns: {len(df.columns)}")
    report_lines.append(f"Column names: {', '.join(df.columns.tolist())}")
    report_lines.append("")

    # === AGE STATISTICS ===
    report_lines.append("=" * 80)
    report_lines.append("AGE STATISTICS")
    report_lines.append("=" * 80)

    # Check if age column exists (try common variations)
    age_col = None
    for col_name in ['target_child_age', 'age', 'child_age', 'Age', 'AGE']:
        if col_name in df.columns:
            age_col = col_name
            break

    if age_col is not None:
        ages = df[age_col]

        # Count NAs
        na_count = ages.isna().sum()
        valid_ages = ages.dropna()

        report_lines.append(f"Age column: {age_col}")
        report_lines.append(f"Total age values: {len(ages):,}")
        report_lines.append(f"Missing (NA) ages: {na_count:,} ({na_count/len(ages)*100:.2f}%)")
        report_lines.append(f"Valid age values: {len(valid_ages):,}")
        report_lines.append("")

        if len(valid_ages) > 0:
            report_lines.append("Age Statistics (months):")
            report_lines.append(f"  Minimum: {valid_ages.min():.2f}")
            report_lines.append(f"  Maximum: {valid_ages.max():.2f}")
            report_lines.append(f"  Mean: {valid_ages.mean():.2f}")
            report_lines.append(f"  Median: {valid_ages.median():.2f}")
            report_lines.append(f"  Std Dev: {valid_ages.std():.2f}")
            report_lines.append("")

            # Age distribution by bins
            report_lines.append("Age Distribution by Bins:")
            bins = [0, 12, 24, 36, 48, 60, 100]
            labels = ['0-12 months', '12-24 months', '24-36 months',
                     '36-48 months', '48-60 months', '60-100 months']

            age_bins = pd.cut(valid_ages, bins=bins, labels=labels, right=False)
            bin_counts = age_bins.value_counts().sort_index()

            for bin_label, count in bin_counts.items():
                percentage = (count / len(valid_ages)) * 100
                report_lines.append(f"  {bin_label}: {count:,} ({percentage:.2f}%)")
            report_lines.append("")
    else:
        report_lines.append("WARNING: No age column found in the dataset")
        report_lines.append(f"Available columns: {', '.join(df.columns.tolist())}")
        report_lines.append("")

    # === SPEAKER ROLE STATISTICS ===
    report_lines.append("=" * 80)
    report_lines.append("SPEAKER ROLE STATISTICS")
    report_lines.append("=" * 80)

    # Check for speaker role column (try common variations)
    speaker_col = None
    for col_name in ['speaker_role', 'role', 'speaker', 'Speaker', 'ROLE']:
        if col_name in df.columns:
            speaker_col = col_name
            break

    if speaker_col is not None:
        speaker_counts = df[speaker_col].value_counts()

        report_lines.append(f"Speaker role column: {speaker_col}")
        report_lines.append(f"Unique speaker roles: {len(speaker_counts)}")
        report_lines.append("")
        report_lines.append("Speaker Role Counts:")

        for role, count in speaker_counts.items():
            percentage = (count / len(df)) * 100
            report_lines.append(f"  {role}: {count:,} ({percentage:.2f}%)")
        report_lines.append("")

        # Check for NAs in speaker role
        na_speakers = df[speaker_col].isna().sum()
        if na_speakers > 0:
            report_lines.append(f"Missing speaker roles: {na_speakers:,} ({na_speakers/len(df)*100:.2f}%)")
            report_lines.append("")
    else:
        report_lines.append("WARNING: No speaker role column found in the dataset")
        report_lines.append("")

    # === MISSING DATA SUMMARY ===
    report_lines.append("=" * 80)
    report_lines.append("MISSING DATA SUMMARY")
    report_lines.append("=" * 80)

    missing_summary = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            missing_summary.append((col, missing_count, missing_pct))

    if missing_summary:
        report_lines.append("Columns with missing data:")
        # Sort by missing count (descending)
        missing_summary.sort(key=lambda x: x[1], reverse=True)

        for col, count, pct in missing_summary:
            report_lines.append(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        report_lines.append("No missing data found in any column.")
    report_lines.append("")

    # === DATA TYPES SUMMARY ===
    report_lines.append("=" * 80)
    report_lines.append("DATA TYPES SUMMARY")
    report_lines.append("=" * 80)

    dtype_counts = df.dtypes.value_counts()
    report_lines.append("Column data types:")
    for dtype, count in dtype_counts.items():
        report_lines.append(f"  {dtype}: {count} columns")
    report_lines.append("")

    # === SAMPLE DATA ===
    report_lines.append("=" * 80)
    report_lines.append("SAMPLE DATA (First 5 rows)")
    report_lines.append("=" * 80)
    report_lines.append(df.head(5).to_string())
    report_lines.append("")

    # === END OF REPORT ===
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Write report to file
    report_text = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nSummary report saved to: {output_path}")

    # Also print to console
    print("\n" + report_text)

    return output_path

def main():
    csv_path = "/Users/clairesmall/Desktop/imagining-syntax/rdata/utterances/childes_full_utterances_20251211_120050.csv"
    output_path = "/Users/clairesmall/Desktop/imagining-syntax/childes_utterances_summary.txt"

    analyze_childes_utterances(csv_path, output_path)

if __name__ == '__main__':
    main()
