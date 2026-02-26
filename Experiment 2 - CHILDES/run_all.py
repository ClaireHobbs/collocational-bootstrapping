"""
Master script to run the complete CHILDES analysis pipeline.

Steps:
1. Download CHILDES data (if not already downloaded)
2. Run overall analysis (1M utterances)
3. Run age-group analysis (6 groups)
4. Generate plots and summary

Usage:
    python run_all.py
"""

import os
import sys

def run_step(script_name, description):
    """Run a script and check for success."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print('='*60 + '\n')

    result = os.system(f'python3 {script_name}')
    if result != 0:
        print(f"\nERROR: {script_name} failed with code {result}")
        return False
    return True

def main():
    print("="*60)
    print("CHILDES Subject-Verb Zipf Analysis Pipeline")
    print("="*60)

    # Check if data exists
    data_path = 'data/childes_utterances.csv'
    if not os.path.exists(data_path):
        print("\nData file not found. Running download script...")
        if not run_step('download_childes.py', 'Download CHILDES data'):
            print("\nDownload failed. Please check your internet connection and childes-db package.")
            sys.exit(1)
    else:
        print(f"\nData file exists: {data_path}")
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"  Contains {len(df):,} utterances")

    # Run overall analysis
    if not run_step('analyze_overall.py', 'Overall Zipf analysis (1M utterances)'):
        print("\nOverall analysis failed.")
        sys.exit(1)

    # Run age-group analysis
    if not run_step('analyze_by_age.py', 'Age-group Zipf analysis (6 groups)'):
        print("\nAge-group analysis failed.")
        sys.exit(1)

    # Generate plots
    if not run_step('plot_results.py', 'Generate plots and summary'):
        print("\nPlotting failed.")
        sys.exit(1)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print("  - output/overall_results.csv")
    print("  - output/age_group_results.csv")
    print("  - output/zipf_by_age.png")
    print("  - output/zipf_fit_overall.png")
    print("  - output/summary.txt")

if __name__ == '__main__':
    main()
