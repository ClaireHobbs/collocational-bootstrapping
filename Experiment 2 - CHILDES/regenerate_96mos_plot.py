"""Regenerate the 96mos age group plot from existing summary CSV."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('output/age_groups_complete_96mos/summary_96mos.csv')
    results = [dict(row) for _, row in df.rename(columns={'age_min_mo': 'age_min', 'age_max_mo': 'age_max', 'n_utterances': 'n_utterances', 'n_pairs': 'n_pairs'}).iterrows()]

    from analyze_age_groups_96mos import plot_z_by_age
    plot_z_by_age(results, 'output/age_groups_complete_96mos')

if __name__ == '__main__':
    main()
