"""
Download English utterances from CHILDES database with specified speaker roles.
"""

import pandas as pd
import childespy as cdb
import os

# Target speaker roles
TARGET_SPEAKERS = [
    'Adult', 'Caretaker', 'Father', 'Friend', 'Grandfather', 'Grandmother',
    'Investigator', 'Mother', 'Narrator', 'Playmate', 'Relative', 'Sibling',
    'Sister', 'Brother', 'Teacher', 'Unidentified', 'Visitor', 'Teenager',
    'Participant', 'Girl', 'Male', 'Student', 'Environment', 'Doctor',
    'Target_Adult'
]

def download_childes_utterances(output_path='data/childes_utterances.csv'):
    """
    Download English utterances from CHILDES with specified speaker roles.
    """
    print("Connecting to CHILDES database...")

    # Get all English utterances
    print("Downloading English utterances (this may take a while)...")
    utterances = cdb.get_utterances(language='eng')

    print(f"Total utterances downloaded: {len(utterances)}")

    # Filter by speaker role
    print(f"Filtering by speaker roles: {len(TARGET_SPEAKERS)} roles")
    filtered = utterances[utterances['speaker_role'].isin(TARGET_SPEAKERS)]

    print(f"Utterances after speaker filter: {len(filtered)}")

    # Keep relevant columns
    columns_to_keep = ['id', 'gloss', 'speaker_role', 'target_child_age']
    available_cols = [c for c in columns_to_keep if c in filtered.columns]
    filtered = filtered[available_cols]

    # Remove rows with missing utterance text
    filtered = filtered.dropna(subset=['gloss'])
    filtered = filtered[filtered['gloss'].str.strip() != '']

    print(f"Utterances after removing empty: {len(filtered)}")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return filtered

if __name__ == '__main__':
    download_childes_utterances()
