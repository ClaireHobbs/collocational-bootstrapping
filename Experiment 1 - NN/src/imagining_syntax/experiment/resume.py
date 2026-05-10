"""
Resume helpers for comprehensive experiments.

Provides save_progress and load_progress for the JSON progress file that lets a
comprehensive run be resumed after interruption, plus helpers used by --extend
to read back per-iter results and count existing iterations on disk.
"""

import glob
import json
import os
from datetime import datetime


def save_progress(progress_file, param_value, status='completed', *, base_seed=None):
    """Save progress to a file for resuming later.

    If base_seed is provided (the runner's --seed argument), persist it so that
    a subsequent --extend invocation can recover the seed derivation without
    requiring the user to retype --seed.
    """
    progress_data = {
        'last_completed_param': param_value,
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    if base_seed is not None:
        progress_data['base_seed'] = base_seed
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)


def load_progress(progress_file):
    """Load progress from file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None


def count_existing_iterations(iterations_dir):
    """Return the number of iter_NNN/ directories under iterations_dir.

    Returns 0 if iterations_dir does not exist."""
    if not os.path.isdir(iterations_dir):
        return 0
    return len(glob.glob(os.path.join(iterations_dir, "iter_*")))


def read_iter_result(iteration_dir, eval_types):
    """Read accuracy_<eval_type>.txt files from a completed iteration directory.

    Returns a dict like {'seen_match': 50.0, ...} matching the shape produced
    by run_single_iteration. Eval types whose accuracy file is missing are
    silently skipped — caller is responsible for handling sparse results."""
    results = {}
    for eval_type in eval_types:
        accuracy_file = os.path.join(iteration_dir, 'results', f'accuracy_{eval_type}.txt')
        if not os.path.exists(accuracy_file):
            continue
        with open(accuracy_file, 'r') as f:
            for line in f:
                if 'Model Accuracy:' in line:
                    accuracy_str = line.split(':', 1)[1].strip()
                    results[eval_type] = float(accuracy_str.split('%')[0])
                    break
    return results
