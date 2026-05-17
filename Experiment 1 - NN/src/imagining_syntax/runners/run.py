#!/usr/bin/env python3
"""
Run comprehensive experiments across multiple parameter values with multiple iterations each.
Supports zipfian (Z parameter, default) and oneshot distributions.

Usage: imsyn run [--parameter-range MIN,MAX] [--step STEP] [--n-iterations N] [--oneshot]
Examples:
  imsyn run
  imsyn run --parameter-range 0.0,3.0 --step 0.1 --n-iterations 5
  imsyn run --oneshot --n-iterations 10
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
from imagining_syntax.data.distributions import (
    format_param_for_filename, get_parameter_name, validate_parameter
)
from imagining_syntax.experiment.stats import calculate_statistics, save_experiment_summary, create_comprehensive_summary
from imagining_syntax.experiment.resume import (
    save_progress, load_progress, count_existing_iterations, read_iter_result
)
from imagining_syntax.experiment.runner import (
    prepare_data_dir, prepare_eval_sets, build_training_args, train_one, evaluate_all,
)
from imagining_syntax.utils.seed import set_global_seed
from imagining_syntax.plotting import plot_oneshot_bars, plot_zipfian_sweep

def generate_param_values(param_min, param_max, param_step):
    """Generate list of parameter values with proper floating point handling."""
    # Use numpy to avoid floating point precision issues
    param_values = np.arange(param_min, param_max + param_step/2, param_step)
    # Round to avoid floating point precision artifacts
    param_values = np.round(param_values, 10)
    # Ensure we don't exceed param_max due to floating point errors
    param_values = param_values[param_values <= param_max]
    return param_values.tolist()

def create_comprehensive_directory(param_min, param_max, param_step, n_iterations, distribution_type, resume_dir=None, experiment_name=None):
    """Create directory structure for comprehensive experiments."""
    if resume_dir:
        # Use existing directory for resuming
        base_dir = resume_dir
    elif experiment_name:
        # Use provided experiment name verbatim under runs/
        base_dir = f"runs/{experiment_name}"
    else:
        # Create new directory
        param_name = get_parameter_name(distribution_type)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"runs/{timestamp}_{param_name}_{param_min}-{param_max}_step{param_step}_n{n_iterations}"

    # Create subdirectories
    dirs = {
        'base': base_dir,
        'experiments': os.path.join(base_dir, 'experiments'),
        'summary': os.path.join(base_dir, 'summary')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def run_experiments_for_param(param_value, distribution_type, n_iterations, experiments_dir, eval_types=None, vocab_size=40, unseen_count=10, *, base_seed=None, p_idx=0, start_iter=0, existing_results=None):
    """Run n iterations for a single parameter value directly in the comprehensive experiments directory.

    When base_seed is set, the iteration with global index `idx = start_iter + i`
    uses iter_seed = base_seed + 1000*p_idx + idx, forwarded to dataset/pairs/
    training sub-processes via --seed. Iteration directories are named
    iter_{idx+1:03d}, so an extend run that adds to existing iter_001..iter_005
    creates iter_006, iter_007, ... with seeds that continue the original
    sequence.

    seed.txt artifact: before sub-scripts launch, the runner writes
    iter_seed as a single integer + newline to `iteration_dir/seed.txt`.
    Tests and post-hoc analysis can read this file to verify the seed
    derivation without inspecting subprocess arguments. Unseeded iterations
    (iter_seed is None) do not produce a seed.txt.

    `existing_results` is a list of result dicts from previously-completed
    iterations under this same param value. They are concatenated with the
    newly-run iterations before computing per-param statistics so the saved
    summary reflects the full cohort, not just the new iterations.
    """
    param_name = get_parameter_name(distribution_type)
    print(f"\n\n{'#'*80}")
    print(f"# RUNNING EXPERIMENTS FOR {distribution_type.upper()} DISTRIBUTION: {param_name}={param_value}")
    print(f"# {n_iterations} iterations")
    if eval_types:
        print(f"# Eval types: {', '.join(eval_types)}")
    else:
        print(f"# Eval types: all (seen_match, seen_mismatch, unseen_match, unseen_mismatch)")
    print(f"{'#'*80}")

    # Create deterministic directory for this parameter value
    param_str = format_param_for_filename(distribution_type, param_value)
    param_name = get_parameter_name(distribution_type)
    param_experiment_dir = os.path.join(experiments_dir, f"{param_name}_{param_str}")

    print(f">>> Experiment directory: {param_experiment_dir}")

    # Create the specific experiment directory structure
    os.makedirs(param_experiment_dir, exist_ok=True)

    # Run our own experiment loop
    print(f"\n>>> Starting {n_iterations} iterations for {param_name}={param_value}...")

    # Create subdirectories
    iterations_dir = os.path.join(param_experiment_dir, 'iterations')
    summary_dir = os.path.join(param_experiment_dir, 'summary')
    os.makedirs(iterations_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Run iterations
    new_results = []
    for i in range(n_iterations):
        idx = start_iter + i  # 0-based global iteration index across the full sequence
        print(f"\n>>> Running iteration {idx+1} for {param_name}={param_value}...")

        iteration_dir = os.path.join(iterations_dir, f'iter_{idx+1:03d}')
        iter_seed = base_seed + 1000 * p_idx + idx if base_seed is not None else None
        result = run_single_iteration(param_value, distribution_type, idx, iteration_dir, eval_types, vocab_size, unseen_count, iter_seed=iter_seed)
        new_results.append(result)

        # Show quick result
        result_str = f">>> Iteration {idx+1} results: "
        result_parts = []
        for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
            if eval_type in result:
                result_parts.append(f"{eval_type}={result[eval_type]:.1f}%")
        print(result_str + ", ".join(result_parts))

    # Combine prior iterations (if extending) with the newly-run ones so the
    # summary reflects the full cohort, not just the increment.
    all_results = list(existing_results or []) + new_results
    total_iters = len(all_results)

    # Calculate statistics and save summary
    statistics = calculate_statistics(all_results)
    save_experiment_summary(param_value, distribution_type, total_iters, all_results, statistics,
                           summary_dir)

    print(f">>> Completed all experiments for {param_name}={param_value}")
    return statistics

def run_single_iteration(param_value, distribution_type, iteration, iteration_dir, eval_types=None, vocab_size=40, unseen_count=10, *, iter_seed=None):
    """Run a single experiment iteration.

    Args:
        param_value: Distribution parameter value (Z for Zipfian)
        distribution_type: "zipfian" or "oneshot"
        iteration: Iteration number (0-based)
        iteration_dir: Directory to store iteration results
        eval_types: List of evaluation types to run. If None, runs all.
                   Options: ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']
        iter_seed: If set, forwarded to data/pairs/training helpers as the seed
                   they should use for their own RNG initialization.
    """
    # Defense-in-depth: every helper below (prepare_data_dir, run_training via
    # train_one, prepare_eval_sets) re-seeds for its own RNG consumption, so
    # this top-level call is redundant for output determinism. Kept so that
    # any code added later before the helpers also runs from a known RNG state.
    if iter_seed is not None:
        set_global_seed(iter_seed)

    # Default to all eval types if none specified
    if eval_types is None:
        eval_types = ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']

    # Create iteration directory structure
    dirs = {
        'base': iteration_dir,
        'data': os.path.join(iteration_dir, 'data'),
        'model': os.path.join(iteration_dir, 'model'),
        'eval': os.path.join(iteration_dir, 'eval'),
        'results': os.path.join(iteration_dir, 'results')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Write seed artifact so tests (and post-hoc analysis) can read the seed
    # used by this iteration without inspecting subprocess arguments.
    if iter_seed is not None:
        with open(os.path.join(iteration_dir, 'seed.txt'), 'w') as _sf:
            _sf.write(f"{iter_seed}\n")

    # Step 1: Generate dataset
    prepare_data_dir(
        param_value, iter_seed, output_root=iteration_dir,
        vocab_size=vocab_size, unseen_count=unseen_count,
        distribution_type=distribution_type,
    )

    # Step 2: Train new model
    train_args = build_training_args(
        data_dir=dirs['data'],
        model_save_dir=dirs['model'],
        seed=iter_seed,
    )
    train_one(train_args, output_root=iteration_dir, model_subdir='model')

    # Step 3: Generate all minimal pairs (prepare_eval_sets generates all 4 conditions)
    eval_sets = prepare_eval_sets(
        param_value, iter_seed, output_root=iteration_dir,
        vocab_size=vocab_size, unseen_count=unseen_count,
        distribution_type=distribution_type,
    )

    # Step 4: Evaluate model on requested eval types only
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accuracies = evaluate_all(
        model_dir=dirs['model'],
        eval_sets=eval_sets,
        output_root=iteration_dir,
        conditions=tuple(eval_types),
        device=device,
    )

    # Free GPU memory between iterations.
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Convert fractional accuracy to percentage (matching the format that
    # calculate_statistics and read_iter_result expect: e.g. 87.5 not 0.875).
    results = {
        cond: info['accuracy'] * 100
        for cond, info in accuracies.items()
    }
    return results

def add_parser(subparsers):
    """Register the `imsyn run` leaf subparser."""
    p = subparsers.add_parser(
        "run",
        help="Run a Zipfian α sweep (or oneshot) and write the autogen plot.",
        description=(
            "Runs a Zipfian α sweep with paper defaults (α=0–3, step 0.1, "
            "10 iterations per α, seed 42). Override any default with the "
            "flags below. Use --oneshot for the α→∞ limit case."
        ),
    )
    p.add_argument('--parameter-range', type=str, default='0.0,3.0',
                   metavar='MIN,MAX',
                   help='Zipfian α range as MIN,MAX (default: 0.0,3.0). '
                        'Ignored when --oneshot is set.')
    p.add_argument('--step', type=float, default=0.1,
                   help='Step size for the α range (default: 0.1). '
                        'Ignored when --oneshot is set.')
    p.add_argument('--n-iterations', type=int, default=10,
                   help='Number of iterations per α value (default: 10).')
    p.add_argument('--oneshot', action='store_true',
                   help='Use the oneshot (α→∞) limit case instead of zipfian. '
                        '--parameter-range and --step are ignored.')
    p.add_argument('--seed', type=int, default=None,
                   help='Base seed; iter i for param p uses '
                        '(seed + 1000*p_idx + i) (default: None / unseeded).')
    p.add_argument('--vocab-size', type=int, default=40,
                   help='Vocabulary size (default: 40).')
    p.add_argument('--unseen-count', type=int, default=10,
                   help='Number of unseen pairs (default: 10).')
    p.add_argument('--eval-types', type=str, nargs='+',
                   choices=['seen_match', 'seen_mismatch',
                            'unseen_match', 'unseen_mismatch'],
                   help='Evaluation types to run (default: all four).')
    p.add_argument('--experiment-name', type=str,
                   help='Override the timestamped runs/ subdirectory name.')
    p.add_argument('--resume', type=str, metavar='DIR',
                   help='Resume from a previous run by experiment directory.')
    p.add_argument('--extend', type=int, metavar='N',
                   help='Extend an existing run with N more iterations.')
    p.set_defaults(func=main)
    return p


def main(args):
    """Run the experiment with parsed args."""
    # Translate --parameter-range and --oneshot into the legacy fields the
    # rest of main() reads.
    args.distribution_type = "oneshot" if args.oneshot else "zipfian"
    if args.oneshot:
        args.param_min = 0.0
        args.param_max = 0.0
        args.param_step = 1.0  # one param value
    else:
        try:
            lo, hi = args.parameter_range.split(",")
            args.param_min = float(lo)
            args.param_max = float(hi)
        except (ValueError, AttributeError):
            print(f"Error: --parameter-range must be MIN,MAX (got {args.parameter_range!r})")
            sys.exit(1)
        args.param_step = args.step

    # Validate arguments based on distribution type
    try:
        validate_parameter(args.distribution_type, args.param_min)
        validate_parameter(args.distribution_type, args.param_max)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.param_min > args.param_max:
        print("Error: parameter-range MIN must be <= MAX")
        sys.exit(1)

    if args.param_step <= 0:
        print("Error: --step must be positive")
        sys.exit(1)

    if args.n_iterations < 1:
        print("Error: --n-iterations must be at least 1")
        sys.exit(1)

    if args.extend is not None:
        if args.extend < 1:
            print("Error: --extend N must be >= 1")
            sys.exit(1)
        if not args.resume:
            print("Error: --extend requires --resume DIR")
            sys.exit(1)

    # Generate parameter values
    param_values = generate_param_values(args.param_min, args.param_max, args.param_step)

    # Handle resuming from previous run
    resume_from_param = None
    progress_data = None
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Resume directory '{args.resume}' does not exist")
            sys.exit(1)

        # Load progress file
        progress_file = os.path.join(args.resume, 'progress.json')
        progress_data = load_progress(progress_file)

        if progress_data:  # pragma: no cover - resume-mid-flow branches; covered transitively by the slow comprehensive resume test which exercises the success path
            # Handle both old and new progress file formats
            last_param = progress_data.get('last_completed_param') or progress_data.get('last_completed_c')
            if last_param is None:
                # Progress file present but lacks a completed param marker (stale/partial
                # state). Fall through to running all param values from the beginning.
                print(f"\nWarning: progress file in '{args.resume}' has no completed "
                      f"param value; starting from beginning")
            elif args.extend is not None:
                # In extend mode we DO NOT filter param_values: every param gets +N
                # more iterations regardless of last_completed_param.
                param_name = get_parameter_name(args.distribution_type)
                print(f"\n*** EXTENDING run; last completed at {param_name}={last_param} "
                      f"({progress_data.get('timestamp', '<unknown>')}) ***")
            else:
                resume_from_param = last_param
                param_name = get_parameter_name(args.distribution_type)
                print(f"\n*** RESUMING from after {param_name}={last_param} ***")
                print(f"Last completed at: {progress_data.get('timestamp', '<unknown>')}")

                # Filter parameter values to only those after the last completed
                param_values = [p for p in param_values if p > last_param]

                if not param_values:
                    print(f"\nAll {param_name} values were already completed!")
                    sys.exit(0)
        else:
            print(f"\nWarning: No progress file found in '{args.resume}', starting from beginning")

    # Determine the base seed actually used for iter_seed derivation. In extend
    # mode, prefer the seed recorded in progress.json (so the new iterations
    # continue the original sequence even if the user forgot to retype --seed).
    effective_base_seed = args.seed
    if args.extend is not None and progress_data is not None:
        recorded_seed = progress_data.get('base_seed')
        if recorded_seed is not None:
            effective_base_seed = recorded_seed
    if args.extend is not None and effective_base_seed is None:
        print("Error: --extend needs a base seed. Pass --seed when progress.json "
              "does not record one.")
        sys.exit(1)

    # In extend mode, n_iter_to_run is the number of NEW iterations to add per
    # param; the positional n_iterations is ignored.
    n_iter_to_run = args.extend if args.extend is not None else args.n_iterations

    param_name = get_parameter_name(args.distribution_type)
    print(f"\n{'*'*80}")
    print(f"* COMPREHENSIVE EXPERIMENT PLAN")
    if args.extend is not None:
        print(f"* EXTENDING each {param_name} value by {args.extend} iterations")
    elif resume_from_param is not None:  # pragma: no cover - resume status banner
        print(f"* RESUMING after {param_name}={resume_from_param}")
    print(f"* Distribution type: {args.distribution_type}")
    print(f"* {param_name} values to process: {param_values}")
    print(f"* Iterations per {param_name}: {n_iter_to_run}")
    print(f"* Total experiments: {len(param_values) * n_iter_to_run}")
    print(f"{'*'*80}")

    # Create or use existing comprehensive experiment directory
    dirs = create_comprehensive_directory(args.param_min, args.param_max, args.param_step,
                                        args.n_iterations, args.distribution_type,
                                        args.resume, args.experiment_name)
    print(f"\nComprehensive experiment directory: {dirs['base']}")

    # Create progress file
    progress_file = os.path.join(dirs['base'], 'progress.json')

    # Load existing statistics if resuming
    all_statistics = {}
    if args.resume:  # pragma: no cover - pre-existing summary load on resume; covered transitively by resume slow test
        summary_json = os.path.join(dirs['summary'], 'comprehensive_summary.json')
        if os.path.exists(summary_json):
            with open(summary_json, 'r') as f:
                existing_data = json.load(f)
                # Handle both old and new JSON formats
                stats_key = 'statistics_by_param' if 'statistics_by_param' in existing_data else 'statistics_by_c'
                if stats_key in existing_data:
                    # Convert string keys back to float
                    all_statistics = {float(k): v for k, v in existing_data[stats_key].items()}

    eval_types_for_read = args.eval_types or ['seen_match', 'seen_mismatch',
                                              'unseen_match', 'unseen_mismatch']

    # In extend mode, the n_iterations field reported in the comprehensive
    # summary should reflect the full per-param cohort (existing + new). Compute
    # it once from the first param's existing iter count; the per-param JSON
    # summary independently records each param's own actual count.
    comprehensive_n_iterations = n_iter_to_run
    if args.extend is not None and param_values:
        first_param = param_values[0]
        first_param_str = format_param_for_filename(args.distribution_type, first_param)
        first_param_dir = os.path.join(
            dirs['experiments'], f"{param_name}_{first_param_str}", 'iterations'
        )
        comprehensive_n_iterations = (
            count_existing_iterations(first_param_dir) + n_iter_to_run
        )

    for i, param_value in enumerate(param_values):
        print(f"\n\n{'*'*80}")
        print(f"* PROGRESS: {i+1}/{len(param_values)} - {param_name}={param_value}")
        print(f"{'*'*80}")

        # Determine start_iter and existing_results for extend mode (no-op otherwise).
        start_iter = 0
        existing_results = None
        if args.extend is not None:
            param_str = format_param_for_filename(args.distribution_type, param_value)
            param_dir = os.path.join(dirs['experiments'], f"{param_name}_{param_str}")
            iter_root = os.path.join(param_dir, 'iterations')
            start_iter = count_existing_iterations(iter_root)
            existing_results = [
                read_iter_result(os.path.join(iter_root, f'iter_{k:03d}'),
                                 eval_types_for_read)
                for k in range(1, start_iter + 1)
            ]

        try:
            statistics = run_experiments_for_param(param_value, args.distribution_type, n_iter_to_run,
                                                 dirs['experiments'], args.eval_types,
                                                 args.vocab_size, args.unseen_count,
                                                 base_seed=effective_base_seed, p_idx=i,
                                                 start_iter=start_iter,
                                                 existing_results=existing_results)
            all_statistics[param_value] = statistics

            # Save progress after each successful parameter value (record base_seed
            # so a later --extend can recover the seed derivation automatically).
            save_progress(progress_file, param_value, base_seed=effective_base_seed)

            # Print quick summary for this parameter value
            if statistics:
                print(f"\nQuick summary for {param_name}={param_value}:")
                for eval_type, stats in statistics.items():
                    mean_acc = stats['mean']
                    std_acc = stats['std']
                    print(f"  {eval_type}: {mean_acc:.1f}% ± {std_acc:.1f}%")

            # Update comprehensive summary after each parameter value (for safety)
            # Include all parameter values that were supposed to be run
            all_param_values = generate_param_values(args.param_min, args.param_max, args.param_step)
            create_comprehensive_summary(all_param_values, args.distribution_type, all_statistics, dirs,
                                        args.param_step, comprehensive_n_iterations)

        except Exception as e:  # pragma: no cover - preserves last-good progress.json on uncaught experiment failure; raise propagates
            print(f"ERROR: Failed to run experiments for {param_name}={param_value}: {e}")
            all_statistics[param_value] = None
            # Save progress with error status
            if i > 0:
                # Save the last successfully completed parameter value
                last_completed = param_values[i-1]
                save_progress(progress_file, last_completed, status='error_at_next',
                              base_seed=effective_base_seed)
            raise  # Re-raise to stop execution

    # Create final comprehensive summary with all parameter values
    all_param_values = generate_param_values(args.param_min, args.param_max, args.param_step)
    create_comprehensive_summary(all_param_values, args.distribution_type, all_statistics, dirs,
                                args.param_step, comprehensive_n_iterations)

    # Auto-generate plot at the end of the sweep.
    images_dir = os.path.join(dirs['base'], 'images')
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(dirs['summary'], 'comprehensive_results.csv')
    if args.distribution_type == "oneshot":
        plot_path = os.path.join(images_dir, 'oneshot_accuracy.png')
        plot_oneshot_bars(csv_path, plot_path)
    else:
        plot_path = os.path.join(images_dir, 'accuracy_vs_alpha.png')
        plot_zipfian_sweep(csv_path, plot_path)
    print(f"* Plot saved to: {plot_path}")

    print(f"\n{'*'*80}")
    print(f"* COMPREHENSIVE EXPERIMENTS COMPLETE!")
    print(f"* Total experiments run: {len(param_values) * args.n_iterations}")
    print(f"* Results saved to: {dirs['base']}")
    print(f"{'*'*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_parser(sub)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)