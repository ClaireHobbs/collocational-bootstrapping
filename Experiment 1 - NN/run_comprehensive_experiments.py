#!/usr/bin/env python3
"""
Run comprehensive experiments across multiple parameter values with multiple iterations each.
Supports both geometric (C parameter) and Zipfian (Z parameter) distributions.

Usage: python3 run_comprehensive_experiments.py <param_min> <param_max> <param_step> <n_iterations> [--distribution_type geometric|zipfian]
Examples:
  python3 run_comprehensive_experiments.py 0.0 1.0 0.1 5 --distribution_type geometric
  python3 run_comprehensive_experiments.py 0.0 2.0 0.5 3 --distribution_type zipfian
"""

import os
import sys
import argparse
import subprocess
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from distribution_interface import (
    format_param_for_filename, get_parameter_name, validate_parameter,
    parse_experiment_directory
)

def run_command(cmd, description, show_output=True):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    if show_output:
        # Stream output in real-time
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print to console in real-time
            output_lines.append(line)
        
        process.wait()
        
        if process.returncode != 0:
            print(f"ERROR: {description} failed with return code {process.returncode}")
            sys.exit(1)
        
        return ''.join(output_lines)
    else:
        # Capture output without showing
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: {description} failed!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            sys.exit(1)
        
        return result.stdout

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
        # Use provided experiment name
        base_dir = f"comprehensive_experiments/{experiment_name}"
    else:
        # Create new directory
        param_name = get_parameter_name(distribution_type)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"comprehensive_experiments/{param_name}_{param_min}-{param_max}_step{param_step}_n{n_iterations}_{timestamp}"

    # Create subdirectories
    dirs = {
        'base': base_dir,
        'experiments': os.path.join(base_dir, 'experiments'),
        'summary': os.path.join(base_dir, 'summary')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def run_experiments_for_param(param_value, distribution_type, n_iterations, experiments_dir, reuse_model=False, eval_types=None, vocab_size=40, unseen_count=10):
    """Run n iterations for a single parameter value directly in the comprehensive experiments directory."""
    param_name = get_parameter_name(distribution_type)
    print(f"\n\n{'#'*80}")
    print(f"# RUNNING EXPERIMENTS FOR {distribution_type.upper()} DISTRIBUTION: {param_name}={param_value}")
    print(f"# {n_iterations} iterations")
    if reuse_model:
        print(f"# (Using shared model)")
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

    # Handle shared model if requested
    shared_model_dir = None
    if reuse_model:
        shared_model_dir = os.path.join(param_experiment_dir, 'shared_model')
        os.makedirs(shared_model_dir, exist_ok=True)

        # Train shared model first
        temp_data_dir = os.path.join(shared_model_dir, 'temp_data')
        os.makedirs(temp_data_dir, exist_ok=True)

        print(f">>> Training shared model for {param_name}={param_value}...")

        # Generate dataset for shared model
        cmd = (f"python3 generate_entropy_dataset.py {param_value} "
               f"--distribution_type {distribution_type} "
               f"--output_dir {temp_data_dir} "
               f"--train_file train.txt "
               f"--val_file val.txt "
               f"--test_file test.txt "
               f"--vocab_size {vocab_size} "
               f"--unseen_count {unseen_count}")
        run_command(cmd, "Shared dataset generation", show_output=False)

        # Train shared model
        shared_model_path = os.path.join(shared_model_dir, 'model')
        cmd = (f"python3 train_and_eval.py --saved_model_name shared_model "
               f"--data_dir {temp_data_dir} --model_save_dir {shared_model_path}")
        run_command(cmd, "Shared model training", show_output=True)

        # Clean up temp data
        import shutil
        shutil.rmtree(temp_data_dir)

    # Run iterations
    all_results = []
    for i in range(n_iterations):
        print(f"\n>>> Running iteration {i+1}/{n_iterations} for {param_name}={param_value}...")

        iteration_dir = os.path.join(iterations_dir, f'iter_{i+1:03d}')
        result = run_single_iteration(param_value, distribution_type, i, iteration_dir, shared_model_dir, eval_types, vocab_size, unseen_count)
        all_results.append(result)

        # Show quick result
        result_str = f">>> Iteration {i+1} results: "
        result_parts = []
        for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
            if eval_type in result:
                result_parts.append(f"{eval_type}={result[eval_type]:.1f}%")
        print(result_str + ", ".join(result_parts))

    # Calculate statistics and save summary
    statistics = calculate_statistics(all_results)
    save_experiment_summary(param_value, distribution_type, n_iterations, all_results, statistics,
                           summary_dir, reuse_model)

    print(f">>> Completed all experiments for {param_name}={param_value}")
    return statistics

def run_single_iteration(param_value, distribution_type, iteration, iteration_dir, shared_model_dir=None, eval_types=None, vocab_size=40, unseen_count=10):
    """Run a single experiment iteration.

    Args:
        param_value: Distribution parameter value (C for geometric, Z for Zipfian)
        distribution_type: "geometric" or "zipfian"
        iteration: Iteration number (0-based)
        iteration_dir: Directory to store iteration results
        shared_model_dir: Directory containing shared model (if reusing)
        eval_types: List of evaluation types to run. If None, runs all.
                   Options: ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']
    """
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
    
    # Create directories, but skip model directory if using shared model
    for key, dir_path in dirs.items():
        if key == 'model' and shared_model_dir:
            continue  # Will copy shared model instead
        os.makedirs(dir_path, exist_ok=True)
    
    # Step 1: Generate dataset
    cmd = (f"python3 generate_entropy_dataset.py {param_value} "
           f"--distribution_type {distribution_type} "
           f"--output_dir {dirs['data']} "
           f"--train_file train.txt "
           f"--val_file val.txt "
           f"--test_file test.txt "
           f"--vocab_size {vocab_size} "
           f"--unseen_count {unseen_count}")
    run_command(cmd, f"Dataset generation for iteration {iteration+1}", show_output=False)
    
    # Step 2: Setup model (train new or copy shared)
    if shared_model_dir:
        # Copy shared model to this iteration's directory
        import shutil
        shutil.copytree(shared_model_dir, dirs['model'])
    else:
        # Train new model
        cmd = (f"python3 train_and_eval.py --saved_model_name iter_{iteration+1} "
               f"--data_dir {dirs['data']} --model_save_dir {dirs['model']}")
        run_command(cmd, f"Model training for iteration {iteration+1}", show_output=True)
    
    # Step 3: Generate minimal pairs (only for requested eval types)
    # Use uniform distribution for unseen pairs: geometric C=1.0, zipfian Z=0.0
    unseen_param = 1.0 if distribution_type == 'geometric' else 0.0

    test_configs = {
        'seen_match': (param_value, False, False),      # seen, match
        'seen_mismatch': (param_value, False, True),    # seen, mismatch
        'unseen_match': (unseen_param, True, False),    # unseen, match
        'unseen_mismatch': (unseen_param, True, True)   # unseen, mismatch
    }

    for pair_type in eval_types:
        if pair_type not in test_configs:
            print(f"WARNING: Unknown eval type '{pair_type}', skipping")
            continue

        param_eval, noun_OOD, prep_obj_mismatch = test_configs[pair_type]
        cmd = (f"python3 generate_entropy_minimal_pairs.py {param_eval} "
               f"--distribution_type {distribution_type} "
               f"--output_dir {dirs['eval']} "
               f"--output_file minimal_pairs_{pair_type}.txt "
               f"--vocab_size {vocab_size} "
               f"--unseen_count {unseen_count}")
        if noun_OOD:
            cmd += " --noun_OOD"
        if prep_obj_mismatch:
            cmd += " --prep_obj_mismatch"

        run_command(cmd, f"Generating {pair_type} pairs for iteration {iteration+1}", show_output=False)
    
    # Step 4: Evaluate model (only for requested eval types)
    results = {}
    for pair_type in eval_types:
        if pair_type not in test_configs:
            continue
            
        input_file = os.path.join(dirs['eval'], f"minimal_pairs_{pair_type}.txt")
        output_file = os.path.join(dirs['results'], f"accuracy_{pair_type}.txt")
        
        cmd = (f"python3 logprob_and_accuracy.py "
               f"--model_dir {dirs['model']} "
               f"--input_file {input_file} "
               f"--output_file {output_file}")
        
        run_command(cmd, f"Evaluating {pair_type} for iteration {iteration+1}", show_output=False)
        
        # Extract accuracy from output file
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Model Accuracy:" in line:
                    accuracy_str = line.split(":")[1].strip()
                    accuracy = float(accuracy_str.split("%")[0])
                    results[pair_type] = accuracy
                    break
    
    return results

def calculate_statistics(all_results):
    """Calculate mean, stddev, min, max for each evaluation type."""
    stats = {}
    
    # Get all eval types that appear in any result
    all_eval_types = set()
    for result in all_results:
        all_eval_types.update(result.keys())
    
    for eval_type in all_eval_types:
        values = [results[eval_type] for results in all_results if eval_type in results]
        if values:  # Only calculate stats if we have data
            stats[eval_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    return stats

def save_experiment_summary(param_value, distribution_type, n_iterations, all_results, stats, summary_dir, reuse_model):
    """Save summary for this parameter value's experiments."""

    # Create JSON summary
    json_file = os.path.join(summary_dir, 'summary.json')
    param_name = get_parameter_name(distribution_type)
    json_data = {
        'param_value': param_value,
        'distribution_type': distribution_type,
        'param_name': param_name,
        'n_iterations': n_iterations,
        'reuse_model': reuse_model,
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            eval_type: {
                'mean': s['mean'],
                'std': s['std'],
                'min': s['min'],
                'max': s['max']
            }
            for eval_type, s in stats.items()
        },
        'raw_results': [
            {
                'iteration': i + 1,
                'results': results
            }
            for i, results in enumerate(all_results)
        ]
    }

    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

def create_comprehensive_summary(param_values, distribution_type, all_statistics, dirs, param_step, n_iterations, reuse_model):
    """Create comprehensive summary across all parameter values."""

    param_name = get_parameter_name(distribution_type)

    # Create text summary
    summary_file = os.path.join(dirs['summary'], 'comprehensive_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Comprehensive Experiment Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Distribution type: {distribution_type}\n")
        f.write(f"{param_name} value range: {min(param_values):.3f} to {max(param_values):.3f}\n")
        f.write(f"{param_name} step size: {param_step}\n")
        f.write(f"Iterations per {param_name} value: {n_iterations}\n")
        f.write(f"Model reuse: {'Yes (shared model per param)' if reuse_model else 'No (separate models)'}\n")
        f.write(f"Total experiments run: {len(param_values) * n_iterations}\n")
        f.write(f"Experiment directory: {dirs['base']}\n\n")

        # Summary table by evaluation type
        # Get all eval types that were actually run
        all_eval_types = set()
        for param_stats in all_statistics.values():
            if param_stats:
                all_eval_types.update(param_stats.keys())

        # Sort for consistent ordering
        eval_types = sorted(all_eval_types)

        for eval_type in eval_types:
            f.write(f"\n{eval_type.upper()} Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{f'{param_name} Value':<10} {'Mean':>12} {'StdDev':>12} {'Min':>12} {'Max':>12}\n")
            f.write("-" * 80 + "\n")

            for param_value in param_values:
                if param_value in all_statistics and all_statistics[param_value] and eval_type in all_statistics[param_value]:
                    stats = all_statistics[param_value][eval_type]
                    f.write(f"{param_value:<10.3f} {stats['mean']:>11.2f}% {stats['std']:>11.2f}% "
                           f"{stats['min']:>11.2f}% {stats['max']:>11.2f}%\n")
                else:
                    f.write(f"{param_value:<10.3f} {'No data':>48}\n")

    # Create comprehensive JSON
    json_file = os.path.join(dirs['summary'], 'comprehensive_summary.json')
    json_data = {
        'param_values': param_values,
        'distribution_type': distribution_type,
        'param_name': param_name,
        'param_step': param_step,
        'n_iterations': n_iterations,
        'reuse_model': reuse_model,
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(param_values) * n_iterations,
        'statistics_by_param': all_statistics
    }

    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Create CSV for easy analysis - use generic column name for compatibility with plotting
    csv_file = os.path.join(dirs['summary'], 'comprehensive_results.csv')
    with open(csv_file, 'w') as f:
        # Header - keep c_value for backward compatibility with existing plotting scripts
        f.write("c_value,eval_type,mean,std,min,max\n")

        # Data rows
        for param_value in param_values:
            if param_value in all_statistics and all_statistics[param_value]:
                for eval_type, stats in all_statistics[param_value].items():
                    f.write(f"{param_value},{eval_type},{stats['mean']:.3f},{stats['std']:.3f},"
                           f"{stats['min']:.3f},{stats['max']:.3f}\n")

    print(f"\n{'='*80}")
    print(f"Comprehensive summary saved to:")
    print(f"  Text: {summary_file}")
    print(f"  JSON: {json_file}")
    print(f"  CSV:  {csv_file}")
    print(f"{'='*80}")

    # Print summary to console
    with open(summary_file, 'r') as f:
        print(f.read())

def save_progress(progress_file, param_value, status='completed'):
    """Save progress to a file for resuming later."""
    progress_data = {
        'last_completed_param': param_value,
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_progress(progress_file):
    """Load progress from file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive experiments across multiple parameter values with multiple iterations each')
    parser.add_argument('param_min', type=float,
                       help='Minimum parameter value (e.g., 0.0 for geometric, 0.0 for zipfian)')
    parser.add_argument('param_max', type=float,
                       help='Maximum parameter value (e.g., 1.0 for geometric, 2.0 for zipfian)')
    parser.add_argument('param_step', type=float,
                       help='Step size for parameter values (e.g., 0.1)')
    parser.add_argument('n_iterations', type=int,
                       help='Number of iterations per parameter value (e.g., 5)')
    parser.add_argument('--distribution_type', type=str, default='geometric',
                       choices=['geometric', 'zipfian', 'oneshot'],
                       help='Distribution type: geometric (default), zipfian, or oneshot')
    parser.add_argument('--reuse-model', action='store_true',
                       help='Train model once per parameter value and reuse for all iterations')
    parser.add_argument('--resume', type=str, metavar='DIR',
                       help='Resume from a previous run using the specified experiment directory')
    parser.add_argument('--eval-types', type=str, nargs='+',
                       choices=['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch'],
                       help='Specific evaluation types to run (default: all)')
    parser.add_argument('--experiment-name', type=str,
                       help='Experiment directory name (overrides timestamp generation)')
    parser.add_argument('--vocab-size', type=int, default=40,
                       help='Vocabulary size (default: 40)')
    parser.add_argument('--unseen-count', type=int, default=10,
                       help='Number of unseen pairs (default: 10)')

    args = parser.parse_args()

    # Validate arguments based on distribution type
    try:
        validate_parameter(args.distribution_type, args.param_min)
        validate_parameter(args.distribution_type, args.param_max)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.param_min > args.param_max:
        print("Error: param_min must be <= param_max")
        sys.exit(1)

    if args.param_step <= 0:
        print("Error: param_step must be positive")
        sys.exit(1)

    if args.n_iterations < 1:
        print("Error: n_iterations must be at least 1")
        sys.exit(1)

    # Generate parameter values
    param_values = generate_param_values(args.param_min, args.param_max, args.param_step)
    
    # Handle resuming from previous run
    resume_from_param = None
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Resume directory '{args.resume}' does not exist")
            sys.exit(1)

        # Load progress file
        progress_file = os.path.join(args.resume, 'progress.json')
        progress_data = load_progress(progress_file)

        if progress_data:
            # Handle both old and new progress file formats
            last_param = progress_data.get('last_completed_param') or progress_data.get('last_completed_c')
            resume_from_param = last_param
            param_name = get_parameter_name(args.distribution_type)
            print(f"\n*** RESUMING from after {param_name}={last_param} ***")
            print(f"Last completed at: {progress_data['timestamp']}")

            # Filter parameter values to only those after the last completed
            param_values = [p for p in param_values if p > last_param]

            if not param_values:
                print(f"\nAll {param_name} values were already completed!")
                sys.exit(0)
        else:
            print(f"\nWarning: No progress file found in '{args.resume}', starting from beginning")

    param_name = get_parameter_name(args.distribution_type)
    print(f"\n{'*'*80}")
    print(f"* COMPREHENSIVE EXPERIMENT PLAN")
    if resume_from_param is not None:
        print(f"* RESUMING after {param_name}={resume_from_param}")
    print(f"* Distribution type: {args.distribution_type}")
    print(f"* {param_name} values to process: {param_values}")
    print(f"* Iterations per {param_name}: {args.n_iterations}")
    print(f"* Total experiments: {len(param_values) * args.n_iterations}")
    print(f"* Model reuse: {'Yes' if args.reuse_model else 'No'}")
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
    if args.resume:
        summary_json = os.path.join(dirs['summary'], 'comprehensive_summary.json')
        if os.path.exists(summary_json):
            with open(summary_json, 'r') as f:
                existing_data = json.load(f)
                # Handle both old and new JSON formats
                stats_key = 'statistics_by_param' if 'statistics_by_param' in existing_data else 'statistics_by_c'
                if stats_key in existing_data:
                    # Convert string keys back to float
                    all_statistics = {float(k): v for k, v in existing_data[stats_key].items()}

    for i, param_value in enumerate(param_values):
        print(f"\n\n{'*'*80}")
        print(f"* PROGRESS: {i+1}/{len(param_values)} - {param_name}={param_value}")
        print(f"{'*'*80}")

        try:
            statistics = run_experiments_for_param(param_value, args.distribution_type, args.n_iterations,
                                                 dirs['experiments'], args.reuse_model, args.eval_types,
                                                 args.vocab_size, args.unseen_count)
            all_statistics[param_value] = statistics

            # Save progress after each successful parameter value
            save_progress(progress_file, param_value)

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
                                        args.param_step, args.n_iterations, args.reuse_model)

        except Exception as e:
            print(f"ERROR: Failed to run experiments for {param_name}={param_value}: {e}")
            all_statistics[param_value] = None
            # Save progress with error status
            if i > 0:
                # Save the last successfully completed parameter value
                last_completed = param_values[i-1]
                save_progress(progress_file, last_completed, status='error_at_next')
            raise  # Re-raise to stop execution

    # Create final comprehensive summary with all parameter values
    all_param_values = generate_param_values(args.param_min, args.param_max, args.param_step)
    create_comprehensive_summary(all_param_values, args.distribution_type, all_statistics, dirs,
                                args.param_step, args.n_iterations, args.reuse_model)

    print(f"\n{'*'*80}")
    print(f"* COMPREHENSIVE EXPERIMENTS COMPLETE!")
    print(f"* Total experiments run: {len(param_values) * args.n_iterations}")
    print(f"* Results saved to: {dirs['base']}")
    print(f"{'*'*80}")

if __name__ == "__main__":
    main()