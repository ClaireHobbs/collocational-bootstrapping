#!/usr/bin/env python3
"""
Run a complete experiment for a single parameter value.
Usage: python3 run_single_C_experiment.py <param_value> [--distribution_type geometric|zipfian]
Example: python3 run_single_C_experiment.py 0.5 --distribution_type geometric
Example: python3 run_single_C_experiment.py 1.5 --distribution_type zipfian
"""

import os
import sys
import argparse
import subprocess
import shutil
from datetime import datetime
from distribution_interface import format_param_for_filename, get_parameter_name, validate_parameter

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    return result.stdout

def create_experiment_directory(param_value, distribution_type):
    """Create directory structure for experiment."""
    validate_parameter(distribution_type, param_value)

    param_str = format_param_for_filename(distribution_type, param_value)
    param_name = get_parameter_name(distribution_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{param_name}_{param_str}_{timestamp}"

    # Create subdirectories
    dirs = {
        'base': base_dir,
        'data': os.path.join(base_dir, 'data'),
        'model': os.path.join(base_dir, 'model'),
        'eval': os.path.join(base_dir, 'eval'),
        'results': os.path.join(base_dir, 'results')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def generate_dataset(param_value, distribution_type, data_dir):
    """Generate training, validation, and test datasets."""
    param_name = get_parameter_name(distribution_type)
    print(f"\n\n{'#'*60}")
    print(f"# Step 1: Generating dataset with {distribution_type} distribution: {param_name}={param_value}")
    print(f"{'#'*60}")

    cmd = (f"python3 generate_entropy_dataset.py {param_value} "
           f"--distribution_type {distribution_type} "
           f"--output_dir {data_dir} "
           f"--train_file train.txt "
           f"--val_file val.txt "
           f"--test_file test.txt")

    run_command(cmd, "Dataset generation")

def train_model(param_value, distribution_type, data_dir, model_dir):
    """Train the model."""
    param_name = get_parameter_name(distribution_type)
    print(f"\n\n{'#'*60}")
    print(f"# Step 2: Training model with {distribution_type} distribution: {param_name}={param_value}")
    print(f"{'#'*60}")

    param_str = format_param_for_filename(distribution_type, param_value)
    model_name = f"model_{param_name}_{param_str}"

    # Run training with data directory and save directly to experiment directory
    cmd = (f"python3 train_and_eval.py --saved_model_name {model_name} "
           f"--data_dir {data_dir} --model_save_dir {model_dir}")
    run_command(cmd, "Model training")

    return model_name

def generate_minimal_pairs(param_value, distribution_type, eval_dir, noun_OOD, prep_obj_mismatch, pair_type):
    """Generate minimal pairs for evaluation."""

    print(f"\n  Generating minimal pairs: {pair_type}")

    cmd = (f"python3 generate_entropy_minimal_pairs.py {param_value} "
           f"--distribution_type {distribution_type} "
           f"--output_dir {eval_dir} "
           f"--output_file minimal_pairs_{pair_type}.txt")

    # Add flags based on parameters
    if noun_OOD:
        cmd += " --noun_OOD"
    if prep_obj_mismatch:
        cmd += " --prep_obj_mismatch"

    run_command(cmd, f"Generating minimal pairs: {pair_type}")

def evaluate_model(model_dir, eval_dir, results_dir, pair_type):
    """Evaluate model on minimal pairs."""
    
    print(f"\n  Evaluating: {pair_type}")
    
    input_file = os.path.join(eval_dir, f"minimal_pairs_{pair_type}.txt")
    output_file = os.path.join(results_dir, f"accuracy_{pair_type}.txt")
    
    cmd = (f"python3 logprob_and_accuracy.py "
           f"--model_dir {model_dir} "
           f"--input_file {input_file} "
           f"--output_file {output_file}")
    
    run_command(cmd, f"Evaluating: {pair_type}")

def create_summary_report(param_value, distribution_type, dirs):
    """Create a summary report of all results."""
    summary_file = os.path.join(dirs['base'], 'summary.txt')
    param_name = get_parameter_name(distribution_type)

    with open(summary_file, 'w') as f:
        f.write(f"Experiment Summary for {distribution_type} distribution: {param_name}={param_value}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Distribution type: {distribution_type}\n")
        f.write(f"Parameter: {param_name}={param_value}\n")
        f.write(f"Experiment directory: {dirs['base']}\n\n")

        f.write("Evaluation Results:\n")
        f.write("-" * 40 + "\n")

        # Read accuracy from each evaluation file
        for eval_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
            accuracy_file = os.path.join(dirs['results'], f'accuracy_{eval_type}.txt')
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r') as af:
                    lines = af.readlines()
                    for line in lines:
                        if "Model Accuracy:" in line:
                            f.write(f"{eval_type}: {line.strip()}\n")

    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")

    # Also print summary to console
    with open(summary_file, 'r') as f:
        print(f.read())

def main():
    parser = argparse.ArgumentParser(description='Run a complete experiment for a single parameter value')
    parser.add_argument('param_value', type=float,
                       help='Distribution parameter value (C for geometric 0-1, Z for Zipfian ≥0, ignored for oneshot)')
    parser.add_argument('--distribution_type', type=str, default='geometric',
                       choices=['geometric', 'zipfian', 'oneshot'],
                       help='Distribution type: geometric (default), zipfian, or oneshot')
    args = parser.parse_args()

    param_value = args.param_value
    distribution_type = args.distribution_type

    # Validate parameter for the distribution type
    try:
        validate_parameter(distribution_type, param_value)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    param_name = get_parameter_name(distribution_type)
    print(f"\n{'*'*60}")
    print(f"* Running experiment for {distribution_type} distribution: {param_name}={param_value}")
    print(f"{'*'*60}")

    # Create experiment directory structure
    dirs = create_experiment_directory(param_value, distribution_type)
    print(f"\nExperiment directory: {dirs['base']}")

    # Step 1: Generate dataset
    generate_dataset(param_value, distribution_type, dirs['data'])

    # Step 2: Train model
    model_name = train_model(param_value, distribution_type, dirs['data'], dirs['model'])

    # Step 3: Generate minimal pairs for all 4 evaluation scenarios
    print(f"\n\n{'#'*60}")
    print(f"# Step 3: Generating minimal pairs for evaluation")
    print(f"{'#'*60}")

    # Test 1: Seen pairings, objects match subject (param_eval = param_train)
    generate_minimal_pairs(param_value, distribution_type, dirs['eval'], noun_OOD=False,
                          prep_obj_mismatch=False, pair_type='seen_match')

    # Test 2: Seen pairings, objects mismatch subject (param_eval = param_train)
    generate_minimal_pairs(param_value, distribution_type, dirs['eval'], noun_OOD=False,
                          prep_obj_mismatch=True, pair_type='seen_mismatch')

    # Test 3: Unseen pairings, objects match subject (use uniform: geometric C=1.0, zipfian Z=0.0)
    unseen_param = 1.0 if distribution_type == 'geometric' else 0.0
    generate_minimal_pairs(unseen_param, distribution_type, dirs['eval'], noun_OOD=True,
                          prep_obj_mismatch=False, pair_type='unseen_match')

    # Test 4: Unseen pairings, objects mismatch subject (use uniform: geometric C=1.0, zipfian Z=0.0)
    generate_minimal_pairs(unseen_param, distribution_type, dirs['eval'], noun_OOD=True,
                          prep_obj_mismatch=True, pair_type='unseen_mismatch')

    # Step 4: Evaluate model on all minimal pairs
    print(f"\n\n{'#'*60}")
    print(f"# Step 4: Evaluating model on all minimal pairs")
    print(f"{'#'*60}")

    for pair_type in ['seen_match', 'seen_mismatch', 'unseen_match', 'unseen_mismatch']:
        evaluate_model(dirs['model'], dirs['eval'], dirs['results'], pair_type)

    # Create summary report
    create_summary_report(param_value, distribution_type, dirs)

    print(f"\n{'*'*60}")
    print(f"* Experiment complete!")
    print(f"* Results saved to: {dirs['base']}")
    print(f"{'*'*60}")

if __name__ == "__main__":
    main()