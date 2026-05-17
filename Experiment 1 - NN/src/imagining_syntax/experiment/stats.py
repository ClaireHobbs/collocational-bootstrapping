"""
Statistics and summary functions for comprehensive experiments.

Functions previously inlined in imagining_syntax.runners.sweep:
  - calculate_statistics
  - save_experiment_summary
  - create_comprehensive_summary
"""

import os
import json
import numpy as np
from datetime import datetime
from imagining_syntax.data.distributions import get_parameter_name


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

def save_experiment_summary(param_value, distribution_type, n_iterations, all_results, stats, summary_dir):
    """Save summary for this parameter value's experiments."""

    # Create JSON summary
    json_file = os.path.join(summary_dir, 'summary.json')
    param_name = get_parameter_name(distribution_type)
    json_data = {
        'param_value': param_value,
        'distribution_type': distribution_type,
        'param_name': param_name,
        'n_iterations': n_iterations,
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

def create_comprehensive_summary(param_values, distribution_type, all_statistics, dirs, param_step, n_iterations):
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
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(param_values) * n_iterations,
        'statistics_by_param': all_statistics
    }

    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Create CSV for easy analysis - use generic column name for compatibility with plotting
    csv_file = os.path.join(dirs['summary'], 'comprehensive_results.csv')
    with open(csv_file, 'w') as f:
        f.write("param_value,eval_type,mean,std,min,max\n")

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
