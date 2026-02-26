#!/usr/bin/env python3
"""
Plot variance in accuracy across different experiment configurations.
Shows how the variance (standard deviation squared) changes with concentration parameter C.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json

def plot_variance_results(csv_path, output_path=None, experiment_info=None):
    """Plot variance in accuracy vs C value for different evaluation types"""
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate variance from standard deviation
    df['variance'] = df['std'] ** 2
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for each config type
    config_styles = {
        'seen_match': {'color': '#3182bd', 'marker': 'o', 'label': 'Seen, Match'},
        'seen_mismatch': {'color': '#feb24c', 'marker': 's', 'label': 'Seen, Mismatch'},
        'unseen_match': {'color': '#de2d26', 'marker': '^', 'label': 'Unseen, Match'},
        'unseen_mismatch': {'color': '#756bb1', 'marker': 'v', 'label': 'Unseen, Mismatch'}
    }
    
    # Plot each evaluation type
    for eval_type, style in config_styles.items():
        data = df[df['eval_type'] == eval_type].copy()
        if len(data) == 0:
            continue
            
        # Sort by C value for proper plotting
        data = data.sort_values('c_value')
        
        # Plot the variance
        plt.plot(data['c_value'], data['variance'], 
                color=style['color'], 
                marker=style['marker'], 
                label=style['label'],
                linewidth=2,
                markersize=8)
        
        # Add range indicators (min to max variance per C value if multiple runs)
        # For single runs, this just shows the point
        range_data = data['max'] - data['min']
        plt.scatter(data['c_value'], range_data, 
                   color=style['color'], 
                   marker='x', 
                   s=50, 
                   alpha=0.6,
                   label=f'{style["label"]} Range' if len(data) > 1 else '')
        
        # Note: No trend line for variance plots - just showing raw data
    
    # Customize the plot
    plt.xlabel('Concentration Parameter (C)', fontsize=14)
    plt.ylabel('Variance in Accuracy', fontsize=14)
    
    # Create title with experiment info
    if experiment_info:
        title = f'Variance in Model Accuracy vs Concentration Parameter C\n{experiment_info}'
    else:
        title = 'Variance in Model Accuracy vs Concentration Parameter C'
    plt.title(title, fontsize=16)
    
    plt.grid(True, alpha=0.3)
    
    # Handle legend
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Filter out empty range labels
    filtered_handles = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if l and l != '':
            filtered_handles.append(h)
            filtered_labels.append(l)
    
    plt.legend(filtered_handles, filtered_labels, loc='best', frameon=True, shadow=True)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(bottom=0)  # Variance can't be negative
    
    # Add minor ticks
    plt.xticks([i/10 for i in range(11)])
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Variance plot saved to: {output_path}")
    else:
        plt.show()

def plot_std_vs_mean(csv_path, output_path=None, experiment_info=None):
    """Plot standard deviation vs mean accuracy to show heteroscedasticity"""
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Define colors for each config type
    config_styles = {
        'seen_match': {'color': '#3182bd', 'marker': 'o', 'label': 'Seen, Match'},
        'seen_mismatch': {'color': '#feb24c', 'marker': 's', 'label': 'Seen, Mismatch'},
        'unseen_match': {'color': '#de2d26', 'marker': '^', 'label': 'Unseen, Match'},
        'unseen_mismatch': {'color': '#756bb1', 'marker': 'v', 'label': 'Unseen, Mismatch'}
    }
    
    # Plot each evaluation type
    for eval_type, style in config_styles.items():
        data = df[df['eval_type'] == eval_type]
        if len(data) == 0:
            continue
            
        plt.scatter(data['mean'], data['std'], 
                   color=style['color'], 
                   marker=style['marker'], 
                   label=style['label'],
                   s=60,
                   alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Mean Accuracy (%)', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    
    # Create title
    if experiment_info:
        title = f'Standard Deviation vs Mean Accuracy\n{experiment_info}'
    else:
        title = 'Standard Deviation vs Mean Accuracy'
    plt.title(title, fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Std vs Mean plot saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot variance in experiment results')
    parser.add_argument('experiment_dir', type=str, 
                        help='Path to experiment directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: saves to experiment_dir/images/variance_plot.png)')
    parser.add_argument('--std-vs-mean', action='store_true',
                        help='Plot standard deviation vs mean instead of variance vs C')
    
    args = parser.parse_args()
    
    # Build paths
    experiment_path = Path(args.experiment_dir)
    csv_path = experiment_path / 'summary' / 'comprehensive_results.csv'
    
    # Check if CSV file exists
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Create images directory in experiment folder
        images_dir = experiment_path / 'images'
        images_dir.mkdir(exist_ok=True)
        if args.std_vs_mean:
            output_path = images_dir / 'std_vs_mean_plot.png'
        else:
            output_path = images_dir / 'variance_plot.png'
    
    # Extract experiment info from directory name or config files
    dir_name = experiment_path.name
    experiment_info = ""
    
    # First check if there's a download_summary.json (for parallel sweeps)
    download_summary_path = experiment_path / 'download_summary.json'
    if download_summary_path.exists():
        try:
            with open(download_summary_path, 'r') as f:
                summary = json.load(f)
            
            params = summary['original_sweep_info']['parameters']
            n_value = params['n_iterations']
            step_value = params['c_step']
            timestamp = summary['original_sweep_info']['timestamp']
            
            # Format timestamp
            if 'T' in timestamp:
                formatted_timestamp = timestamp.replace('T', ' ').split('.')[0]
            else:
                formatted_timestamp = timestamp
                
            experiment_info = f"n={n_value}, step={step_value}, Run: {formatted_timestamp}"
        except:
            pass
    
    # If no download_summary.json, try parsing directory name
    if not experiment_info:
        parts = dir_name.split('_')
        
        try:
            # Find n value
            n_idx = next(i for i, part in enumerate(parts) if part.startswith('n'))
            n_value = parts[n_idx][1:]
            
            # Find step size
            step_idx = next(i for i, part in enumerate(parts) if part.startswith('step'))
            step_value = parts[step_idx][4:]
            
            # Find timestamp
            timestamp = f"{parts[-2]}_{parts[-1]}"
            
            # Format timestamp
            if len(timestamp) == 15:
                formatted_timestamp = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
            else:
                formatted_timestamp = timestamp
                
            experiment_info = f"n={n_value}, step={step_value}, Run: {formatted_timestamp}"
        except:
            experiment_info = f"Experiment: {dir_name}"
    
    print(f"Using data from: {csv_path}")
    
    if args.std_vs_mean:
        plot_std_vs_mean(str(csv_path), str(output_path), experiment_info)
    else:
        plot_variance_results(str(csv_path), str(output_path), experiment_info)

if __name__ == "__main__":
    main()