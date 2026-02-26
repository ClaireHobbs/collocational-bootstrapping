#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import argparse
from pathlib import Path
from distribution_interface import parse_experiment_directory, get_parameter_name

def plot_accuracy_results(csv_path, output_path=None, experiment_info=None, show_trend=False, distribution_type=None):
    """Plot mean accuracy vs parameter value for different evaluation types"""
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
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
        
        # Plot the actual data points
        plt.plot(data['c_value'], data['mean'], 
                color=style['color'], 
                marker=style['marker'], 
                label=style['label'],
                linewidth=2,
                markersize=8)
        
        # Add error bars for standard deviation (clamped to 0-100 range)
        lower_bound = np.maximum(data['mean'] - data['std'], 0)
        upper_bound = np.minimum(data['mean'] + data['std'], 100)
        plt.fill_between(data['c_value'],
                        lower_bound,
                        upper_bound,
                        alpha=0.2,
                        color=style['color'])
        
        # Add min/max indicators
        plt.scatter(data['c_value'], data['min'], 
                   color=style['color'], 
                   marker='_', 
                   s=100, 
                   alpha=0.7)
        plt.scatter(data['c_value'], data['max'], 
                   color=style['color'], 
                   marker='_', 
                   s=100, 
                   alpha=0.7)
        
        # Connect min and max with thin lines
        for i in range(len(data)):
            plt.plot([data.iloc[i]['c_value'], data.iloc[i]['c_value']], 
                    [data.iloc[i]['min'], data.iloc[i]['max']], 
                    color=style['color'], 
                    linewidth=0.5, 
                    alpha=0.5)
        
        # Add best fit trend line (optional)
        if show_trend and len(data) >= 3:  # Need at least 3 points for meaningful fit
            x = data['c_value'].values
            y = data['mean'].values
            
            # Try polynomial fit (degree 2 or 3 usually works well)
            # We'll try degree 2 first, can adjust if needed
            degree = 2
            
            # Fit polynomial
            coefficients = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coefficients)
            
            # Generate smooth curve for plotting
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = polynomial(x_smooth)
            
            # Plot trend line
            plt.plot(x_smooth, y_smooth, 
                    color=style['color'], 
                    linestyle='--', 
                    linewidth=1.5, 
                    alpha=0.7,
                    label=f'{style["label"]} trend')
            
            # Calculate R-squared
            y_pred = polynomial(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Customize the plot
    # Auto-detect parameter name from distribution type or use generic label
    if distribution_type:
        param_name = get_parameter_name(distribution_type)
        x_label = f'{param_name} Value'
        if distribution_type == 'geometric':
            title_param = f'Concentration Parameter {param_name}'
        else:
            title_param = f'{distribution_type.title()} Parameter {param_name}'
    else:
        x_label = 'Parameter Value'
        title_param = 'Parameter'

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Mean Accuracy (%)', fontsize=12)

    # Create title with experiment info
    if experiment_info:
        title = f'Model Accuracy vs {title_param}\n{experiment_info}'
    else:
        title = f'Model Accuracy vs {title_param}'
    plt.title(title, fontsize=14)
    
    plt.grid(True, alpha=0.3)
    
    # Customize legend to separate data and trend lines
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Separate data lines from trend lines
    data_handles = []
    data_labels = []
    trend_handles = []
    trend_labels = []
    
    for h, l in zip(handles, labels):
        if 'trend' in l:
            trend_handles.append(h)
            trend_labels.append(l)
        else:
            data_handles.append(h)
            data_labels.append(l)
    
    # Create legend with two columns if we have trend lines
    if trend_handles:
        # Combine handles and labels with a separator
        all_handles = data_handles + [plt.Line2D([0], [0], color='none')] + trend_handles
        all_labels = data_labels + [''] + trend_labels
        plt.legend(all_handles, all_labels, loc='best', frameon=True, shadow=True, ncol=1)
    else:
        plt.legend(data_handles, data_labels, loc='best', frameon=True, shadow=True)
    
    # Set dynamic x-axis limits based on actual data range
    x_min = df['c_value'].min()
    x_max = df['c_value'].max()
    x_range = x_max - x_min
    padding = max(0.05, x_range * 0.05)  # At least 0.05 or 5% of range
    plt.xlim(x_min - padding, x_max + padding)
    plt.ylim(0, 105)

    # Add x-axis ticks at every 0.1 increment
    tick_values = np.arange(0, x_max + 0.1, 0.1)
    tick_values = np.round(tick_values, 1)  # Avoid floating point artifacts
    plt.xticks(tick_values, fontsize=8, rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot comprehensive experiment results')
    parser.add_argument('experiment_dir', type=str,
                        help='Path to experiment directory (e.g., comprehensive_experiments/C_0.0-1.0_step0.1_n5_20250705_001604)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: saves to experiment_dir/images/accuracy_vs_param_plot.png)')
    parser.add_argument('--show-trend', action='store_true',
                        help='Show trend lines on the plot (disabled by default)')
    parser.add_argument('--distribution_type', type=str, choices=['geometric', 'zipfian', 'oneshot'],
                        help='Distribution type (auto-detected if not specified)')

    args = parser.parse_args()
    
    # Build paths
    experiment_path = Path(args.experiment_dir)
    csv_path = experiment_path / 'summary' / 'comprehensive_results.csv'

    # Check if CSV file exists
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Auto-detect distribution type from directory name or JSON metadata
    distribution_type = args.distribution_type
    if not distribution_type:
        # Try to parse from directory name
        dir_name = experiment_path.name
        detected_type, _ = parse_experiment_directory(dir_name)
        if detected_type:
            distribution_type = detected_type
        else:
            # Try to read from JSON metadata
            json_path = experiment_path / 'summary' / 'comprehensive_summary.json'
            if json_path.exists():
                try:
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        distribution_type = data.get('distribution_type', 'geometric')
                except:
                    distribution_type = 'geometric'  # Default fallback
            else:
                distribution_type = 'geometric'  # Default fallback

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Create images directory in experiment folder
        images_dir = experiment_path / 'images'
        images_dir.mkdir(exist_ok=True)
        param_name = get_parameter_name(distribution_type) if distribution_type else 'param'
        output_path = images_dir / f'accuracy_vs_{param_name.lower()}_plot.png'
    
    # Extract experiment info from directory name or config files
    dir_name = experiment_path.name
    experiment_info = ""
    
    # First check if there's a download_summary.json (for parallel sweeps)
    download_summary_path = experiment_path / 'download_summary.json'
    if download_summary_path.exists():
        try:
            import json
            with open(download_summary_path, 'r') as f:
                summary = json.load(f)
            
            params = summary['original_sweep_info']['parameters']
            n_value = params['n_iterations']
            step_value = params['c_step']
            vocab_size = params.get('vocab_size', 40)  # Default to 40 if not found
            unseen_count = params.get('unseen_count', 10)  # Default to 10 if not found
            timestamp = summary['original_sweep_info']['timestamp']
            
            # Format timestamp to be more readable
            # 2025-07-05T20:22:44.458333 -> 2025-07-05 20:22:44
            if 'T' in timestamp:
                formatted_timestamp = timestamp.replace('T', ' ').split('.')[0]
            else:
                formatted_timestamp = timestamp
                
            experiment_info = f"n={n_value}, step={step_value}, vocab={vocab_size}, unseen={unseen_count}, Run: {formatted_timestamp}"
        except:
            pass
    
    # If no download_summary.json, try to find metadata.json in the experiment directory
    if not experiment_info:
        metadata_path = experiment_path / "metadata.json"
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                params = metadata.get('parameters', {})
                n_value = params.get('n_iterations', 'unknown')
                step_value = params.get('c_step', 'unknown')
                vocab_size = params.get('vocab_size', 40)
                unseen_count = params.get('unseen_count', 10)
                timestamp = metadata.get('timestamp', 'unknown')
                
                if 'T' in str(timestamp):
                    formatted_timestamp = str(timestamp).replace('T', ' ').split('.')[0]
                else:
                    formatted_timestamp = str(timestamp)
                
                experiment_info = f"n={n_value}, step={step_value}, vocab={vocab_size}, unseen={unseen_count}, Run: {formatted_timestamp}"
            except:
                pass
    
    # If no metadata files found, try parsing directory name
    if not experiment_info:
        # Expected format: C_0.0-1.0_step0.1_n5_20250705_001604
        parts = dir_name.split('_')
        
        try:
            # Find n value
            n_idx = next(i for i, part in enumerate(parts) if part.startswith('n'))
            n_value = parts[n_idx][1:]  # Remove 'n' prefix
            
            # Find step size
            step_idx = next(i for i, part in enumerate(parts) if part.startswith('step'))
            step_value = parts[step_idx][4:]  # Remove 'step' prefix
            
            # Find timestamp (last two parts joined)
            timestamp = f"{parts[-2]}_{parts[-1]}"
            
            # Format timestamp to be more readable
            # 20250705_001604 -> 2025-07-05 00:16:04
            if len(timestamp) == 15:  # YYYYMMDD_HHMMSS
                formatted_timestamp = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
            else:
                formatted_timestamp = timestamp
                
            experiment_info = f"n={n_value}, step={step_value}, vocab=40, unseen=10, Run: {formatted_timestamp}"
        except:
            # If all parsing fails, just use the directory name with default vocab info
            experiment_info = f"vocab=40, unseen=10, Experiment: {dir_name}"
    
    print(f"Using data from: {csv_path}")
    print(f"Detected distribution type: {distribution_type}")
    plot_accuracy_results(str(csv_path), str(output_path), experiment_info, show_trend=args.show_trend, distribution_type=distribution_type)

if __name__ == "__main__":
    main()