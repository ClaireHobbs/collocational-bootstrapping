#!/usr/bin/env python3
"""
Plot training and validation loss curves from comprehensive experiments.

Usage:
    python plot_losses.py <experiment_dir>
    python plot_losses.py comprehensive_experiments/Z_0.0-3.0_step0.1_n10_20260205_143022

This script reads losses.json files from each iteration and creates visualizations of:
1. Loss curves across training steps for selected Z values
2. Final loss vs Z value
3. Best checkpoint step vs Z value
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def parse_z_from_dirname(dirname_suffix):
    """Reverse the format_param_for_filename convention for zipfian Z dirs.

    Single-digit names are integer Z values (e.g. '0'->0.0, '1'->1.0).
    Multi-digit names encode two decimal places with leading zeros stripped
    (e.g. '10'->0.10, '140'->1.40, '290'->2.90).
    """
    if len(dirname_suffix) == 1:
        return float(int(dirname_suffix))
    int_part = dirname_suffix[:-2] if len(dirname_suffix) > 2 else "0"
    dec_part = dirname_suffix[-2:]
    return float(f"{int_part}.{dec_part}")


def load_losses_for_experiment(experiment_dir):
    """Load all losses.json files from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    experiments_dir = experiment_dir / "experiments"

    if not experiments_dir.exists():
        raise ValueError(f"Experiments directory not found: {experiments_dir}")

    # Data structure: {z_value: {iteration: losses_data}}
    all_losses = defaultdict(dict)

    # Find all Z_* directories
    for z_dir in sorted(experiments_dir.iterdir()):
        if not z_dir.is_dir() or not z_dir.name.startswith("Z_"):
            continue

        # Extract Z value from directory name using naming convention
        dirname_suffix = z_dir.name.split("_", 1)[1]
        z_value = parse_z_from_dirname(dirname_suffix)

        iterations_dir = z_dir / "iterations"
        if not iterations_dir.exists():
            continue

        # Load each iteration's losses
        for iter_dir in sorted(iterations_dir.iterdir()):
            if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
                continue

            losses_path = iter_dir / "model" / "losses.json"
            if losses_path.exists():
                with open(losses_path, 'r') as f:
                    losses_data = json.load(f)
                    iter_num = int(iter_dir.name.split("_")[1])
                    all_losses[z_value][iter_num] = losses_data

    return dict(all_losses)


def plot_loss_curves_selected_z(all_losses, output_path, z_values_to_plot=None):
    """Plot training/validation loss curves for selected Z values."""

    if z_values_to_plot is None:
        # Default: show a few representative Z values
        available_z = sorted(all_losses.keys())
        z_values_to_plot = [0.0, 0.5, 1.0, 1.4, 2.0, 3.0]
        z_values_to_plot = [z for z in z_values_to_plot if z in available_z]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, z_val in enumerate(z_values_to_plot[:6]):
        ax = axes[idx]
        iterations_data = all_losses.get(z_val, {})

        if not iterations_data:
            ax.set_title(f"Z = {z_val:.1f} (no data)")
            continue

        # Collect all train/val losses across iterations
        all_train = []
        all_val = []
        eval_steps = None

        for iter_num, data in iterations_data.items():
            if eval_steps is None:
                eval_steps = data.get('eval_steps', [])
            all_train.append(data.get('train_losses', []))
            all_val.append(data.get('val_losses', []))

        if not all_train or not eval_steps:
            ax.set_title(f"Z = {z_val:.1f} (no loss data)")
            continue

        all_train = np.array(all_train)
        all_val = np.array(all_val)

        # Calculate mean and std
        train_mean = np.mean(all_train, axis=0)
        train_std = np.std(all_train, axis=0)
        val_mean = np.mean(all_val, axis=0)
        val_std = np.std(all_val, axis=0)

        # Plot
        ax.plot(eval_steps, train_mean, 'b-', label='Train', linewidth=2)
        ax.fill_between(eval_steps, train_mean - train_std, train_mean + train_std,
                       color='blue', alpha=0.2)

        ax.plot(eval_steps, val_mean, 'r-', label='Validation', linewidth=2)
        ax.fill_between(eval_steps, val_mean - val_std, val_mean + val_std,
                       color='red', alpha=0.2)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(rf'$\alpha$ = {z_val:.1f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(r'Training and Validation Loss Curves by $\alpha$ Value' + '\n(mean ± std across 10 iterations)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_comparison_3z(all_losses, output_path, z_values_to_plot=None):
    """Plot training/validation loss curves for 3 selected Z values in a 1x3 layout."""

    if z_values_to_plot is None:
        z_values_to_plot = [0.0, 1.4, 3.0]

    n_runs = None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, z_val in enumerate(z_values_to_plot):
        ax = axes[idx]
        iterations_data = all_losses.get(z_val, {})

        if not iterations_data:
            ax.set_title(f"z = {z_val:.1f} (no data)", fontsize=14)
            continue

        if n_runs is None:
            n_runs = len(iterations_data)

        all_train = []
        all_val = []
        eval_steps = None

        for iter_num, data in iterations_data.items():
            if eval_steps is None:
                eval_steps = data.get('eval_steps', [])
            all_train.append(data.get('train_losses', []))
            all_val.append(data.get('val_losses', []))

        if not all_train or not eval_steps:
            ax.set_title(f"z = {z_val:.1f} (no loss data)", fontsize=14)
            continue

        all_train = np.array(all_train)
        all_val = np.array(all_val)

        train_mean = np.mean(all_train, axis=0)
        train_std = np.std(all_train, axis=0)
        val_mean = np.mean(all_val, axis=0)
        val_std = np.std(all_val, axis=0)

        # Use high-contrast colors: dark blue vs dark red/orange
        ax.plot(eval_steps, train_mean, color='#0059b3', linestyle='-',
                label='Train', linewidth=2.5)
        ax.fill_between(eval_steps, train_mean - train_std, train_mean + train_std,
                        color='#0059b3', alpha=0.15)

        ax.plot(eval_steps, val_mean, color='#cc3300', linestyle='--',
                label='Validation', linewidth=2.5)
        ax.fill_between(eval_steps, val_mean - val_std, val_mean + val_std,
                        color='#cc3300', alpha=0.15)

        ax.set_xlabel('Training Steps', fontsize=14)
        if idx == 0:
            ax.set_ylabel('Loss', fontsize=14)
        ax.set_title(rf'$\alpha$ = {z_val:.1f}', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=12)

    n_label = n_runs if n_runs else '?'
    plt.suptitle(r'Training and Validation Loss at Three $\alpha$ Values' + f' (n = {n_label} runs)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_loss_vs_z(all_losses, output_path):
    """Plot final train/val/test loss vs Z value."""

    z_values = sorted(all_losses.keys())

    train_means, train_stds = [], []
    val_means, val_stds = [], []
    test_means, test_stds = [], []

    for z_val in z_values:
        iterations_data = all_losses[z_val]

        final_train = []
        final_val = []
        final_test = []

        for iter_num, data in iterations_data.items():
            if 'train_losses' in data and data['train_losses']:
                final_train.append(data['train_losses'][-1])
            if 'val_losses' in data and data['val_losses']:
                final_val.append(data['val_losses'][-1])
            if 'test_loss' in data:
                final_test.append(data['test_loss'])

        train_means.append(np.mean(final_train) if final_train else np.nan)
        train_stds.append(np.std(final_train) if final_train else 0)
        val_means.append(np.mean(final_val) if final_val else np.nan)
        val_stds.append(np.std(final_val) if final_val else 0)
        test_means.append(np.mean(final_test) if final_test else np.nan)
        test_stds.append(np.std(final_test) if final_test else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(z_values, train_means, yerr=train_stds,
                fmt='b-o', label='Train Loss', capsize=3, markersize=6)
    ax.errorbar(z_values, val_means, yerr=val_stds,
                fmt='r-s', label='Validation Loss', capsize=3, markersize=6)

    if any(not np.isnan(t) for t in test_means):
        ax.errorbar(z_values, test_means, yerr=test_stds,
                    fmt='g-^', label='Test Loss', capsize=3, markersize=6)

    ax.set_xlabel(r'$\alpha$ Value', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title(r'Final Loss vs Zipfian Parameter $\alpha$' + '\n(mean ± std across iterations)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_best_step_vs_z(all_losses, output_path):
    """Plot which training step had the best validation loss for each Z value."""

    z_values = sorted(all_losses.keys())

    best_step_means = []
    best_step_stds = []

    for z_val in z_values:
        iterations_data = all_losses[z_val]

        best_steps = []
        for iter_num, data in iterations_data.items():
            if 'best_step' in data:
                best_steps.append(data['best_step'])

        best_step_means.append(np.mean(best_steps) if best_steps else np.nan)
        best_step_stds.append(np.std(best_steps) if best_steps else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(z_values, best_step_means, yerr=best_step_stds,
                fmt='purple', marker='o', capsize=3, markersize=6, linewidth=2)

    ax.axhline(y=1199, color='gray', linestyle='--', alpha=0.5, label='Max step (1199)')

    ax.set_xlabel(r'$\alpha$ Value', fontsize=12)
    ax.set_ylabel('Best Checkpoint Step', fontsize=12)
    ax.set_title(r'Training Step with Best Validation Loss vs $\alpha$' + '\n(mean ± std across iterations)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1300)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_iterations(all_losses, output_path, z_value=1.4):
    """Plot all individual iteration loss curves for a single Z value."""

    if z_value not in all_losses:
        print(f"Z value {z_value} not found in data")
        return

    iterations_data = all_losses[z_value]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for iter_num, data in sorted(iterations_data.items()):
        eval_steps = data.get('eval_steps', [])
        train_losses = data.get('train_losses', [])
        val_losses = data.get('val_losses', [])

        if eval_steps and train_losses:
            ax1.plot(eval_steps, train_losses, color=colors[iter_num-1],
                    alpha=0.7, label=f'Iter {iter_num}')
        if eval_steps and val_losses:
            ax2.plot(eval_steps, val_losses, color=colors[iter_num-1],
                    alpha=0.7, label=f'Iter {iter_num}')

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'Training Loss (Z = {z_value})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title(f'Validation Loss (Z = {z_value})')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'All 10 Iterations for Z = {z_value}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training losses from comprehensive experiments')
    parser.add_argument('experiment_dir', type=str, help='Path to comprehensive experiment directory')
    parser.add_argument('--z-detail', type=float, default=1.4,
                       help='Z value to show detailed iteration plots (default: 1.4)')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    print(f"Loading losses from: {experiment_dir}")
    all_losses = load_losses_for_experiment(experiment_dir)

    if not all_losses:
        print("No loss data found. Make sure the experiment has losses.json files.")
        print("(These are only generated by experiments run after the tracking was added)")
        return

    print(f"Found data for {len(all_losses)} Z values")

    # Create output directory
    output_dir = experiment_dir / "loss_plots"
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_loss_comparison_3z(all_losses, output_dir / "loss_comparison_3z_alpha.png")
    plot_loss_curves_selected_z(all_losses, output_dir / "loss_curves_by_z.png")
    plot_final_loss_vs_z(all_losses, output_dir / "final_loss_vs_z.png")
    plot_best_step_vs_z(all_losses, output_dir / "best_step_vs_z.png")
    plot_all_iterations(all_losses, output_dir / "iterations_detail.png", z_value=args.z_detail)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
