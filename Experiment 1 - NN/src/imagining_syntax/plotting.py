"""Plotting for imagining_syntax experiments.

Three entry points:

- plot_zipfian_sweep(csv_path, output_path) — 4-line plot of mean accuracy vs α.
- plot_oneshot_bars(csv_path, output_path) — 4-bar chart of one-α experiment.
- plot_paper_figure(zipfian_csv, oneshot_csv, output_path) — Figure 2 of the
  Collocational Bootstrapping paper: zipfian sweep + α→∞ point with a
  vertical separator and chance-baseline line.

CSVs are the comprehensive_results.csv format produced by
experiment.stats.create_comprehensive_summary: header
`param_value,eval_type,mean,std,min,max`."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS = ("seen_match", "seen_mismatch", "unseen_match", "unseen_mismatch")
COLORS = {
    "seen_match": "#4472C4",
    "unseen_match": "#C00000",
    "seen_mismatch": "#FFC000",
    "unseen_mismatch": "#7030A0",
}
MARKERS = {
    "seen_match": "o",
    "seen_mismatch": "s",
    "unseen_match": "^",
    "unseen_mismatch": "v",
}
LABELS = {
    "seen_match": "Seen, Match",
    "seen_mismatch": "Seen, Mismatch",
    "unseen_match": "Unseen, Match",
    "unseen_mismatch": "Unseen, Mismatch",
}
CHANCE_BASELINE = 50.0  # binary-choice minimal-pair baseline


def plot_zipfian_sweep(csv_path, output_path):
    """4-line plot of mean accuracy vs α with ±std bands and chance baseline."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in CONDITIONS:
        subset = df[df["eval_type"] == cond].sort_values("param_value")
        if subset.empty:
            continue
        ax.plot(subset["param_value"], subset["mean"], color=COLORS[cond],
                marker=MARKERS[cond], markersize=5, linewidth=2, label=LABELS[cond])
        upper = np.minimum(subset["mean"] + subset["std"], 100)
        lower = np.maximum(subset["mean"] - subset["std"], 0)
        ax.fill_between(subset["param_value"], lower, upper,
                        color=COLORS[cond], alpha=0.2)
    ax.axhline(CHANCE_BASELINE, color="gray", linestyle=":",
               linewidth=1, label="Chance Baseline")
    ax.set_xlabel("α value", fontsize=12)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax.set_title("Model accuracy vs Zipfian α", fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_oneshot_bars(csv_path, output_path):
    """4-bar chart of mean accuracy ±std for a one-parameter (oneshot) run."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    means, stds, colors, labels = [], [], [], []
    for cond in CONDITIONS:
        subset = df[df["eval_type"] == cond]
        if subset.empty:
            continue
        means.append(float(subset["mean"].iloc[0]))
        stds.append(float(subset["std"].iloc[0]))
        colors.append(COLORS[cond])
        labels.append(LABELS[cond])
    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=5, edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax.set_title("Oneshot accuracy by condition", fontsize=14)
    ax.set_ylim(0, 105)
    ax.axhline(CHANCE_BASELINE, color="gray", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_figure(zipfian_csv, oneshot_csv, output_path):
    """Figure 2 of the Collocational Bootstrapping paper.

    Reads two CSVs and produces a single figure: zipfian sweep on the left
    with the four condition lines + chance baseline, and a separate α→∞
    point on the right (offset past the sweep's α_max). Error bars span the
    observed [min, max] across iterations (not ±std) — they're naturally
    bounded by the [0, 100] accuracy range and match the cap-and-line style
    used by the original `plot_comprehensive_results.py` min/max indicators."""
    z_df = pd.read_csv(zipfian_csv)
    o_df = pd.read_csv(oneshot_csv)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    z_min = float(z_df["param_value"].min())
    z_max = float(z_df["param_value"].max())
    oneshot_x = z_max + 0.3

    for cond in CONDITIONS:
        z_sub = z_df[z_df["eval_type"] == cond].sort_values("param_value")
        if not z_sub.empty:
            mean = z_sub["mean"].to_numpy()
            lower_err = mean - z_sub["min"].to_numpy()  # span from observed min
            upper_err = z_sub["max"].to_numpy() - mean  # span to observed max
            ax.errorbar(
                z_sub["param_value"], mean, yerr=[lower_err, upper_err],
                color=COLORS[cond], marker=MARKERS[cond], markersize=4,
                linewidth=1.0, elinewidth=0.8, capsize=2, capthick=0.8,
                label=LABELS[cond],
            )
        o_sub = o_df[o_df["eval_type"] == cond]
        if not o_sub.empty:
            mean_v = float(o_sub["mean"].iloc[0])
            lower_v = mean_v - float(o_sub["min"].iloc[0])
            upper_v = float(o_sub["max"].iloc[0]) - mean_v
            ax.errorbar(
                oneshot_x, mean_v, yerr=[[lower_v], [upper_v]],
                color=COLORS[cond], marker=MARKERS[cond], markersize=4,
                elinewidth=0.8, capsize=2, capthick=0.8, linestyle="none",
            )

    ax.axhline(CHANCE_BASELINE, color="gray", linestyle="--",
               linewidth=1.0, label="Chance Baseline")

    ax.set_xlabel("α Value", fontsize=10)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(z_min - 0.1, oneshot_x + 0.2)

    sweep_ticks = list(np.arange(z_min, z_max + 1e-9, 0.2))
    xticks = sweep_ticks + [oneshot_x]
    xticklabels = [f"{v:.1f}" for v in sweep_ticks] + ["α → ∞"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="both", labelsize=8)

    ax.legend(loc="lower right", frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
