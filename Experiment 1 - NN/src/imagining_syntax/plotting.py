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
    point on the right (offset past the sweep's α_max), with a dashed
    vertical separator and a large 'α → ∞' annotation in the figure body.
    Error bars are ±std (raw, unclipped — bars may extend below 0 when
    runs straddle the floor of the accuracy range)."""
    z_df = pd.read_csv(zipfian_csv)
    o_df = pd.read_csv(oneshot_csv)

    # Paper-figure colors (4 distinct hues for visual cohesion with the
    # distribution-curves figure). Kept local so the module-level COLORS
    # used by plot_zipfian_sweep / plot_oneshot_bars are untouched.
    paper_colors = {
        "seen_match":      "#3182bd",  # blue
        "seen_mismatch":   "#feb24c",  # yellow-orange
        "unseen_match":    "#de2d26",  # red
        "unseen_mismatch": "#756bb1",  # purple
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    z_min = float(z_df["param_value"].min())
    z_max = float(z_df["param_value"].max())
    oneshot_x = z_max + 0.2
    separator_x = z_max + 0.1

    # Zipfian sweep: mean line with markers, then ±std error bars (raw)
    for cond in CONDITIONS:
        z_sub = z_df[z_df["eval_type"] == cond].sort_values("param_value")
        if z_sub.empty:
            continue
        color = paper_colors[cond]
        ax.plot(z_sub["param_value"], z_sub["mean"],
                color=color, linewidth=2.5,
                marker=MARKERS[cond], markersize=8,
                label=LABELS[cond], zorder=3)
        ax.errorbar(z_sub["param_value"], z_sub["mean"],
                    yerr=z_sub["std"],
                    color=color, capsize=4, capthick=1.5,
                    linewidth=1.5, zorder=2)

    # Oneshot point — render in reverse order so seen_match lands on top
    oneshot_order = ("unseen_mismatch", "unseen_match", "seen_mismatch", "seen_match")
    for cond in oneshot_order:
        o_sub = o_df[o_df["eval_type"] == cond]
        if o_sub.empty:
            continue
        mean_v = float(o_sub["mean"].iloc[0])
        std_v = float(o_sub["std"].iloc[0])
        ax.errorbar(oneshot_x, mean_v,
                    yerr=std_v,
                    marker=MARKERS[cond], markersize=10,
                    color=paper_colors[cond],
                    capsize=5, capthick=2,
                    linewidth=1.5, linestyle="none", zorder=4)

    # Dashed vertical separator between sweep and oneshot
    ax.axvline(x=separator_x, color="gray", linestyle="--",
               linewidth=1, alpha=0.5)

    # Big 'α → ∞' annotation inside the chart, near the oneshot point
    ax.annotate(r"$\alpha \to \infty$", xy=(oneshot_x, 55),
                fontsize=28, fontweight="bold", ha="center", va="top")

    # Chance baseline
    ax.axhline(y=CHANCE_BASELINE, color="black", linestyle="--",
               linewidth=1.5, alpha=0.5, zorder=1, label="Chance Baseline")

    # Axis labels (large, paper-style)
    ax.set_xlabel(r"$\alpha$ Value", fontsize=28, fontweight="bold")
    ax.set_ylabel("Mean Accuracy (%)", fontsize=28, fontweight="bold")

    # X-axis ticks: 0.2-step sweep ticks + the oneshot tick labelled ∞
    sweep_ticks = list(np.arange(z_min, z_max + 1e-9, 0.2))
    xticks = sweep_ticks + [oneshot_x]
    xticklabels = [f"{v:.1f}" for v in sweep_ticks] + [r"$\infty$"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=22)

    ax.set_xlim(z_min - 0.1, oneshot_x + 0.25)
    ax.set_ylim(-10, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=22)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend in paper order: Seen Match, Unseen Match, Seen Mismatch,
    # Unseen Mismatch, Chance Baseline
    handles, legend_labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(legend_labels, handles))
    desired = ["Seen, Match", "Unseen, Match",
               "Seen, Mismatch", "Unseen, Mismatch", "Chance Baseline"]
    ordered_labels = [l for l in desired if l in label_to_handle]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, loc="lower right",
              fontsize=22, frameon=True, fancybox=True, shadow=True,
              framealpha=0.9, markerscale=1.75)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
