#!/usr/bin/env python3
"""
Scatter plot comparing R² performance of two MLR summary files.

Usage:
    python plot_mlr_comparison.py <file_x> <file_y> [--out output.pdf]

Example:
    python plot_comparison.py \
        results/summary/1mer.txt \
        results/summary/1mer+deepdnashape.txt \
        --out comparison.pdf

File format (fixed-width):
    Dataset                                   Mean R2   Mean MAE
    ------------------------------------------------------------------
    C2H2_BCL6B_...                              0.47     0.05
    ...
    ------------------------------------------------------------------
    Average (N datasets)                        0.50     0.05
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import wilcoxon


def parse_summary(filepath, metric="r2"):
    """
    Parse an MLR summary file.
    Returns a dict mapping dataset_name -> metric value.
    Skips header, separator, and average lines.

    metric: "r2", "adj_r2", or "mae"
    """
    # detect whether the file has an Adj_R2 column by checking the header
    has_adj_r2 = False
    with open(filepath) as f:
        for line in f:
            if line.startswith("Dataset"):
                has_adj_r2 = "Adj_R2" in line
                break

    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            # skip header, separator, and average lines
            if not line or line.startswith("Dataset") or line.startswith("-") or line.startswith("Average"):
                continue
            parts = line.split()
            try:
                if has_adj_r2:
                    # columns: dataset, R2, Adj_R2, MAE, ...
                    r2 = float(parts[1])
                    adj_r2 = float(parts[2])
                    mae = float(parts[3])
                else:
                    # old format: dataset, R2, MAE, ...
                    r2 = float(parts[1])
                    adj_r2 = None
                    mae = float(parts[2])
            except (ValueError, IndexError):
                continue
            dataset = parts[0]
            if metric == "adj_r2":
                if adj_r2 is None:
                    continue
                results[dataset] = adj_r2
            elif metric == "mae":
                results[dataset] = mae
            else:
                results[dataset] = r2
    return results


def get_tf_class(dataset_name):
    """Extract the TF family from the dataset name (first underscore-delimited field)."""
    return dataset_name.split("_")[0]

def desaturate(color, factor=0.3):
    """Blend color toward gray. factor=0 is original, factor=1 is full gray."""
    r, g, b, a = color
    return (r + (0.5 - r) * factor, g + (0.5 - g) * factor, b + (0.5 - b) * factor, a)

def wilcoxon_pvalue(x_vals, y_vals):
    """Wilcoxon signed-rank test comparing paired metric values (two-sided)."""
    diff = y_vals - x_vals
    if np.all(diff == 0):
        return 1.0
    stat, p = wilcoxon(diff)
    return p


def format_pvalue(p):
    """Format p-value for display."""
    if p < 1e-4:
        return f"$p$ = {p:.2e}"
    return f"$p$ = {p:.4f}"


MARKERS = ["o", "^", "^", "D", "v", "P", "*", "s", "p", "h"]

def make_style_map(tf_classes):
    """Assign a color + marker to each unique TF class.
    Colors cycle through tab20 (20 colors); markers change when colors repeat."""
    unique_classes = sorted(set(tf_classes))
    cmap = plt.get_cmap("tab20", 20)
    style = {}
    for i, cls in enumerate(unique_classes):
        color = cmap(i % 20)
        marker = MARKERS[(i // 20) % len(MARKERS)]
        style[cls] = (color, marker)
    return style



# ── Font size configuration ──────────────────────────────────────────
FONTSIZE_TICK = 12          # x and y tick labels
FONTSIZE_AXIS_LABEL = 14    # x and y axis labels
FONTSIZE_ANNOTATION = 13    # annotated text (mean, counts)
FONTSIZE_POINT_LABEL = 5    # per-point labels (if enabled)
FONTSIZE_LEGEND = 7         # legend entries
FONTSIZE_LEGEND_TITLE = 8   # legend title


def main():
    parser = argparse.ArgumentParser(description="Compare two MLR summary files via R² scatter plot.")
    parser.add_argument("file_x", help="Summary file for the x-axis (baseline).")
    parser.add_argument("file_y", help="Summary file for the y-axis (comparison).")
    parser.add_argument("--out", default=None, help="Output file path (default: comparison_<x>_vs_<y>.pdf).")
    parser.add_argument("--metric", choices=["r2", "adj_r2", "mae"], default="r2",
                        help="Metric to plot: r2 (default), adj_r2 (adjusted R²), or mae.")
    parser.add_argument("--legend-only", action="store_true",
                        help="Save only the legend as a standalone figure.")
    args = parser.parse_args()

    data_x = parse_summary(args.file_x, metric=args.metric)
    data_y = parse_summary(args.file_y, metric=args.metric)

    # keep only datasets present in both files
    common = sorted(set(data_x) & set(data_y))
    if not common:
        sys.exit("Error: no common datasets found between the two files.")

    x_vals = np.array([data_x[d] for d in common])
    y_vals = np.array([data_y[d] for d in common])
    tf_classes = [get_tf_class(d) for d in common]
    style_map = make_style_map(tf_classes)

    label_x = Path(args.file_x).name.replace('.txt', '')
    label_y = Path(args.file_y).name.replace('.txt', '')

    fig, ax = plt.subplots(figsize=(6, 6))

    # axis limits: fixed [0,1] for R² metrics, data-driven for MAE
    if args.metric == "mae":
        lo = 0
        hi = max(x_vals.max(), y_vals.max()) * 1.05
    else:
        lo, hi = 0, 1

    # diagonal reference line
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1, linestyle="--", zorder=1)

    # scatter per TF class (each class gets its own color + marker)
    for cls in sorted(style_map):
        color, marker = style_map[cls]
        mask = [c == cls for c in tf_classes]
        ax.scatter(x_vals[mask], y_vals[mask], c=[color], marker=marker,
                   s=80, edgecolors="white", linewidths=0.4, zorder=2)

    # dataset labels on hover would need interactivity; instead annotate nothing by default
    # (uncomment below to annotate all points — can get crowded)
    # for d, xv, yv in zip(common, x_vals, y_vals):
    #     ax.annotate(d.split("_")[1], (xv, yv), fontsize=FONTSIZE_POINT_LABEL, ha="left", va="bottom")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    metric_labels = {"r2": r"$R^2$", "adj_r2": r"Adjusted $R^2$", "mae": "MAE"}
    metric_label = metric_labels[args.metric]
    ax.set_xlabel(f"{metric_label}  ({label_x})", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylabel(f"{metric_label}  ({label_y})", fontsize=FONTSIZE_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    # ax.set_title(f"MLR performance: {label_x}  vs  {label_y}", fontsize=11)
    ax.set_aspect("equal")
    # ax.grid(True, linewidth=0.4, alpha=0.5)

    # mean R² annotations
    ax.text(0.02, 0.96, f"mean $x$: {x_vals.mean():.3f}", transform=ax.transAxes,
            fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    ax.text(0.02, 0.91, f"mean $y$: {y_vals.mean():.3f}", transform=ax.transAxes,
            fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    if args.metric == "mae":
        n_below = (y_vals < x_vals).sum()
        ax.text(0.02, 0.86, f"$y < x$: {n_below}/{len(common)}", transform=ax.transAxes,
                fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    else:
        n_above = (y_vals > x_vals).sum()
        ax.text(0.02, 0.86, f"$y > x$: {n_above}/{len(common)}", transform=ax.transAxes,
                fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    p = wilcoxon_pvalue(x_vals, y_vals)
    ax.text(0.02, 0.81, f"Wilcoxon {format_pvalue(p)}", transform=ax.transAxes,
            fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")

    # legend for TF classes
    legend_handles = [
        mlines.Line2D([], [], color=style_map[cls][0], marker=style_map[cls][1],
                       linestyle="None", markersize=6, label=cls)
        for cls in sorted(style_map)
    ]

    if args.legend_only:
        plt.close(fig)
        fig_leg = plt.figure(figsize=(3, 4))
        fig_leg.legend(handles=legend_handles, title="TF class",
                       fontsize=FONTSIZE_LEGEND, title_fontsize=FONTSIZE_LEGEND_TITLE,
                       loc="center", frameon=False,
                       ncol=1)
        plots_dir = Path(__file__).parent.parent / "plots/mlr"
        plots_dir.mkdir(exist_ok=True)
        out_path = args.out or str(plots_dir / f"legend.png")
        fig_leg.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        return

    # ax.legend(handles=legend_handles, title="TF class", fontsize=FONTSIZE_LEGEND,
    #           title_fontsize=FONTSIZE_LEGEND_TITLE, loc="lower right", framealpha=0.8,
    #           ncol=1 if len(color_map) <= 12 else 2)

    plt.tight_layout()

    plots_dir = Path(__file__).parent.parent / f"plots/mlr/{args.metric}"
    plots_dir.mkdir(exist_ok=True)
    out_path = args.out or str(plots_dir / f"{label_x}_vs_{label_y}_{args.metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
