#!/usr/bin/env python3
"""
Multi-panel scatter plot comparing a metric across up to 4 pairs of MLR
summary files. Panels are arranged in a 2×2 grid with a shared TF-class
legend in a single column on the right.

Usage:
    python plot_comparison_multi.py \
        --panel file_x1 file_y1 \
        --panel file_x2 file_y2 \
        [--panel file_x3 file_y3] \
        [--panel file_x4 file_y4] \
        [--metric r2] \
        [--out output.pdf]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import wilcoxon


# ── Label replacements applied to filename-derived axis labels ───────
LABEL_REPLACEMENTS = {
    "deepdnashape_mgw": "Deep DNAshape MGW",
    "deepdnashape": "Deep DNAshape",
}

# ── Font sizes ────────────────────────────────────────────────────────
FONTSIZE_TICK         = 11
FONTSIZE_AXIS_LABEL   = 12
FONTSIZE_ANNOTATION   = 10
FONTSIZE_LEGEND       = 12
FONTSIZE_LEGEND_TITLE = 13

MARKERS = ["o", "^", "D", "v", "P", "*", "s", "p", "h"]


def parse_summary(filepath, metric="r2"):
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
            if not line or line.startswith("Dataset") or line.startswith("-") or line.startswith("Average"):
                continue
            parts = line.split()
            try:
                if has_adj_r2:
                    r2, adj_r2, mae = float(parts[1]), float(parts[2]), float(parts[3])
                else:
                    r2, adj_r2, mae = float(parts[1]), None, float(parts[2])
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
    return dataset_name.split("_")[0]


def wilcoxon_pvalue(x_vals, y_vals):
    diff = y_vals - x_vals
    if np.all(diff == 0):
        return 1.0
    _, p = wilcoxon(diff)
    return p


def format_pvalue(p):
    if p < 1e-4:
        return f"$p$ = {p:.2e}"
    return f"$p$ = {p:.4f}"


def make_style_map(tf_classes):
    unique_classes = sorted(set(tf_classes))
    cmap = plt.get_cmap("tab20", 20)
    return {
        cls: (cmap(i % 20), MARKERS[(i // 20) % len(MARKERS)])
        for i, cls in enumerate(unique_classes)
    }


def apply_label(name):
    label = Path(name).name.replace(".txt", "")
    for src, dst in LABEL_REPLACEMENTS.items():
        label = label.replace(src, dst)
    return label


def draw_panel(ax, x_vals, y_vals, tf_classes, style_map, label_x, label_y, metric):
    metric_labels = {"r2": r"$R^2$", "adj_r2": r"Adjusted $R^2$", "mae": "MAE"}
    metric_label = metric_labels[metric]

    lo, hi = (0, max(x_vals.max(), y_vals.max()) * 1.05) if metric == "mae" else (0, 1)

    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1, linestyle="--", zorder=1)

    for cls in sorted(style_map):
        color, marker = style_map[cls]
        mask = np.array([c == cls for c in tf_classes])
        if mask.any():
            ax.scatter(x_vals[mask], y_vals[mask], c=[color], marker=marker,
                       s=60, edgecolors="white", linewidths=0.4, zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{metric_label}  ({label_x})", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylabel(f"{metric_label}  ({label_y})", fontsize=FONTSIZE_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.set_aspect("equal")

    n = len(x_vals)
    ax.text(0.02, 0.96, f"mean $x$: {x_vals.mean():.3f}", transform=ax.transAxes,
            fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    ax.text(0.02, 0.91, f"mean $y$: {y_vals.mean():.3f}", transform=ax.transAxes,
            fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    if metric == "mae":
        ax.text(0.02, 0.86, f"$y < x$: {(y_vals < x_vals).sum()}/{n}",
                transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    else:
        ax.text(0.02, 0.86, f"$y > x$: {(y_vals > x_vals).sum()}/{n}",
                transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")
    p = wilcoxon_pvalue(x_vals, y_vals)
    ax.text(0.02, 0.81, f"Wilcoxon {format_pvalue(p)}", transform=ax.transAxes,
            fontsize=FONTSIZE_ANNOTATION, va="top", color="dimgray")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-panel scatter plot for up to 4 MLR summary file pairs."
    )
    parser.add_argument("--panel", nargs=2, metavar=("FILE_X", "FILE_Y"),
                        action="append", default=[],
                        help="A (file_x, file_y) pair. Repeat up to 4 times.")
    parser.add_argument("--metric", choices=["r2", "adj_r2", "mae"], default="r2",
                        help="Metric to plot (default: r2).")
    parser.add_argument("--out", default="comparison_4panel.png",
                        help="Output file path (default: comparison_4panel.png in cwd).")
    args = parser.parse_args()

    if not args.panel:
        sys.exit("Error: provide at least one --panel FILE_X FILE_Y.")
    if len(args.panel) > 4:
        sys.exit("Error: at most 4 --panel pairs are supported.")

    # load and validate each panel's data
    panel_data = []
    for file_x, file_y in args.panel:
        data_x = parse_summary(file_x, metric=args.metric)
        data_y = parse_summary(file_y, metric=args.metric)
        common = sorted(set(data_x) & set(data_y))
        if not common:
            sys.exit(f"Error: no common datasets between {file_x} and {file_y}.")
        x_vals = np.array([data_x[d] for d in common])
        y_vals = np.array([data_y[d] for d in common])
        tf_classes = [get_tf_class(d) for d in common]
        panel_data.append((x_vals, y_vals, tf_classes, apply_label(file_x), apply_label(file_y)))

    # build shared style_map from union of all TF classes
    all_tf_classes = [cls for _, _, tfc, _, _ in panel_data for cls in tfc]
    style_map = make_style_map(all_tf_classes)

    # figure: 2×2 panels + narrow legend column
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.3],
                           wspace=0.2, hspace=0.25, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_leg = fig.add_subplot(gs[:, 2])
    ax_leg.axis("off")
    pos = ax_leg.get_position()
    ax_leg.set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])

    for i, ax in enumerate(axes):
        if i < len(panel_data):
            x_vals, y_vals, tf_classes, label_x, label_y = panel_data[i]
            draw_panel(ax, x_vals, y_vals, tf_classes, style_map,
                       label_x, label_y, args.metric)
        else:
            ax.set_visible(False)

    legend_handles = [
        mlines.Line2D([], [], color=style_map[cls][0], marker=style_map[cls][1],
                      linestyle="None", markersize=6, label=cls)
        for cls in sorted(style_map)
    ]
    ax_leg.legend(handles=legend_handles, title="TF class",
                  fontsize=FONTSIZE_LEGEND, title_fontsize=FONTSIZE_LEGEND_TITLE,
                  loc="upper left", frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
