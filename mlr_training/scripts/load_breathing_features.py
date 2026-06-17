#!/usr/bin/env python3
"""
Load and inspect DNA breathing pkl features for a given experiment directory.

Usage:
    python load_breathing_features.py <experiment_dir>
    python load_breathing_features.py <experiment_dir> --max-seqs 1000
    python load_breathing_features.py <experiment_dir> --summary-only
    python load_breathing_features.py <experiment_dir> --seq-idx 42

Each pkl file corresponds to one DNA sequence and contains:
    bubbles     : (n_temp, seq_len, n_temp2) float64
    coord       : (n_temp,) float64
    flip_verbose: (n_temp, 5) float64

PKL directory: $DNABREATHING_DATA/pyDNA_EPBD_outputs/  (default: <mlr_training>/data/pyDNA_EPBD_outputs/)
"""

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

from config import BREATHING_DIR as PKL_BASE


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_sorted_pkl_files(exp_dir):
    """Return pkl files sorted numerically by index."""
    pkl_files = list(Path(exp_dir).glob("*.pkl"))
    pkl_files.sort(key=lambda p: int(p.stem))
    return pkl_files


def inspect_single(pkl_path):
    """Print detailed info for a single pkl file."""
    data = load_pkl(pkl_path)
    print(f"\n=== {pkl_path} ===")
    for key, val in data.items():
        arr = np.asarray(val)
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"    min={arr.min():.4g}, max={arr.max():.4g}, "
              f"mean={arr.mean():.4g}, nonzero={np.count_nonzero(arr)}")
    return data


def load_all(exp_dir, max_seqs=None):
    """
    Load all pkl files in exp_dir up to max_seqs.

    Returns dict:
        bubbles     : (N, n_temp, seq_len, n_temp2)
        coord       : (N, n_temp)
        flip_verbose: (N, n_temp, 5)
        indices     : list of int (pkl file stems)
    """
    pkl_files = get_sorted_pkl_files(exp_dir)
    if max_seqs is not None:
        pkl_files = pkl_files[:max_seqs]

    print(f"Loading {len(pkl_files)} pkl files from {exp_dir} ...")

    bubbles_list = []
    coord_list = []
    flip_list = []
    indices = []

    y_slice = 10
    z_slice = 6

    for i, pkl_path in enumerate(pkl_files):
        data = load_pkl(pkl_path)
        bubbles_list.append(data["bubbles"][26:-26,y_slice,z_slice] / 80000)
        coord_list.append(data["coord"])
        flip_list.append(data["flip_verbose"][26:-26,0] / 80000)
        indices.append(int(pkl_path.stem))
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i+1}/{len(pkl_files)} ...")

    result = {
        "bubbles":      np.stack(bubbles_list, axis=0),
        "coord":        np.stack(coord_list, axis=0),
        "flip_verbose": np.stack(flip_list, axis=0),
        "indices":      indices,
    }
    print("Done.")
    return result


def load_tf_summary(exp_dir, max_seqs=None, agg="max"):
    """
    Load all pkl files in exp_dir and return per-sequence aggregated values
    for bubbles and flip (scalar per sequence).

    agg : "max" or "mean" — how to aggregate across positions for each sequence.

    Returns dict:
        bubbles : (N,) float64  — per-sequence max/mean over positions
        flip    : (N,) float64  — per-sequence max/mean over positions
    """
    pkl_files = get_sorted_pkl_files(exp_dir)
    if max_seqs is not None:
        pkl_files = pkl_files[:max_seqs]

    y_slice = 10
    z_slice = 6
    agg_fn = np.max if agg == "max" else np.mean

    bubbles_agg = []
    flip_agg = []

    for pkl_path in pkl_files:
        data = load_pkl(pkl_path)
        b = data["bubbles"][26:-26, y_slice, z_slice] / 80000   # (positions,)
        f = data["flip_verbose"][26:-26, 0] / 80000              # (positions,)
        bubbles_agg.append(agg_fn(b))
        flip_agg.append(agg_fn(f))

    return {
        "bubbles": np.array(bubbles_agg),
        "flip":    np.array(flip_agg),
    }


def plot_all_tf_boxplots(agg="max", max_seqs=None, out_path=None):
    """
    For every TF directory under PKL_BASE, compute per-sequence aggregated
    bubbles and flip values, then draw side-by-side boxplots.

    agg      : "max" or "mean"
    out_path : file to save figure (default: boxplots_<agg>.png in cwd)
    """
    tf_dirs = sorted(Path(PKL_BASE).iterdir())
    tf_dirs = [d for d in tf_dirs if d.is_dir()]
    labels = ['_'.join(d.name.split('_')[:4]) for d in tf_dirs]
    # labels = ['_'.join(d.name.split('_')[:2]) for d in tf_dirs]

    bubbles_data = [None] * len(tf_dirs)
    flip_data = [None] * len(tf_dirs)

    _load = partial(load_tf_summary, max_seqs=max_seqs, agg=agg)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_load, d): i for i, d in enumerate(tf_dirs)}
        for fut in as_completed(futures):
            i = futures[fut]
            print(f"  Done: {tf_dirs[i].name}")
            summary = fut.result()
            bubbles_data[i] = summary["bubbles"]
            flip_data[i] = summary["flip"]

    if out_path is None:
        out_path = f"./plots/statistics/boxplots_{agg}.png"

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(labels) * 0.5), 18))
    # fig, axes = plt.subplots(2, 1, figsize=(max(14, len(labels) * 0.5), 10))

    for ax, data, title in zip(axes, [bubbles_data, flip_data], ["Bubbles", "Flip"]):
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        # ax.set_title(f"{title} — per-sequence {agg} across positions")
        ax.set_ylabel(f"{agg.capitalize()} {title} Value", fontsize=14)
        # ax.set_ylabel(f"{agg.capitalize()} {title} Value", fontsize=12)
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis='both', labelsize=14)
        # ax.tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure: {out_path}")
    return out_path


def _load_tf_bubbles(exp_dir, max_seqs=None):
    """Return (per_seq_max, all_values) for bubbles in exp_dir."""
    pkl_files = get_sorted_pkl_files(exp_dir)
    if max_seqs is not None:
        pkl_files = pkl_files[:max_seqs]
    y_slice, z_slice = 10, 6
    per_seq_max = []
    all_values = []
    for pkl_path in pkl_files:
        b = load_pkl(pkl_path)["bubbles"][26:-26, y_slice, z_slice] / 80000
        per_seq_max.append(b.max())
        all_values.append(b)
    return np.array(per_seq_max), np.concatenate(all_values)


def plot_bubble_positive_proportion(max_seqs=None, out_path=None):
    """
    For every TF directory under PKL_BASE, compute the proportion of sequences
    where the per-sequence max bubble value exceeds the global 95th percentile
    of all bubble values across all TFs. Plot as a bar chart.

    max_seqs : cap per TF (default: all)
    out_path : output file (default: ./plots/statistics/bubble_positive_proportion.png)
    """
    tf_dirs = sorted(Path(PKL_BASE).iterdir())
    tf_dirs = [d for d in tf_dirs if d.is_dir()]
    labels = ['_'.join(d.name.split('_')[:4]) for d in tf_dirs]
    # labels = ['_'.join(d.name.split('_')[:2]) for d in tf_dirs]

    # Load per-sequence max and all raw bubble values for each TF
    per_seq_max = [None] * len(tf_dirs)
    all_values  = [None] * len(tf_dirs)
    _load = partial(_load_tf_bubbles, max_seqs=max_seqs)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_load, d): i for i, d in enumerate(tf_dirs)}
        for fut in as_completed(futures):
            i = futures[fut]
            print(f"  Done: {tf_dirs[i].name}")
            per_seq_max[i], all_values[i] = fut.result()

    proportions = []
    for vals, raw in zip(per_seq_max, all_values):
        p95 = np.percentile(raw, 95)   # 95th percentile of all position-level values
        proportions.append(float((vals > p95).mean()))

    if out_path is None:
        out_path = "./plots/statistics/bubble_positive_proportion.png"

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.5), 5))
    ax.bar(range(len(labels)), proportions)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_ylabel("Proportion of bubble-positive sequences", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, max(proportions) * 1.1)
    ax.margins(x=0.01)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure: {out_path}")
    return out_path


def _parse_mlr_summary(filepath):
    """Parse an MLR summary file → dict of dataset_name -> Mean R2."""
    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("Dataset") or line.startswith("-") or line.startswith("Average"):
                continue
            parts = line.split()
            try:
                r2 = float(parts[-2])
            except (ValueError, IndexError):
                continue
            results[parts[0]] = r2
    return results


def _get_tf_class(dataset_name):
    return dataset_name.split("_")[0]


def _make_color_map(tf_classes):
    unique_classes = sorted(set(tf_classes))
    cmap = plt.get_cmap("tab20", len(unique_classes))
    return {cls: cmap(i) for i, cls in enumerate(unique_classes)}


def plot_breathing_vs_mlr(mlr_file, feature="bubbles", agg="mean", max_seqs=None, out_path=None):
    """
    Scatter plot of per-TF mean breathing feature vs MLR R².
    Each point is one TF dataset, coloured by TF class.
    Spearman r is annotated on the plot.

    mlr_file : path to an MLR summary text file
    feature  : "bubbles" or "flip"
    agg      : "mean" or "max" — aggregation over positions per sequence
    out_path : output file (default: ./plots/statistics/breathing_vs_mlr_<feature>.png)
    """
    mlr = _parse_mlr_summary(mlr_file)

    tf_dirs = sorted(Path(PKL_BASE).iterdir())
    tf_dirs = [d for d in tf_dirs if d.is_dir()]

    # Match TF dirs to MLR entries by directory name
    matched_dirs   = [d for d in tf_dirs if d.name in mlr]
    matched_names  = [d.name for d in matched_dirs]
    r2_vals        = np.array([mlr[n] for n in matched_names])

    print(f"Matched {len(matched_dirs)} TF datasets to MLR file.")

    # Load per-TF mean breathing feature: mean over sequences of per-seq agg value
    breathing_vals = [None] * len(matched_dirs)
    _load = partial(load_tf_summary, max_seqs=max_seqs, agg=agg)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_load, d): i for i, d in enumerate(matched_dirs)}
        for fut in as_completed(futures):
            i = futures[fut]
            print(f"  Done: {matched_dirs[i].name}")
            breathing_vals[i] = float(fut.result()[feature].mean())

    breathing_vals = np.array(breathing_vals)

    tf_classes = [_get_tf_class(n) for n in matched_names]
    color_map  = _make_color_map(tf_classes)
    colors     = [color_map[c] for c in tf_classes]

    rho, pval = spearmanr(breathing_vals, r2_vals)

    mlr_label = Path(mlr_file).name.replace(".txt", "")
    feat_label = feature if feature == "bubbles" else "flip"

    if out_path is None:
        out_path = f"./plots/correlations/breathing_vs_mlr_{feat_label}_{mlr_label}.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(breathing_vals, r2_vals, c=colors, s=60,
               edgecolors="white", linewidths=0.4, zorder=2)

    ax.set_xlabel(f"Mean {agg} {feat_label} value per TF", fontsize=13)
    ax.set_ylabel(f"MLR Mean R²  ({mlr_label})", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.text(0.04, 0.96, f"Spearman r = {rho:.3f}  (p = {pval:.2g})",
            transform=ax.transAxes, fontsize=11, va="top", color="dimgray")

    legend_handles = [
        mpatches.Patch(color=color_map[cls], label=cls)
        for cls in sorted(color_map)
    ]
    ax.legend(handles=legend_handles, title="TF class", fontsize=7,
              title_fontsize=8, loc="lower right", framealpha=0.8,
              ncol=1 if len(color_map) <= 12 else 2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    return out_path


def print_summary(features):
    """Print shape and basic stats for each feature array."""
    print("\n=== Feature Summary ===")
    for key in ("bubbles", "coord", "flip_verbose"):
        arr = features[key]
        # per-sample max: reduce over all axes except the sample axis (0)
        reduce_axes = tuple(range(1, arr.ndim))
        per_sample_max = arr.max(axis=reduce_axes)   # shape (N,)
        p95 = np.percentile(arr, 95)
        n_above = int((per_sample_max > p95).sum())
        n_total = len(per_sample_max)
        print(f"  {key}:")
        print(f"    shape  : {arr.shape}")
        print(f"    dtype  : {arr.dtype}")
        print(f"    min    : {arr.min():.4g}")
        print(f"    max    : {arr.max():.4g}")
        print(f"    mean   : {arr.mean():.4g}")
        print(f"    std    : {arr.std():.4g}")
        nz = np.count_nonzero(arr)
        total = arr.size
        print(f"    nonzero: {nz}/{total} ({100*nz/total:.1f}%)")
        print(f"    p95 (all values)      : {p95:.4g}")
        print(f"    seqs with max > p95   : {n_above}/{n_total} ({100*n_above/n_total:.1f}%)")
    print(f"\n  N sequences loaded: {len(features['indices'])}")
    print(f"  Index range: {features['indices'][0]} .. {features['indices'][-1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect DNA breathing pkl features."
    )
    parser.add_argument(
        "experiment", nargs="?", default=None,
        help=(
            "Experiment directory name (e.g. C2H2_BCL6B_TGCGGG20NGA_TTTCTAGGAA_10_3) "
            "or full path to the directory. Not required when using --boxplot."
        ),
    )
    parser.add_argument(
        "--max-seqs", type=int, default=None,
        help="Maximum number of sequences to load per TF (default: all).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only print summary stats, do not return the full array.",
    )
    parser.add_argument(
        "--seq-idx", type=int, default=None,
        help="Inspect a single sequence pkl file by its integer index (e.g. 42).",
    )
    parser.add_argument(
        "--boxplot", action="store_true",
        help=(
            "Plot boxplots of bubbles and flip distributions across all TF datasets. "
            "Does not require an experiment argument."
        ),
    )
    parser.add_argument(
        "--bubble-proportion", action="store_true",
        help=(
            "Plot bar chart of bubble-positive sequence proportion per TF. "
            "Does not require an experiment argument."
        ),
    )
    parser.add_argument(
        "--breathing-vs-mlr", metavar="MLR_FILE",
        help=(
            "Plot scatter of mean breathing feature vs MLR R². "
            "Provide path to an MLR summary file. Does not require an experiment argument."
        ),
    )
    parser.add_argument(
        "--feature", choices=["bubbles", "flip"], default="bubbles",
        help="Breathing feature to use for --breathing-vs-mlr (default: bubbles).",
    )
    parser.add_argument(
        "--agg", choices=["max", "mean"], default="mean",
        help="Aggregation over positions per sequence (default: mean).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file path for the boxplot figure (default: boxplots_<agg>.png).",
    )
    args = parser.parse_args()

    # Boxplot mode: no experiment argument needed
    if args.boxplot:
        plot_all_tf_boxplots(agg=args.agg, max_seqs=args.max_seqs, out_path=args.out)
        return

    if args.bubble_proportion:
        plot_bubble_positive_proportion(max_seqs=args.max_seqs, out_path=args.out)
        return

    if args.breathing_vs_mlr:
        plot_breathing_vs_mlr(
            mlr_file=args.breathing_vs_mlr,
            feature=args.feature,
            agg=args.agg,
            max_seqs=args.max_seqs,
            out_path=args.out,
        )
        return

    if args.experiment is None:
        parser.error("experiment is required unless --boxplot is specified")

    # Resolve experiment directory
    exp_path = Path(args.experiment)
    if not exp_path.is_absolute():
        exp_path = Path(PKL_BASE) / args.experiment
    if not exp_path.is_dir():
        sys.exit(f"Error: directory not found: {exp_path}")

    # Single-sequence inspection mode
    if args.seq_idx is not None:
        pkl_path = exp_path / f"{args.seq_idx}.pkl"
        if not pkl_path.exists():
            sys.exit(f"Error: {pkl_path} not found")
        inspect_single(pkl_path)
        return

    # Bulk load
    features = load_all(exp_path, max_seqs=args.max_seqs)
    print_summary(features)

    if not args.summary_only:
        # Drop into interactive namespace if run directly
        print("\nArrays available as: features['bubbles'], features['coord'], features['flip_verbose']")
        print("Example: features['bubbles'].shape ->", features["bubbles"].shape)

    return features


if __name__ == "__main__":
    main()
