"""
plot_mlr_coefs.py
-----------------
Two modes:

1. Summary mode (default): plot per-TF average |coef| for bubbles and flip from a summary file.
       python plot_mlr_coefs.py <summary_file> [--out OUT]

2. Model mode: plot per-position coefficients of breathing_bubbles and breathing_flip from a .pkl model.
       python plot_mlr_coefs.py --model <model.pkl> [--out OUT]

Examples:
    python plot_mlr_coefs.py ../summary/mlr/1mer+breathing.txt
    python plot_mlr_coefs.py --model ../models/mlr/1mer+breathing/bHLH_TCF3_TACCCG20NCCC_CACCTG_12_3.pkl
"""

import argparse
import os
import pickle
import sys

import logomaker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import INPUT_DIR as HT_SELEX_DIR


def parse_summary(filepath):
    """
    Parse an MLR summary file and return lists of (label, bubbles_coef, flip_coef).

    The label is derived the same way as plot_all_tf_boxplots():
        '_'.join(dataset_name.split('_')[:4])
    """
    labels = []
    bubbles = []
    flips = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("Dataset") or line.startswith("-") or line.startswith("Average"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            dataset = parts[0]
            # columns: Dataset  Mean_R2  Mean_MAE  |coef|_bubbles  |coef|_flip
            try:
                bubbles_val = float(parts[-2])
                flip_val    = float(parts[-1])
            except ValueError:
                continue

            label = '_'.join(dataset.split('_')[:4])
            labels.append(label)
            bubbles.append(bubbles_val)
            flips.append(flip_val)

    return labels, bubbles, flips


def plot_coef_bars(labels, values, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.5), 8))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim([0.0, 0.12])
    ax.tick_params(axis="y", labelsize=12)
    ax.margins(x=0.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def load_model_breathing_coefs(pkl_path):
    """
    Load a model pkl and extract per-position coefficients for
    breathing_bubbles_binary and breathing_flip.

    Returns:
        bubbles_positions : list of int
        bubbles_coefs     : list of float
        flip_positions    : list of int
        flip_coefs        : list of float
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    coefs = data["coefficients"]

    bubbles_items = sorted(
        [(int(k.split("pos")[-1]), v) for k, v in coefs.items()
         if k.startswith("breathing_bubbles")],
        key=lambda x: x[0],
    )
    flip_items = sorted(
        [(int(k.split("pos")[-1]), v) for k, v in coefs.items()
         if k.startswith("breathing_flip")],
        key=lambda x: x[0],
    )

    bubbles_positions = [p for p, _ in bubbles_items]
    bubbles_coefs     = [v for _, v in bubbles_items]
    flip_positions    = [p for p, _ in flip_items]
    flip_coefs        = [v for _, v in flip_items]

    return bubbles_positions, bubbles_coefs, flip_positions, flip_coefs


def plot_position_coef_bars(positions, values, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.5), 5))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], fontsize=12)
    ax.set_xlabel("Base position", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.margins(x=0.02)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.90)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def compute_pwm(txt_path):
    """
    Read an HT-SELEX .txt file (sequence  score  count) and return a
    score-weighted PWM as a DataFrame with columns A, C, G, T.
    """
    seqs, weights = [], []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            seq = parts[0].upper()
            try:
                weight = float(parts[1])
            except ValueError:
                continue
            if any(b not in "ACGT" for b in seq):
                continue
            seqs.append(seq)
            weights.append(weight)

    if not seqs:
        return None

    L = len(seqs[0])
    bases = list("ACGT")
    counts = np.zeros((L, 4))
    weights = np.array(weights)

    for seq, w in zip(seqs, weights):
        for i, b in enumerate(seq):
            counts[i, bases.index(b)] += w

    # Normalise each row to sum to 1
    counts /= counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(counts, columns=bases)


def plot_pwm(txt_path, title, out_path):
    """Plot a sequence logo from an HT-SELEX file using logomaker."""
    pwm = compute_pwm(txt_path)
    if pwm is None:
        print(f"Warning: no valid sequences in {txt_path}", file=sys.stderr)
        return

    info_matrix = logomaker.transform_matrix(pwm, from_type="probability", to_type="information")

    fig, ax = plt.subplots(figsize=(max(8, len(pwm) * 0.5), 3))
    logomaker.Logo(info_matrix, ax=ax)
    ax.set_xlabel("Base position", fontsize=14)
    ax.set_ylabel("Information (bits)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.85)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MLR bubbles/flip coefficients as bar plots.")
    parser.add_argument("summary_file", nargs="?", help="Path to the MLR summary .txt file")
    parser.add_argument("--model", metavar="PKL",
                        help="Path to a model .pkl file; plots per-position breathing coefficients")
    parser.add_argument("--out", default=None,
                        help="Output prefix. Two files will be written: <prefix>_bubbles.png and <prefix>_flip.png")
    args = parser.parse_args()

    # ── Model mode ──────────────────────────────────────────────────────────
    if args.model:
        if not os.path.isfile(args.model):
            print(f"Error: file not found: {args.model}", file=sys.stderr)
            sys.exit(1)

        fname = os.path.splitext(os.path.basename(args.model))[0]
        bub_pos, bub_coefs, flip_pos, flip_coefs = load_model_breathing_coefs(args.model)

        if args.out is None:
            feature_tag = os.path.basename(os.path.dirname(os.path.abspath(args.model)))
            base = os.path.join("./plots/mlr_coefs", feature_tag, fname)
            out_bubbles = base + "_pos_bubbles.png"
            out_flip    = base + "_pos_flip.png"
        else:
            out_bubbles = args.out + "_bubbles.png"
            out_flip    = args.out + "_flip.png"

        plot_position_coef_bars(bub_pos,  bub_coefs,
                                "MLR coefficient (breathing_bubbles)",
                                fname, out_bubbles)
        plot_position_coef_bars(flip_pos, flip_coefs,
                                "MLR coefficient (breathing_flip)",
                                fname, out_flip)

        # PWM logo
        txt_path = os.path.join(HT_SELEX_DIR, fname + ".txt")
        if os.path.isfile(txt_path):
            out_pwm = os.path.join("./plots/pwm", fname + "_pwm.png")
            plot_pwm(txt_path, fname, out_pwm)
        else:
            print(f"Warning: HT-SELEX file not found, skipping PWM: {txt_path}", file=sys.stderr)
        return

    # ── Summary mode ────────────────────────────────────────────────────────
    if not args.summary_file:
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.summary_file):
        print(f"Error: file not found: {args.summary_file}", file=sys.stderr)
        sys.exit(1)

    labels, bubbles, flips = parse_summary(args.summary_file)
    if not labels:
        print("Error: no data parsed from summary file.", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.summary_file))[0]
        base = os.path.join("./plots/mlr_coefs", stem)
        out_bubbles = base + "_coef_bubbles.png"
        out_flip    = base + "_coef_flip.png"
    else:
        out_bubbles = args.out + "_bubbles.png"
        out_flip    = args.out + "_flip.png"

    plot_coef_bars(labels, bubbles, "Bubbles |coef|", "Mean MLR |coefficient| of bubbles", out_bubbles)
    plot_coef_bars(labels, flips,   "Flip |coef|",    "Mean MLR |coefficient| of flip",    out_flip)


if __name__ == "__main__":
    main()
