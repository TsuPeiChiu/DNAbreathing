#!/usr/bin/env python3
"""
Convert HT-SELEX DNA sequences to one-hot encoding + deepDNAshape features
for multiple linear regression.

Usage:
    python onehot_encode_selex.py <input_file>
    python onehot_encode_selex.py --all

Input format (space-delimited):
    sequence score count

Output: CSV with one-hot encoded columns (pos0_A, pos0_C, pos0_G, pos0_T, ...),
        deepDNAshape feature columns (deepdnashape_pos0_MGW, ...),
        plus score and count columns.
"""

import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from config import INPUT_DIR, SHAPE_DIR, BREATHING_DIR, PREPROCESSED_DIR as OUTPUT_DIR

NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

# Intra-base features have one value per position (N values for N bp)
# Inter-base features have one value per base step (N-1 values for N bp)
INTRA_FEATURES = ["MGW", "Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening"]
INTER_FEATURES = ["Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"]
ALL_FEATURES = INTRA_FEATURES + INTER_FEATURES


def onehot_encode_sequence(seq):
    """One-hot encode a DNA sequence. Returns a flat array of length 4 * len(seq)."""
    encoding = np.zeros(4 * len(seq), dtype=np.int8)
    for i, nuc in enumerate(seq.upper()):
        if nuc in NUCLEOTIDE_MAP:
            encoding[i * 4 + NUCLEOTIDE_MAP[nuc]] = 1
    return encoding


def load_shape_features(file_stem, seq_len, num_seqs):
    """Load deepDNAshape features for a given file. Returns (DataFrame, list of col names)."""
    shape_data = {}
    shape_cols = []

    for feature in ALL_FEATURES:
        shape_file = os.path.join(SHAPE_DIR, feature, file_stem + ".txt")
        if not os.path.isfile(shape_file):
            print(f"  Warning: missing shape file {shape_file}, skipping {feature}")
            continue

        values = np.loadtxt(shape_file)
        if values.shape[0] != num_seqs:
            print(f"  Warning: {feature} has {values.shape[0]} rows, expected {num_seqs}, skipping")
            continue

        if feature in INTER_FEATURES:
            n_pos = seq_len - 1
        else:
            n_pos = seq_len

        for i in range(n_pos):
            col_name = f"deepdnashape_{feature}_pos{i}"
            shape_data[col_name] = values[:, i]
            shape_cols.append(col_name)

    return pd.DataFrame(shape_data), shape_cols


def load_breathing_features(file_stem, num_seqs):
    """
    Load raw DNA breathing pkl files for all sequences in a given experiment.

    PKL files are 1-indexed and match the row order in the input txt file:
        BREATHING_DIR/{file_stem}/{i}.pkl  for i in 1..num_seqs

    Returns a dict with stacked arrays (shape: num_seqs × ...):
        bubbles     : (N, seq_len, 20, 20)  raw bubble counts
        coord       : (N, seq_len)           cumulative bp opening distance
        flip_verbose: (N, seq_len, 5)        flip counts at 5 thresholds
    Missing pkl files are skipped and filled with NaN arrays of the inferred shape.
    Returns None if the experiment directory does not exist.
    """
    exp_dir = Path(BREATHING_DIR) / file_stem
    if not exp_dir.is_dir():
        print(f"  Warning: breathing directory not found: {exp_dir}, skipping.")
        return None

    bubbles_list = []
    coord_list = []
    flip_list = []

    ref_shapes = {}  # inferred from the first successfully loaded pkl

    for i in range(1, num_seqs + 1):
        pkl_path = exp_dir / f"{i}.pkl"
        if not pkl_path.exists():
            if ref_shapes:
                bubbles_list.append(np.full(ref_shapes["bubbles"], np.nan))
                coord_list.append(np.full(ref_shapes["coord"], np.nan))
                flip_list.append(np.full(ref_shapes["flip_verbose"], np.nan))
            else:
                bubbles_list.append(None)
                coord_list.append(None)
                flip_list.append(None)
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        b = np.asarray(data["bubbles"])
        c = np.asarray(data["coord"])
        fv = np.asarray(data["flip_verbose"])

        if not ref_shapes:
            ref_shapes["bubbles"] = b.shape
            ref_shapes["coord"] = c.shape
            ref_shapes["flip_verbose"] = fv.shape
            # back-fill any None placeholders accumulated before first valid pkl
            for j in range(len(bubbles_list)):
                bubbles_list[j] = np.full(b.shape, np.nan)
                coord_list[j] = np.full(c.shape, np.nan)
                flip_list[j] = np.full(fv.shape, np.nan)

        bubbles_list.append(b)
        coord_list.append(c)
        flip_list.append(fv)

    if not ref_shapes:
        print(f"  Warning: no valid pkl files found in {exp_dir}.")
        return None

    return {
        "bubbles":      np.stack(bubbles_list, axis=0),
        "coord":        np.stack(coord_list, axis=0),
        "flip_verbose": np.stack(flip_list, axis=0),
    }


def process_file(input_path, output_path):
    """Read a HT-SELEX txt file and write one-hot encoded CSV with deepDNAshape features."""
    sequences = []
    scores = []
    counts = []

    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            sequences.append(parts[0])
            scores.append(float(parts[1]))
            counts.append(int(parts[2]))

    if not sequences:
        print(f"  Skipping {input_path}: no valid rows found.")
        return

    seq_len = len(sequences[0])
    file_stem = Path(input_path).stem

    # One-hot encode all sequences
    onehot_matrix = np.array([onehot_encode_sequence(seq) for seq in sequences])

    # Build column names: pos0_A, pos0_C, pos0_G, pos0_T, pos1_A, ...
    col_names = [f"1mer_pos{i}_{nuc}" for i in range(seq_len) for nuc in "ACGT"]

    df = pd.DataFrame(onehot_matrix, columns=col_names)

    # Load, normalize, and append deepDNAshape features
    shape_df, shape_cols = load_shape_features(file_stem, seq_len, len(sequences))
    if not shape_df.empty:
        # Normalize shape features (x - mean) / std
        for col in shape_cols:
            col_std = shape_df[col].std()
            col_mean = shape_df[col].mean()
            if col_std > 0:
                shape_df[col] = (shape_df[col] - col_mean) / col_std
            else:
                shape_df[col] = 0.0
        df = pd.concat([df, shape_df], axis=1)

    # Load DNA breathing features
    breathing = load_breathing_features(file_stem, len(sequences))
    bubbles = breathing['bubbles'][:,26:-26, 10, 6] / 80000 # [samples, dna sequence (removed flanking), bubble length, displacement > 3.5 A]
    flip_verbose = breathing['flip_verbose'][:,26:-26,0] / 80000 # [samples, dna sequence (removed flanking), threshold of 0.5 * sqrt(2)]

    bubbles_p95 = np.percentile(bubbles, 95)
    bubbles_binary = (bubbles > bubbles_p95)  # [samples, positions], bool

    n_breathing_pos = bubbles.shape[1]

    # Normalize bubbles and flip per TF: (x - mean) / std across all sequences at each position
    def normalize_per_pos(arr):
        """Z-score normalize each position column; return 0 where std == 0."""
        mean = arr.mean(axis=0)          # shape: (n_pos,)
        std  = arr.std(axis=0)           # shape: (n_pos,)
        out  = np.where(std > 0, (arr - mean) / std, 0.0)
        return out

    bubbles_norm      = normalize_per_pos(bubbles)
    flip_verbose_norm = normalize_per_pos(flip_verbose)

    breathing_cols = (
        {f"breathing_bubbles_pos{i}": bubbles[:, i] for i in range(n_breathing_pos)} |
        {f"breathing_bubbles_binary_pos{i}": bubbles_binary[:, i] for i in range(n_breathing_pos)} |
        {f"breathing_flip_pos{i}": flip_verbose[:, i] for i in range(n_breathing_pos)} |
        {f"breathing_bubbles_norm_pos{i}": bubbles_norm[:, i] for i in range(n_breathing_pos)} |
        {f"breathing_flip_norm_pos{i}": flip_verbose_norm[:, i] for i in range(n_breathing_pos)}
    )
    df = pd.concat([df, pd.DataFrame(breathing_cols)], axis=1)

    df["score"] = scores
    df["count"] = counts

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    n_shape = len(shape_cols)
    n_onehot = len(col_names)
    n_breathing = 5 * n_breathing_pos
    print(f"  Saved {output_path} ({len(df)} rows, {seq_len}bp, {n_onehot} onehot + {n_shape} shape + {n_breathing} breathing features)")


def main():
    parser = argparse.ArgumentParser(description="One-hot encode HT-SELEX data.")
    parser.add_argument("input_file", nargs="?", help="Single input txt file to process.")
    parser.add_argument("--all", action="store_true", help="Process all txt files in the input directory.")
    args = parser.parse_args()

    if args.all:
        txt_files = sorted(Path(INPUT_DIR).glob("*.txt"))
        print(f"Processing {len(txt_files)} files from {INPUT_DIR}")
        for txt_file in txt_files:
            out_name = txt_file.stem + ".csv"
            output_path = os.path.join(OUTPUT_DIR, out_name)
            process_file(str(txt_file), output_path)
        print("Done.")
    elif args.input_file:
        input_path = args.input_file
        if not os.path.isfile(input_path):
            sys.exit(f"Error: file not found: {input_path}")
        out_name = Path(input_path).stem + ".csv"
        output_path = os.path.join(OUTPUT_DIR, out_name)
        process_file(input_path, output_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
