#!/usr/bin/env python3
"""
Summarize MLR results from saved .pkl files.

Usage:
    python summary.py <results_dir>

Example:
    python summary.py models/mlr/1mer
"""

import argparse
import sys
import pickle
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize MLR results.")
    parser.add_argument("results_dir", help="Directory containing .pkl result files.")
    args = parser.parse_args()

    folder = Path(args.results_dir)
    if not folder.is_dir():
        sys.exit(f"Error: directory not found: {args.results_dir}")

    pkl_files = sorted(folder.glob("*.pkl"))
    if not pkl_files:
        sys.exit(f"Error: no .pkl files found in {args.results_dir}")

    all_r2, all_adj_r2, all_mae = [], [], []

    print(f"{'Dataset':<50} {'Mean R2':>10} {'Mean Adj_R2':>13} {'Mean MAE':>10} {'|coef| bubbles':>16} {'|coef| flip':>13}")
    print("-" * 134)

    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)

        r2 = results["mean_r2"]
        adj_r2 = results.get("mean_adj_r2", float("nan"))
        mae = results["mean_mae"]
        name = results.get("dataset", pkl_file.stem)

        all_r2.append(r2)
        all_adj_r2.append(adj_r2)
        all_mae.append(mae)

        coefficients = results.get("coefficients", {})
        bubbles_vals = [abs(v) for k, v in coefficients.items() if k.startswith("breathing_bubbles")]
        flip_vals    = [abs(v) for k, v in coefficients.items() if k.startswith("breathing_flip")]
        bubbles_str = f"{np.mean(bubbles_vals):>16.6f}" if bubbles_vals else f"{'N/A':>16}"
        flip_str    = f"{np.mean(flip_vals):>13.6f}"    if flip_vals    else f"{'N/A':>13}"

        adj_r2_str = f"{adj_r2:>13.4f}" if not np.isnan(adj_r2) else f"{'N/A':>13}"
        print(f"{name:<50} {r2:>10.4f} {adj_r2_str} {mae:>10.4f} {bubbles_str} {flip_str}")

    print("-" * 134)
    adj_r2_vals = [v for v in all_adj_r2 if not np.isnan(v)]
    adj_r2_avg_str = f"{np.mean(adj_r2_vals):>13.4f}" if adj_r2_vals else f"{'N/A':>13}"
    print(f"{'Average (' + str(len(pkl_files)) + ' datasets)':<70} {np.mean(all_r2):>10.4f} {adj_r2_avg_str} {np.mean(all_mae):>10.4f}")


if __name__ == "__main__":
    main()
