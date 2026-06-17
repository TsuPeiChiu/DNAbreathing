#!/usr/bin/env python3
"""
Multiple Linear Regression on one-hot encoded HT-SELEX data.

Predicts the binding score from one-hot encoded DNA sequence features.

Usage:
    python mlr_model.py <csv_file>
    python mlr_model.py --all <folder>
    python mlr_model.py --all <folder> --features 1mer
    python mlr_model.py --all <folder> --features deepdnashape
    python mlr_model.py --all <folder> --features 1mer deepdnashape
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import os

from config import MODELS_DIR as RESULTS_DIR

FEATURE_PREFIXES = {
    "1mer": ["1mer_"],
    "deepdnashape": ["deepdnashape_"],
    "deepdnashape_mgw": ["deepdnashape_MGW"],
    "breathing": ["breathing_bubbles_norm", "breathing_flip_norm"],
}


def train_and_evaluate(csv_path, feature_types):
    df = pd.read_csv(csv_path)

    prefixes = [p for ft in feature_types for p in FEATURE_PREFIXES[ft]]
    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]

    if not feature_cols:
        print(f"  Skipping {Path(csv_path).stem}: no columns matching {feature_types}")
        return None
    X = df[feature_cols].values
    y = df["score"].values

    print(f"Dataset: {Path(csv_path).stem}")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Select best alpha via inner CV on full data
    alphas = np.logspace(-4, 4, 20)
    grid = GridSearchCV(
        Ridge(), param_grid={"alpha": alphas},
        scoring="r2", cv=5, n_jobs=-1,
    )
    grid.fit(X, y)
    best_alpha = grid.best_params_["alpha"]
    print(f"  Best alpha (L2): {best_alpha:.6f}")

    # 10-fold cross-validation with best alpha
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores, adj_r2_scores, mse_scores, mae_scores = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        n, p = X_test.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
        print(f"  Fold {fold}: R2={r2:.4f}, Adj_R2={adj_r2:.4f}, MSE={mse:.6f}, MAE={mae:.4f}")

    print(f"\n  Mean R2:      {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
    print(f"  Mean Adj_R2:  {np.mean(adj_r2_scores):.4f} +/- {np.std(adj_r2_scores):.4f}")
    print(f"  Mean MSE:     {np.mean(mse_scores):.6f} +/- {np.std(mse_scores):.6f}")
    print(f"  Mean MAE:     {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")

    # Train final model on all data
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X, y)

    # Save model and results
    model_type = "mlr"
    feature_tag = "+".join(sorted(feature_types))
    results_dir = os.path.join(RESULTS_DIR, model_type, feature_tag)
    os.makedirs(results_dir, exist_ok=True)
    stem = Path(csv_path).stem
    model_path = os.path.join(results_dir, f"{stem}.pkl")

    results = {
        "alpha": best_alpha,
        "dataset": stem,
        "feature_types": feature_types,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "feature_names": feature_cols,
        "cv_r2": r2_scores,
        "cv_adj_r2": adj_r2_scores,
        "cv_mse": mse_scores,
        "cv_mae": mae_scores,
        "mean_r2": np.mean(r2_scores),
        "mean_adj_r2": np.mean(adj_r2_scores),
        "mean_mse": np.mean(mse_scores),
        "mean_mae": np.mean(mae_scores),
        "model": final_model,
        "coefficients": dict(zip(feature_cols, final_model.coef_)),
        "intercept": final_model.intercept_,
    }

    with open(model_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n  Model saved to {model_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train MLR on one-hot encoded HT-SELEX data.")
    parser.add_argument("csv_file", nargs="?", help="Path to a single preprocessed CSV file.")
    parser.add_argument("--all", metavar="FOLDER", help="Process all CSV files in the given folder.")
    parser.add_argument(
        "--features", nargs="+", choices=list(FEATURE_PREFIXES.keys()),
        default=list(FEATURE_PREFIXES.keys()),
        help="Feature types to use for training (default: all). Choices: %(choices)s",
    )
    args = parser.parse_args()

    feature_types = args.features
    print(f"Using feature types: {feature_types}")

    if args.all:
        folder = Path(args.all)
        if not folder.is_dir():
            sys.exit(f"Error: directory not found: {args.all}")
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            sys.exit(f"Error: no CSV files found in {args.all}")
        print(f"Processing {len(csv_files)} datasets from {folder}\n")
        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] {csv_file.name}")
            train_and_evaluate(str(csv_file), feature_types)
            print()
        print("All done.")
    elif args.csv_file:
        if not os.path.isfile(args.csv_file):
            sys.exit(f"Error: file not found: {args.csv_file}")
        train_and_evaluate(args.csv_file, feature_types)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
