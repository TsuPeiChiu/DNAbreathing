"""
Plot runtime benchmarks for pyDNA_EPBD and deepDNAshape.

Reads:
  benchmarks/pyDNA_EPBD/results/benchmark_results.csv
  benchmarks/deepDNAshape/results/benchmark_results.csv

Plots time per sequence (s) vs. sequence length (bp), with a log best-fit
curve as a dashed line for each tool.

Usage:
  python benchmarks/plot_runtime.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EPBD_CSV     = os.path.join(SCRIPT_DIR, "pyDNA_EPBD",  "results", "benchmark_results.csv")
DEEP_CSV     = os.path.join(SCRIPT_DIR, "deepDNAshape", "results", "benchmark_results.csv")
OUTPUT_PNG   = os.path.join(SCRIPT_DIR, "runtime_comparison.png")

N_SEQS = 50   # sequences per benchmark run (see generate_inputs.py)

# ── Load data ─────────────────────────────────────────────────────────────────
epbd = pd.read_csv(EPBD_CSV)
deep = pd.read_csv(DEEP_CSV)

epbd["time_per_seq"] = epbd["wall_time_s"] / N_SEQS
deep["time_per_seq"] = deep["wall_time_s"] / N_SEQS

# ── Log fit: y = a * ln(x) + b ────────────────────────────────────────────────
def log_model(x, a, b):
    return a * x + b

x_fit = np.linspace(5, 50, 300)

epbd_params, _ = curve_fit(log_model, epbd["seq_length"], epbd["time_per_seq"])
deep_params, _ = curve_fit(log_model, deep["seq_length"], deep["time_per_seq"])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

colors = {"epbd": "#2196F3", "deep": "#F44336"}

# Data points + solid line
ax.plot(epbd["seq_length"], epbd["time_per_seq"],
        marker="x", linestyle='--', color=colors["epbd"], label="pyDNA-EPBD", linewidth=1.5)
ax.plot(deep["seq_length"], deep["time_per_seq"],
        marker="x", linestyle='--', color=colors["deep"], label="Deep DNAshape", linewidth=1.5)

# Best-fit dashed lines
# ax.plot(x_fit, log_model(x_fit, *epbd_params),
#         color=colors["epbd"], linestyle="--", linewidth=1.5,
#         label=f"pyDNA_EPBD fit: {epbd_params[0]:.2f}·ln(x) + {epbd_params[1]:.2f}")
# ax.plot(x_fit, log_model(x_fit, *deep_params),
#         color=colors["deep"], linestyle="--", linewidth=1.5,
#         label=f"deepDNAshape fit: {deep_params[0]:.2f}·ln(x) + {deep_params[1]:.2f}")

ax.set_xlabel("Sequence length (bp)", fontsize=12)
ax.set_ylabel("Average runtime per sequence (s)", fontsize=12)
# ax.set_title("Runtime vs. Sequence Length", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"Saved plot to: {OUTPUT_PNG}")
