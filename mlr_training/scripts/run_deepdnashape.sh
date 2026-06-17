#!/bin/bash
# Pipeline to run deepDNAshape on all HT-SELEX files for all 13 DNA shape features
# Usage: bash scripts/run_deepdnashape.sh  (run from mlr_training/)
# Requires: deepDNAshape tool available in conda env "deepdnashape" (see README)

set -euo pipefail

INPUT_DIR="${DNABREATHING_DATA:-data}/ht-selex-data-revision"
OUTPUT_DIR="${DNABREATHING_DATA:-data}/deepdnashape_results"
SEQ_DIR="${OUTPUT_DIR}/sequences"
CONDA_ENV="deepdnashape"

# All 13 DNA shape features
FEATURES=(MGW Shear Stretch Stagger Buckle ProT Opening Shift Slide Rise Tilt Roll HelT)

# Create output directories
mkdir -p "${SEQ_DIR}"
for feat in "${FEATURES[@]}"; do
    mkdir -p "${OUTPUT_DIR}/${feat}"
done

echo "=== Step 1: Extracting sequences from input files ==="
for input_file in "${INPUT_DIR}"/*.txt; do
    basename=$(basename "${input_file}" .txt)
    seq_file="${SEQ_DIR}/${basename}.seq.txt"
    if [ ! -f "${seq_file}" ]; then
        awk '{print $1}' "${input_file}" > "${seq_file}"
        echo "  Extracted sequences from ${basename} ($(wc -l < "${seq_file}") sequences)"
    else
        echo "  Sequences already extracted for ${basename}, skipping"
    fi
done

echo ""
echo "=== Step 2: Running deepDNAshape for all features ==="
total_jobs=$(( $(ls "${SEQ_DIR}"/*.seq.txt 2>/dev/null | wc -l) * ${#FEATURES[@]} ))
current_job=0

for feat in "${FEATURES[@]}"; do
    for seq_file in "${SEQ_DIR}"/*.seq.txt; do
        basename=$(basename "${seq_file}" .seq.txt)
        output_file="${OUTPUT_DIR}/${feat}/${basename}.txt"
        current_job=$((current_job + 1))

        if [ -f "${output_file}" ] && [ -s "${output_file}" ]; then
            echo "  [${current_job}/${total_jobs}] ${feat}/${basename} already exists, skipping"
            continue
        fi

        echo "  [${current_job}/${total_jobs}] Predicting ${feat} for ${basename}..."
        conda run -n "${CONDA_ENV}" deepDNAshape \
            --file "${seq_file}" \
            --feature "${feat}" \
            --output "${output_file}" \
            --batch_size 2048 \
            --gpu
    done
done

echo ""
echo "=== Pipeline complete ==="
echo "Results are in: ${OUTPUT_DIR}"
echo "Features predicted: ${FEATURES[*]}"
echo "Files processed: $(ls "${SEQ_DIR}"/*.seq.txt | wc -l)"
