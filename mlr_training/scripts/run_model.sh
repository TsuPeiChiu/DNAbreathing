#!/bin/bash

DATA_DIR="${DNABREATHING_DATA:-data}"

python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features 1mer
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features breathing
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features 1mer breathing
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features 1mer breathing deepdnashape
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features deepdnashape
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features 1mer deepdnashape
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features deepdnashape_mgw
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features 1mer deepdnashape_mgw
python scripts/mlr_model.py --all "${DATA_DIR}/preprocessed_data" --features 1mer breathing deepdnashape_mgw

python scripts/summary.py models/mlr/1mer > results/summary/1mer.txt
python scripts/summary.py models/mlr/breathing > results/summary/breathing.txt
python scripts/summary.py models/mlr/1mer+breathing > results/summary/1mer+breathing.txt
python scripts/summary.py models/mlr/1mer+breathing+deepdnashape > results/summary/1mer+breathing+deepdnashape.txt
python scripts/summary.py models/mlr/deepdnashape > results/summary/deepdnashape.txt
python scripts/summary.py models/mlr/1mer+deepdnashape > results/summary/1mer+deepdnashape.txt
python scripts/summary.py models/mlr/deepdnashape_mgw > results/summary/deepdnashape_mgw.txt
python scripts/summary.py models/mlr/1mer+deepdnashape_mgw > results/summary/1mer+deepdnashape_mgw.txt
python scripts/summary.py models/mlr/1mer+breathing+deepdnashape_mgw > results/summary/1mer+breathing+deepdnashape_mgw.txt
