##################
# Main and Supp Figures
##################

python scripts/plot_comparison.py results/summary/1mer.txt results/summary/1mer+breathing.txt --metric r2
python scripts/plot_comparison.py results/summary/1mer+deepdnashape.txt results/summary/1mer+breathing.txt --metric r2
python scripts/plot_comparison.py results/summary/1mer+deepdnashape.txt results/summary/1mer+breathing+deepdnashape.txt --metric r2
python scripts/plot_comparison.py results/summary/1mer+breathing.txt results/summary/1mer+breathing+deepdnashape.txt --metric r2

python scripts/plot_comparison.py results/summary/deepdnashape.txt results/summary/breathing.txt --metric r2
python scripts/plot_comparison.py results/summary/deepdnashape_mgw.txt results/summary/breathing.txt --metric r2
python scripts/plot_comparison.py results/summary/1mer+deepdnashape_mgw.txt results/summary/1mer+breathing.txt --metric r2

python scripts/plot_comparison.py results/summary/1mer.txt results/summary/breathing.txt --metric r2
python scripts/plot_comparison.py results/summary/1mer+deepdnashape_mgw.txt results/summary/1mer+breathing+deepdnashape_mgw.txt --metric r2
python scripts/plot_comparison.py results/summary/1mer.txt results/summary/1mer+deepdnashape.txt --metric r2
python scripts/plot_comparison.py results/summary/1mer.txt results/summary/deepdnashape.txt --metric r2

