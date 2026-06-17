python scripts/plot_comparison_multi.py \
    --panel results/summary/1mer.txt results/summary/1mer+breathing.txt \
    --panel results/summary/1mer+deepdnashape_mgw.txt results/summary/1mer+breathing.txt \
    --panel results/summary/1mer+deepdnashape.txt results/summary/1mer+breathing.txt \
    --panel results/summary/1mer+deepdnashape.txt results/summary/1mer+breathing+deepdnashape.txt \
    --out results/figures/fig_4.png

python scripts/plot_comparison_multi.py \
    --panel results/summary/1mer.txt results/summary/breathing.txt \
    --panel results/summary/deepdnashape.txt results/summary/breathing.txt \
    --panel results/summary/deepdnashape_mgw.txt results/summary/breathing.txt \
    --panel results/summary/1mer+deepdnashape_mgw.txt results/summary/1mer+breathing+deepdnashape_mgw.txt \
    --out results/figures/fig_5.png

python scripts/plot_comparison_multi.py \
    --panel results/summary/1mer.txt results/summary/1mer+deepdnashape_mgw.txt \
    --panel results/summary/1mer.txt results/summary/1mer+deepdnashape.txt \
    --panel results/summary/1mer+breathing.txt results/summary/1mer+breathing+deepdnashape.txt \
    --out results/figures/fig_s7.png