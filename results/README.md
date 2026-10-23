# Reproducing the paper plots

This release keeps the three paper plotting entry points needed for the main
evaluation:

- `plot_acc_time.ipynb`: cumulative wall time, query latency, and Recall@100.
- `plot_qps.ipynb`: aggregate QPS-recall with 24 concurrent query threads.
- `plot_skew.ipynb`: skewed-workload cumulative time, latency, and graph growth.

All three plot-ready DRIP tables live under `results/drip/`. The larger
accuracy/time and skew tables are checked in as gzip archives; the notebooks
deliberately read only `.csv` files, so unpack them before plotting:

```bash
gzip -dk results/drip/*.csv.gz
```

This keeps the release compact without hiding data preparation inside the
notebooks. The small QPS table is already an uncompressed CSV.

When `script/run_drip.sh` writes new raw CSVs into `results/drip/`, the
notebooks replace the released DRIP series with the latest matching run
that contains measured `cummulative_wall_time_ms`. Selection uses the
experiment parameter signature and the timestamp embedded in the filename.
Accuracy/time and skew runs must contain the complete 1M-query workload, so
legacy CSVs, smoke runs, and interrupted runs are ignored.
External baseline series continue to come from the released plot-ready tables.
Set `DRIP_INCLUDE_HNSW=1` only when rerunning the HNSW series as well.

The checked-in Figure 6 DRIP series comes from the 1M-query rerun with 24
physical cores and local memory per NUMA node. Its cumulative time is measured
elapsed wall time, `cummulative_wall_time_ms / 1000`. The other released
series retain their existing values. Latency smoothing and prefix-average
recall are applied before plotting.

From the repository root:

```bash
python3 -m pip install -r results/requirements.txt
jupyter lab results/plot_acc_time.ipynb
```

Run `results/plot_qps.ipynb` and `results/plot_skew.ipynb` in the same
environment. The notebooks write:

- `results/figures/main_acc_time.pdf`
- `results/figures/main_qps.pdf`
- `results/figures/main_acc_time_skew.pdf`

The checked-in tables are plot-ready samples derived from the paper runs. They
preserve the displayed curves after the paper's latency smoothing and recall
aggregation. The skew table also retains graph-growth measurements used by the
adaptive-refinement comparison.
