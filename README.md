# DRIP: Cracking the Proximity Graph Index for Vector Search

> A progressive ANNS index that balances fast startup with adaptive refinement.

**DRIP** (Dynamic Refinement for Improved Performance) is a novel approximate nearest neighbor search (ANNS) index that bridges the gap between quick deployment and high-quality search. Unlike traditional graph-based indices that require expensive upfront construction, DRIP starts with a lightweight hybrid structure and progressively refines itself during query processing, adapting to the actual query workload.

## Key Features

- **Instant Startup**: Minimal upfront construction cost enables immediate query serving
- **Progressive Refinement**: Index quality improves automatically as queries are processed
- **Adaptive Evolution**: Hot regions accessed by queries are refined first, cold regions later
- **Workload-Aware**: Index structure adapts to match the actual query distribution
- **Converges to HNSW**: Eventually evolves into a fully-built, high-performance graph index

## How DRIP Works

### Index Structure

DRIP builds upon the hierarchical structure of HNSW (Hierarchical Navigable Small World graphs). The key innovation is a **hybrid graph-cluster structure**:

- **Upper layers (H, H-1, ..., init+1)**: Fully constructed graph layers for coarse navigation
- **Middle layers (init, init-1, ..., 1)**: Hybrid layers containing both graph nodes and IVF-like clusters
- **Base layer (0)**: Eventually becomes a fully constructed graph

### Initialization

Instead of building the entire index upfront, DRIP:

1. Assigns each vector to layers following HNSW's exponential distribution
2. Fully constructs only the top layers (above `init` level)
3. Partially constructs lower layers using vectors from layer `init`
4. Groups remaining vectors into clusters anchored at layer `init` graph nodes

This creates a lightweight index ready for immediate query serving.

### Progressive Refinement

DRIP refines the index through **cluster splitting** during query processing:

1. **Adaptive Refinement**: Maintains visit counters for each cluster; splits hot clusters first
2. **Two-Step Splitting** (Early Dripping):
   - First step: Insert next-layer vectors into the graph when visit count reaches `τ × Θ`
   - Second step: Reassign remaining vectors when visit count reaches `Θ`
   - This prevents suboptimal assignments when the next-layer graph is incomplete
3. **Per-Query Budget**: Each query has a refinement budget to control latency
4. **Leisure Dripping**: Uses leftover budget to refine cold regions (optional)

### Query Processing

Queries traverse the multi-layer structure:

1. **Pure-graph layers**: Standard HNSW greedy search with ef=1
2. **Hybrid layers**:
   - Search graph to find candidate centroids (beam search with `efsG_l`)
   - Probe top `nprob_l` clusters
3. **Base layer**: Direct beam search with `efsG_0`

As refinement progresses, clusters become empty and layers evolve from hybrid to pure-graph, improving query efficiency.

## Getting Started

### Prerequisites

* **AVX512 support** is required
* Install dependencies:

```bash
apt install libfmt-dev libgoogle-glog-dev libgflags-dev libgtest-dev
```

### Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`Release` is also the default when `CMAKE_BUILD_TYPE` is omitted. This enables
the optimized build used for the paper performance measurements.

Source code is in [`./saqlib/index/drip/`](./saqlib/index/drip/), test code in [`./src/test_drip.cpp`](./src/test_drip.cpp), and binaries are output to `./bin/`.

### Datasets

Download and preprocess datasets following instructions in [`./data/README.md`](./data/README.md).

## Running Experiments

### Running DRIP

Basic usage:

```bash
./bin/test_drip \
  --dataset=sift10m \
  --M=64 \
  --ef_build=64 \
  --ef_assign=10 \
  --init=2 \
  --ef_base=241 \
  --p_probe=0.025 \
  --top_k=100 \
  --target_numq=1000000 \
  --num_threads=24 \
  --budget=1200 \
  --theta1=800 \
  --early_drip_tau=0.5 \
  --leisure_cracking_after=1000 \
  --random_seed=42 \
  --qps_test_qsr=0
```

### Key Parameters

**Index Construction:**
- `--M`: HNSW graph degree, paper symbol `M` (default: 64)
- `--ef_build`: HNSW construction beam width, paper symbol `ef_build` (default: 64)
- `--ef_assign`: Cluster-assignment beam width, paper symbol `ef_assign` (default: 10)
- `--init`: Initial hybrid layer, paper symbol `init` (default: 2)

**Query Processing:**
- `--ef_base`: Base-layer search beam, paper symbol `ef_base` (default: 400)
- `--p_probe`: Initial probing ratio used to derive `nprobe,init`, paper symbol `p_probe` (default: 0.06)
- `--top_k`: Number of nearest neighbors to return (default: 100)
- `--target_numq`: Number of queries to process (default: 1,000,000)

**Progressive Refinement:**
- `--budget`: Per-query refinement limit, paper symbol `budget` (default: 1200)
  - Set to 0 to disable online refinement
  - Higher values enable more aggressive refinement per query
- `--theta1`: Layer-1 cluster splitting threshold, paper symbol `Theta_1` (default: 800)
  - Thresholds for other layers are computed automatically
- `--early_drip_tau`: Early-dripping ratio, paper symbol `tau`, with `theta_l = tau * Theta_l` (default: 0.5)
- `--leisure_cracking_after`: Enable leisure dripping after N queries (default: 1000)
  - Set to -1 to disable leisure dripping

The old flags `--efc`, `--efc_ivf`, `--init_level`, `--efs_graph`,
`--efs_cpr`, `--drip_threshold`, and `--refine_budget` remain available as
deprecated aliases so existing experiment commands continue to run.

**Evaluation:**
- `--qps_test_qsr`: Query sample rate for QPS-recall curve generation (default: 1.0)
  - Set to 0 to skip QPS testing

### Understanding Results

The program outputs a CSV file to `./results/drip/` containing per-query metrics:

- `cummulative_wall_time_ms`: Construction duration plus measured elapsed
  milliseconds from the start of concurrent search through completion of this
  query. Plot this column for cumulative wall time.
- `cummulative_time_ms`: Legacy value equal to construction time multiplied
  by the thread count plus the sum of completed query latencies. It is kept
  for compatibility and is not elapsed wall time.
- `latency_ms`: Per-query latency
- `recall`: Query recall compared to ground truth
- `visited_nodes`: Number of nodes visited during search
- `dist_cmp_graph`: Distance computations in graph traversal
- `dist_cmp_ivf`: Distance computations in cluster probing
- `crack_cost`: Refinement (dripped) cost during this query
- `idx_clus_points_cnt`: Current number of vectors in clusters
- `idx_graph_points_cnt`: Current number of vectors in graphs
- `idx_construct_progress`: Overall construction progress (0.0 to 1.0)

If `-qps_test_qsr > 0`, an additional QPS-recall curve CSV is generated after the main experiment.

### Reproducing Paper Figures

The release notebooks and checked-in plot-ready data are under
[`results/`](./results/README.md). A fresh clone can redraw the archived
accumulated-time, QPS, and skewed-workload figures. The checked-in Figure 6
DRIP series uses measured cumulative wall time from the 1M-query rerun;
other checked-in series retain their released values. New raw CSVs contain
`cummulative_wall_time_ms`, and the notebooks use them when a complete
matching run is available. Generated PDFs are written to `results/figures/`.

The paper's MSMARCO experiment uses a disjoint 10M-base / 1M-query split.
Run [`script/prepare_msmarco10M_1M.sh`](./script/prepare_msmarco10M_1M.sh) to
download the pinned source shards, reproduce the deterministic split, and
compute exact top-1000 ground truth. All datasets are then run through the
common [`script/run_drip.sh`](./script/run_drip.sh) launcher. See
[`data/README.md`](./data/README.md) for storage and dependency notes.
The launcher runs the four Figure 6 DRIP datasets by default, with 1M logical
queries each. It assigns one run per NUMA node, using 24 physical cores and
memory from that node. Check free memory before setting `DRIP_NUMA_NODES`:
MSMARCO is the largest dataset and is assigned to the fourth selected node.
Set `DRY_RUN=1` to inspect the commands. Set `DRIP_INCLUDE_HNSW=1` only if
you also want to rerun the HNSW series.
The Wiki main experiment reads the `skew` query and ground-truth files selected
by `--query_type=skew`.

## Implementation Details

The DRIP implementation can be found in:
- **Core index**: [`saqlib/index/drip/drip.hpp`](./saqlib/index/drip/drip.hpp)
- **Base HNSW**: [`saqlib/index/drip/hnsw.hpp`](./saqlib/index/drip/hnsw.hpp) (DRIP extends HNSW)
- **Test harness**: [`src/test_drip.cpp`](./src/test_drip.cpp)
- **Parameter benchmarker**: [`src/drip_param_benchmarker.hpp`](./src/drip_param_benchmarker.hpp)

Key classes and methods:
- `DRIP::construct()`: Initialize the index
- `DRIP::search_knn_cracking()`: Query with progressive refinement
- `DRIP::refinement()`: Adaptive cluster splitting logic
- `DRIP::leisure_dripping()`: Background refinement of cold regions
