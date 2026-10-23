# Data for the DRIP experiments

Run all commands from the repository root. Raw vectors and ground truth are not
stored in Git. The experiment binary reads the paths below; keep vector row
order unchanged after generating ground truth, because neighbor IDs are zero-based
row numbers in the base file.

## Build the data utilities

Build DRIP as described in the [main README](../README.md). Exact top-1000
ground truth for the 10M-vector datasets is computed by `bin/compute_gt_mkl`,
which requires Intel MKL and OpenMP. Check that the binary was built before
starting the ground-truth step:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_drip compute_gt_mkl -j
```

## SIFT10M

Download the [SIFT10M archive from CUHK GQR](https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html).
It contains the 10M x 128 base vectors and 1K x 128 original queries in
`fvecs` format. From the repository root:

```bash
mkdir -p data/sift10m
curl --fail --location \
  --output data/sift10m/sift10m.tar.gz \
  https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/sift10m.tar.gz
tar -xzf data/sift10m/sift10m.tar.gz -C data
./bin/compute_gt_mkl --dataset=sift10m --gt_topk=1000 --gt_mkl_threads=24
```

The resulting inputs are `data/sift10m/sift10m_base.fvecs`,
`data/sift10m/sift10m_query.fvecs`, and
`data/sift10m/sift10m_groundtruth.ivecs`. DRIP replays the 1K query vectors
to serve the launcher's default 1M logical queries.

## MSMARCO10M

The preparation script downloads six source shards from the pinned
[`CohereLabs/msmarco-v2.1-embed-english-v3`](https://huggingface.co/datasets/CohereLabs/msmarco-v2.1-embed-english-v3)
revision. It takes the first 11M passage embeddings in source order, selects
1M distinct query vectors with seed `20260722`, and uses the remaining 10M
vectors as the base. The two sets are disjoint. Install the Python dependencies
and run:

```bash
python3 -m pip install -r python/requirements.txt
./script/prepare_msmarco10M_1M.sh
```

The script writes and checks:

- `data/msmarco10M-1M/msmarco10M-1M_base.fvecs` (10M x 1024)
- `data/msmarco10M-1M/msmarco10M-1M_query.fvecs` (1M x 1024)
- `data/msmarco10M-1M/msmarco10M-1M_groundtruth.ivecs` (exact top 1000)
- `data/msmarco10M-1M/split_metadata.json` (source revision and split settings)

Allow at least 100 GB of free storage for the source shards, converted files,
ground truth, and temporary files. The exact ground-truth step resumes from a
validated partial file after interruption. For this dataset, the launcher's
default 1M logical queries are all distinct.

## Binary-format input layout

The launcher also accepts existing LAION-I2I and Wiki10M benchmark vectors.
Place their inputs at these exact paths:

| Dataset | Base vectors | Query vectors | Exact top-1000 ground truth |
| --- | --- | --- | --- |
| LAION-I2I (10M x 768, 100K queries) | `data/laion10m/laion_base.fbin` | `data/laion10m/laion10m_query.fbin` | `data/laion10m/laion10m_groundtruth.ivecs` |
| Wiki10M (10M x 384, 100K skewed queries) | `data/wiki10m/wiki_base.fbin` | `data/wiki10m/wiki10m_query_skew.fbin` | `data/wiki10m/wiki10m_skew_groundtruth.ivecs` |

Each `fbin` file starts with two little-endian 32-bit integers (`rows`,
`dimensions`) followed by row-major float32 vectors. Each `ivecs` record
starts with a little-endian 32-bit neighbor count followed by that many
little-endian 32-bit, zero-based base-row IDs. Keep queries and their ground
truth in the same order. The launcher replays shorter query sets to 1M logical
queries.

## Skewed query workloads

After the SIFT10M and MSMARCO10M-1M base files are ready, generate the fixed
10K-distinct-query Zipf workloads and exact ground truth for Figure 8 with:

```bash
./script/prepare_paper_skew_queries.sh
```

The script uses one popular centroid, Zipf exponent 1.26, and seed `1234567`.
It writes the `1nearest_zipf1.26_10k` query and ground-truth files expected by
the skewed command group in `script/run_drip.sh`.
