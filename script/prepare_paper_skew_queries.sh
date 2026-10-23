#!/usr/bin/env bash
set -Eeuo pipefail

# Generate the paper's deterministic 10K-distinct-query Zipf workloads and
# exact top-1000 ground truth for SIFT10M and MSMARCO10M-1M.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUERY_TYPE="1nearest_zipf1.26_10k"
THREADS="${SKEW_THREADS:-96}"
SEED="${SKEW_RANDOM_SEED:-1234567}"

if [[ ! -x "$ROOT_DIR/bin/query_generator_nearest" ]]; then
    echo "Missing $ROOT_DIR/bin/query_generator_nearest. Build the project first." >&2
    exit 1
fi
if [[ ! -x "$ROOT_DIR/bin/compute_gt_mkl" ]]; then
    echo "Missing $ROOT_DIR/bin/compute_gt_mkl. Install MKL/OpenMP and rebuild." >&2
    exit 1
fi

prepare_dataset() {
    local dataset="$1"
    local expected_query_bytes="$2"
    local data_dir="$ROOT_DIR/data/$dataset"
    local base_path="$data_dir/${dataset}_base.fvecs"
    local query_path="$data_dir/${dataset}_query_${QUERY_TYPE}.fvecs"
    local gt_path="$data_dir/${dataset}_${QUERY_TYPE}_groundtruth.ivecs"

    if [[ ! -s "$base_path" ]]; then
        echo "Missing base vectors: $base_path" >&2
        exit 1
    fi

    if [[ -e "$query_path" ]]; then
        local actual_query_bytes
        actual_query_bytes="$(stat -c '%s' "$query_path")"
        if [[ "$actual_query_bytes" != "$expected_query_bytes" ]]; then
            echo "Unexpected skew-query size ($actual_query_bytes bytes): $query_path" >&2
            exit 1
        fi
        echo "Validated existing skew queries: $query_path"
    else
        "$ROOT_DIR/bin/query_generator_nearest" \
            "--data_path=$base_path" \
            "--query_path=$query_path" \
            --num_centers=1 \
            --query_num=10000 \
            --zipf_alpha=1.26 \
            "--random_seed=$SEED" \
            "--num_threads=$THREADS"
    fi

    if [[ -e "$gt_path" ]]; then
        local actual_gt_bytes
        actual_gt_bytes="$(stat -c '%s' "$gt_path")"
        if [[ "$actual_gt_bytes" != "40040000" ]]; then
            echo "Unexpected skew ground-truth size ($actual_gt_bytes bytes): $gt_path" >&2
            exit 1
        fi
        echo "Validated existing skew ground truth: $gt_path"
    else
        gt_command=(
            "$ROOT_DIR/bin/compute_gt_mkl"
            "--dataset=$dataset"
            "--query_type=$QUERY_TYPE"
            "--gt_topk=1000"
            "--gt_mkl_threads=${GT_MKL_THREADS:-$THREADS}"
            "--gt_query_batch=${GT_QUERY_BATCH:-4096}"
            "--gt_base_block=${GT_BASE_BLOCK:-32768}"
        )
        if command -v numactl >/dev/null 2>&1; then
            (cd "$ROOT_DIR" && exec numactl --interleave=all "${gt_command[@]}")
        else
            (cd "$ROOT_DIR" && exec "${gt_command[@]}")
        fi
    fi
}

# fvecs record sizes: (128 + 1) * 4 and (1024 + 1) * 4 bytes.
prepare_dataset "sift10m" "5160000"
prepare_dataset "msmarco10M-1M" "41000000"

echo "Paper skew workloads are ready."
