#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="msmarco10M-1M"
REPOSITORY="CohereLabs/msmarco-v2.1-embed-english-v3"
REVISION="e78737fe92ac1b783211b705c12207ca75fcc9b7"
OUTPUT_DIR="$ROOT_DIR/data/$DATASET"
SOURCE_DIR="$OUTPUT_DIR/source"
BASE_PATH="$OUTPUT_DIR/${DATASET}_base.fvecs"
QUERY_PATH="$OUTPUT_DIR/${DATASET}_query.fvecs"
GT_PATH="$OUTPUT_DIR/${DATASET}_groundtruth.ivecs"
EXPECTED_BASE_BYTES=41000000000
EXPECTED_QUERY_BYTES=4100000000
EXPECTED_GT_BYTES=4004000000
MAX_WORKERS="${HF_MAX_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

source_files=(
    passages_npy/msmarco_v2.1_doc_segmented_00.npy
    passages_npy/msmarco_v2.1_doc_segmented_01.npy
    passages_npy/msmarco_v2.1_doc_segmented_02.npy
    passages_npy/msmarco_v2.1_doc_segmented_03.npy
    passages_npy/msmarco_v2.1_doc_segmented_04.npy
    passages_npy/msmarco_v2.1_doc_segmented_05.npy
)

force_args=()
if [[ "${FORCE:-0}" == "1" ]]; then
    force_args=(--force)
fi

validate_size() {
    local path="$1"
    local expected="$2"
    local label="$3"
    local actual
    actual="$(stat -c '%s' "$path")"
    if [[ "$actual" != "$expected" ]]; then
        echo "$label has $actual bytes; expected $expected: $path" >&2
        exit 1
    fi
}

if [[ "${FORCE:-0}" != "1" && ( -e "$BASE_PATH" || -e "$QUERY_PATH" ) ]]; then
    if [[ ! -e "$BASE_PATH" || ! -e "$QUERY_PATH" ]]; then
        echo "Only one generated split file exists; inspect it before rerunning." >&2
        exit 1
    fi
    validate_size "$BASE_PATH" "$EXPECTED_BASE_BYTES" "Base fvecs"
    validate_size "$QUERY_PATH" "$EXPECTED_QUERY_BYTES" "Query fvecs"
    echo "Validated the existing disjoint 10M/1M split."
else
    if ! command -v hf >/dev/null 2>&1; then
        echo "Missing 'hf'. Install the data dependencies with:" >&2
        echo "  python3 -m pip install -r python/requirements.txt" >&2
        exit 1
    fi
    if ! "$PYTHON_BIN" -c 'import numpy' >/dev/null 2>&1; then
        echo "Missing NumPy for $PYTHON_BIN. Install it before preparing MSMARCO." >&2
        exit 1
    fi

    mkdir -p "$SOURCE_DIR"
    hf download "$REPOSITORY" "${source_files[@]}" \
        --repo-type dataset \
        --revision "$REVISION" \
        --local-dir "$SOURCE_DIR" \
        --max-workers "$MAX_WORKERS"

    "$PYTHON_BIN" "$ROOT_DIR/python/prepare_msmarco10M_1M.py" \
        --source-dir "$SOURCE_DIR/passages_npy" \
        --output-dir "$OUTPUT_DIR" \
        "${force_args[@]}"

    validate_size "$BASE_PATH" "$EXPECTED_BASE_BYTES" "Base fvecs"
    validate_size "$QUERY_PATH" "$EXPECTED_QUERY_BYTES" "Query fvecs"
fi

if [[ -e "$GT_PATH" ]]; then
    validate_size "$GT_PATH" "$EXPECTED_GT_BYTES" "Ground truth"
    echo "Validated the existing exact top-1000 ground truth."
else
    if [[ ! -x "$ROOT_DIR/bin/compute_gt_mkl" ]]; then
        echo "Missing $ROOT_DIR/bin/compute_gt_mkl." >&2
        echo "Install Intel MKL + OpenMP, then rebuild the project." >&2
        exit 1
    fi
    gt_command=(
        "$ROOT_DIR/bin/compute_gt_mkl"
        "--dataset=$DATASET"
        "--gt_topk=1000"
        "--gt_mkl_threads=${GT_MKL_THREADS:-96}"
        "--gt_query_batch=${GT_QUERY_BATCH:-4096}"
        "--gt_base_block=${GT_BASE_BLOCK:-32768}"
    )
    if command -v numactl >/dev/null 2>&1; then
        (cd "$ROOT_DIR" && exec numactl --interleave=all "${gt_command[@]}")
    else
        (cd "$ROOT_DIR" && exec "${gt_command[@]}")
    fi
    validate_size "$GT_PATH" "$EXPECTED_GT_BYTES" "Ground truth"
fi

echo "MSMARCO10M-1M data and exact ground truth are ready."
echo "Run the paper experiments with: ./script/run_drip.sh"
