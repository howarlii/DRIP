#!/usr/bin/env python3
"""Create the reproducible msmarco10M-1M fvecs split from Cohere shards.

The first 11,000,000 passage embeddings in the source shard order are used.
Membership in the 1,000,000-vector query set is selected uniformly without
replacement with a fixed seed; the remaining 10,000,000 vectors are base data.
Output order follows the source order, so the split can be reproduced without
storing a large index list.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DATASET = "msmarco10M-1M"
SOURCE_REPOSITORY = "CohereLabs/msmarco-v2.1-embed-english-v3"
SOURCE_REVISION = "e78737fe92ac1b783211b705c12207ca75fcc9b7"
DEFAULT_TOTAL_VECTORS = 11_000_000
DEFAULT_QUERY_VECTORS = 1_000_000
DEFAULT_SEED = 20260722


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert a continuous 11M-vector Cohere passage range into fvecs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=root / "data" / DATASET / "source" / "passages_npy",
        help="Directory containing msmarco_v2.1_doc_segmented_*.npy shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / DATASET,
        help="Directory for the generated fvecs and metadata.",
    )
    parser.add_argument(
        "--start-vector",
        type=int,
        default=0,
        help="Zero-based source vector offset of the continuous range.",
    )
    parser.add_argument(
        "--total-vectors",
        type=int,
        default=DEFAULT_TOTAL_VECTORS,
        help="Number of consecutive source vectors to split.",
    )
    parser.add_argument(
        "--query-vectors",
        type=int,
        default=DEFAULT_QUERY_VECTORS,
        help="Number of vectors assigned to the query split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="NumPy RNG seed used for query membership.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=32_768,
        help="Rows processed at a time while converting each source shard.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated output files and stale temporary files.",
    )
    return parser.parse_args()


def write_fvec_records(output: Any, vectors: np.ndarray, dimension: int) -> None:
    """Write a batch in standard fvecs layout without per-vector Python loops."""
    if vectors.size == 0:
        return
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    records = np.empty((values.shape[0], dimension + 1), dtype=np.dtype("<i4"))
    records[:, 0] = dimension
    records[:, 1:] = values.view(np.int32)
    output.write(records.tobytes(order="C"))


def validate_fvecs(path: Path, rows: int, dimension: int) -> None:
    expected_words = rows * (dimension + 1)
    expected_bytes = expected_words * np.dtype("<i4").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes} bytes "
            f"for {rows}x{dimension} fvecs."
        )

    words = np.memmap(path, dtype="<i4", mode="r")
    try:
        if words.size != expected_words or not np.all(words[:: dimension + 1] == dimension):
            raise RuntimeError(f"{path} has an invalid fvecs dimension header.")
    finally:
        del words


def remove_if_present(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> None:
    args = parse_args()
    if args.start_vector < 0:
        raise ValueError("--start-vector must be non-negative")
    if args.total_vectors <= 0:
        raise ValueError("--total-vectors must be positive")
    if not 0 < args.query_vectors < args.total_vectors:
        raise ValueError("--query-vectors must be greater than 0 and smaller than --total-vectors")
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_paths = sorted(source_dir.glob("msmarco_v2.1_doc_segmented_*.npy"))
    if not source_paths:
        raise FileNotFoundError(f"No source .npy shards found in {source_dir}")

    base_rows = args.total_vectors - args.query_vectors
    base_path = output_dir / f"{DATASET}_base.fvecs"
    query_path = output_dir / f"{DATASET}_query.fvecs"
    metadata_path = output_dir / "split_metadata.json"
    temp_base_path = base_path.with_suffix(base_path.suffix + ".tmp")
    temp_query_path = query_path.with_suffix(query_path.suffix + ".tmp")
    generated_paths = (base_path, query_path, metadata_path, temp_base_path, temp_query_path)

    existing_paths = [path for path in generated_paths if path.exists()]
    if existing_paths and not args.force:
        names = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Output already exists: {names}. Re-run with --force to replace it.")
    if args.force:
        for path in generated_paths:
            remove_if_present(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    is_query = np.zeros(args.total_vectors, dtype=np.bool_)
    is_query[rng.choice(args.total_vectors, size=args.query_vectors, replace=False)] = True

    selection_start = args.start_vector
    selection_stop = selection_start + args.total_vectors
    source_offset = 0
    dimension: int | None = None
    written_base = 0
    written_query = 0
    source_metadata: list[dict[str, Any]] = []

    print(
        f"Selecting source vectors [{selection_start}, {selection_stop}) from {source_dir}; "
        f"seed={args.seed}, base={base_rows}, query={args.query_vectors}."
    )

    try:
        with temp_base_path.open("wb") as base_output, temp_query_path.open("wb") as query_output:
            for source_path in source_paths:
                matrix = np.load(source_path, mmap_mode="r", allow_pickle=False)
                if matrix.ndim != 2:
                    raise ValueError(f"{source_path} must be a 2D matrix, got shape {matrix.shape}")
                if dimension is None:
                    dimension = int(matrix.shape[1])
                elif matrix.shape[1] != dimension:
                    raise ValueError(
                        f"Dimension mismatch in {source_path}: expected {dimension}, got {matrix.shape[1]}"
                    )

                shard_start = source_offset
                shard_stop = shard_start + matrix.shape[0]
                overlap_start = max(selection_start, shard_start)
                overlap_stop = min(selection_stop, shard_stop)
                if overlap_start < overlap_stop:
                    local_start = overlap_start - shard_start
                    local_stop = overlap_stop - shard_start
                    source_metadata.append(
                        {
                            "file": source_path.name,
                            "source_rows": int(matrix.shape[0]),
                            "selected_rows": [int(local_start), int(local_stop)],
                        }
                    )
                    for chunk_start in range(local_start, local_stop, args.chunk_rows):
                        chunk_stop = min(chunk_start + args.chunk_rows, local_stop)
                        relative_start = overlap_start + (chunk_start - local_start) - selection_start
                        relative_stop = relative_start + (chunk_stop - chunk_start)
                        query_mask = is_query[relative_start:relative_stop]
                        chunk = matrix[chunk_start:chunk_stop]

                        query_chunk = chunk[query_mask]
                        base_chunk = chunk[~query_mask]
                        if not np.isfinite(query_chunk).all() or not np.isfinite(base_chunk).all():
                            raise ValueError(f"Non-finite vector value found in {source_path}")
                        write_fvec_records(base_output, base_chunk, dimension)
                        write_fvec_records(query_output, query_chunk, dimension)
                        written_base += len(base_chunk)
                        written_query += len(query_chunk)

                    print(
                        f"Processed {source_path.name}: source rows [{local_start}, {local_stop}); "
                        f"written base/query = {written_base}/{written_query}."
                    )

                source_offset = shard_stop
                del matrix
                if source_offset >= selection_stop:
                    break

            base_output.flush()
            query_output.flush()
            os.fsync(base_output.fileno())
            os.fsync(query_output.fileno())

        if dimension is None or source_offset < selection_stop:
            raise RuntimeError(
                f"Only {source_offset} source vectors were available; need {selection_stop}. "
                "Download the next source shard and retry."
            )
        if written_base != base_rows or written_query != args.query_vectors:
            raise RuntimeError(
                f"Split count mismatch: got {written_base} base and {written_query} query vectors."
            )

        validate_fvecs(temp_base_path, written_base, dimension)
        validate_fvecs(temp_query_path, written_query, dimension)
        os.replace(temp_base_path, base_path)
        os.replace(temp_query_path, query_path)

        metadata = {
            "dataset": DATASET,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_vector_range": [selection_start, selection_stop],
            "split_seed": args.seed,
            "dimension": dimension,
            "base_vectors": written_base,
            "query_vectors": written_query,
            "distance": "L2 squared",
            "source_shards": source_metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception:
        print(
            f"Conversion failed. Partial files, if any, were left at "
            f"{temp_base_path} and {temp_query_path} for inspection."
        )
        raise

    print(f"Wrote and validated {base_path} ({written_base}x{dimension}).")
    print(f"Wrote and validated {query_path} ({written_query}x{dimension}).")
    print(f"Wrote split metadata to {metadata_path}.")


if __name__ == "__main__":
    main()
