#!/usr/bin/env python3
"""
Convert split fbin.vec files to standard fbin format.

The dataset is stored in split files (base.fbin.vec.aa, base.fbin.vec.ab, etc.)
without the standard fbin header (8 bytes for N and D).
These files contain raw float32 vectors that may be split across boundaries.

Output format: Standard fbin with 8-byte header (N, D) followed by all vector data as int32.

Parameters:
- D = 256 (dimension of each vector)
- N is automatically calculated from the total bytes read
- Each element is 4 bytes (float32)
- If N > 10M, output multiple fbin files split by 10M vectors (xxx.fbin.1, xxx.fbin.2, etc.)
"""

import glob
import os
import numpy as np
import sys
import struct
from threading import Thread, Lock
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.io import write_fbin


def convert_split_to_fbin(
    split_files_pattern: str,
    output_filename: str,
    D: int = 256,
    num_threads: int = 4,
    verbose: bool = True,
    chunk_size_m: int = 10
):
    """
    Convert split fbin.vec files to standard fvecs format.

    Args:
        split_files_pattern: Glob pattern for split files (e.g., "data/mqa170m/base.fbin.vec.*")
        output_filename: Output fvecs filename
        D: Dimension of each vector (default: 256)
        num_threads: Number of threads for writing (default: 4)
        verbose: Whether to print progress information
        chunk_size_m: Chunk size in millions for splitting output (default: 10)
    """

    # Get sorted list of split files
    split_files = sorted(glob.glob(split_files_pattern))
    if not split_files:
        raise FileNotFoundError(f"No files matching pattern: {split_files_pattern}")

    if verbose:
        print(f"Found {len(split_files)} split files:")
        for f in split_files:
            file_size = os.path.getsize(f) / 1e9
            print(f"  - {f} ({file_size:.2f} GB)")

    bytes_per_vector = D * 4  # 4 bytes per float32

    # Read all split files using thread pool
    if verbose:
        print(f"\nReading {len(split_files)} split files using {num_threads} threads...")

    read_results = {}
    read_errors = []
    read_lock = Lock()

    def read_file(idx_and_path):
        """Function to read a single split file"""
        idx, split_file = idx_and_path
        try:
            with open(split_file, 'rb') as f:
                data = f.read()
            file_size = len(data)
            if verbose:
                with read_lock:
                    print(f"  Read file {idx+1}/{len(split_files)}: {os.path.basename(split_file)} ({file_size / 1e9:.2f} GB)")
            return idx, data
        except Exception as e:
            with read_lock:
                error_msg = f"File {idx+1} ({split_file}): {e}"
                read_errors.append(error_msg)
                print(f"  ERROR reading: {error_msg}", file=sys.stderr)
            return idx, None

    # Use ThreadPoolExecutor for efficient I/O parallelism
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(read_file, (idx, path)) for idx, path in enumerate(split_files)]

        for future in as_completed(futures):
            idx, data = future.result()
            if data is not None:
                read_results[idx] = data
            else:
                raise RuntimeError(f"Failed to read file {idx+1}")

    if read_errors:
        print(f"\nERROR: {len(read_errors)} files failed to read:", file=sys.stderr)
        for error in read_errors:
            print(f"  {error}", file=sys.stderr)
        raise RuntimeError("Some files failed to read")

    # Calculate total bytes and pre-allocate buffer
    total_read = sum(len(read_results[idx]) for idx in range(len(split_files)))

    # Concatenate all byte data in correct order
    # Pre-allocate buffer to avoid repeated reallocations
    byte_stream = np.empty(total_read, dtype=np.uint8)
    offset = 0
    for idx in range(len(split_files)):
        data_len = len(read_results[idx])
        byte_stream[offset:offset + data_len] = np.frombuffer(read_results[idx], dtype=np.uint8)
        offset += data_len

    read_results.clear()  # Free memory

    if verbose:
        print(f"\nTotal bytes read: {total_read:,} ({total_read / 1e9:.2f} GB)")

    # Convert byte stream to float32 array
    if verbose:
        print(f"\nConverting {total_read:,} bytes to float32 array...")

    # Create numpy array from bytes (view as float32, then reshape)
    float_array = np.frombuffer(byte_stream.tobytes(), dtype=np.float32)

    # Calculate N from actual total bytes read
    N = len(float_array) // D
    total_bytes_actual = N * bytes_per_vector

    # Reshape to (N, D) - take only the first N*D elements
    float_array = float_array[:N * D].reshape(N, D)

    if verbose:
        print(f"  Conversion Parameters:")
        print(f"    N (vectors): {N:,}")
        print(f"    D (dimension): {D}")
        print(f"    Bytes per vector: {bytes_per_vector}")
        print(f"    Total bytes: {total_bytes_actual:,} ({total_bytes_actual / 1e9:.2f} GB)")
        print(f"  Array shape: {float_array.shape}")
        print(f"  Data type: {float_array.dtype}")

    # Determine if we need to split output
    chunk_size = chunk_size_m * 1_000_000  # Convert millions to actual count
    num_chunks = (N + chunk_size - 1) // chunk_size  # Ceiling division

    if num_chunks == 1:
        # Single output file
        if verbose:
            print(f"\nWriting to fbin format: {output_filename}")
        write_fbin(output_filename, float_array)
        if verbose:
            output_size = os.path.getsize(output_filename)
            print(f"Output file size: {output_size / 1e9:.2f} GB")
            print(f"Conversion complete!")
    else:
        # Multiple output files with threading
        if verbose:
            print(f"\nN ({N:,}) > {chunk_size_m}M, splitting into {num_chunks} files")
            print(f"Using {num_threads} threads for writing...")

        write_errors = []
        write_lock = Lock()

        def write_chunk(chunk_info):
            """Function to write a single chunk to fbin file"""
            chunk_idx, chunk_data = chunk_info
            try:
                output_file = f"{output_filename}.{chunk_idx}"
                write_fbin(output_file, chunk_data)
                output_size = os.path.getsize(output_file)
                if verbose:
                    with write_lock:
                        print(f"  Written chunk {chunk_idx}: {output_file} ({output_size / 1e9:.2f} GB)")
                return chunk_idx, True, None
            except Exception as e:
                error_msg = f"Chunk {chunk_idx}: {e}"
                if verbose:
                    with write_lock:
                        print(f"  ERROR writing: {error_msg}", file=sys.stderr)
                return chunk_idx, False, error_msg

        # Prepare chunks
        chunks = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, N)
            chunk_data = float_array[start_idx:end_idx]
            chunks.append((chunk_idx, chunk_data))

        # Use ThreadPoolExecutor for efficient writing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_chunk, chunk_info) for chunk_info in chunks]

            for future in as_completed(futures):
                chunk_idx, success, error = future.result()
                if not success:
                    write_errors.append((chunk_idx, error))

        if write_errors:
            print(f"\nERROR: {len(write_errors)} chunks failed to write:", file=sys.stderr)
            for chunk_idx, error in write_errors:
                print(f"  {error}", file=sys.stderr)
            raise RuntimeError("Some chunks failed to write")

        if verbose:
            total_output_size = 0
            for chunk_idx in range(num_chunks):
                output_file = f"{output_filename}.{chunk_idx}"
                total_output_size += os.path.getsize(output_file)
            print(f"\nAll chunks written successfully!")
            print(f"Total output size: {total_output_size / 1e9:.2f} GB")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert split fbin.vec files to standard format"
    )
    parser.add_argument(
        "--split-pattern",
        default="./data/mqa170m/source/base.fbin.vec.*",
        help="Glob pattern for split files (default: ./data/mqa170m/base.fbin.vec.*)"
    )
    parser.add_argument(
        "--output",
        default="./data/mqa170m/mqa_base.fbin",
        help="Output fbin filename (default: ./data/mqa170m/mqa_base.fbin)"
    )
    parser.add_argument(
        "--D",
        type=int,
        default=128,
        help="Dimension of each vector (default: 128)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Chunk size in millions for splitting output when N > chunk_size (default: 10)"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Number of threads for reading and writing (default: 16)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    convert_split_to_fbin(
        split_files_pattern=args.split_pattern,
        output_filename=args.output,
        D=args.D,
        num_threads=args.num_threads,
        chunk_size_m=args.chunk_size,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
