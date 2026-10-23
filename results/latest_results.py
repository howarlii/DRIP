"""Load released plot tables and overlay the latest complete DRIP runs.

The experiment binary writes timestamped raw CSVs into ``results/drip``.  The
helpers below match the exact release parameter signatures, reject incomplete
main runs, and use the timestamp embedded in the filename.  This keeps a newer
smoke run or an unrelated parameter sweep from silently replacing a paper run.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_THREADS = 24
TIMESTAMP_RE = re.compile(r"\.v\d+\.(\d{8}_\d{6})(?:\.qps)?\.csv$")

MAIN_DATASETS = {
    "MSMARCO": ("msmarco10M-1M", "276", "0.04", 1_000_000),
    "SIFT": ("sift10m", "241", "0.025", 1_000_000),
    "Laion-I2I": ("laion10m", "768", "0.03", 1_000_000),
    "Wiki": ("wiki10m_skew", "350", "0.04", 1_000_000),
}

SKEW_DATASETS = {
    "MSMARCO": ("msmarco10M-1M", "102", "0.012", 1_000_000),
    "SIFT": ("sift10m", "98", "0.01", 1_000_000),
}
SKEW_QUERY_TYPE = "1nearest_zipf1.26_10k"


def _load_released_table(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `gzip -dk results/drip/*.csv.gz` from the "
            "repository root before executing the notebook."
        )
    return pd.read_csv(path)


def _stem(
    slug: str,
    ef_base: str,
    p_probe: str,
    *,
    init: int,
    budget: int,
    leisure: int,
    query_type: str | None = None,
    suffix: str = "",
) -> str:
    dataset = slug if query_type is None else f"{slug}_{query_type}"
    return (
        f"{dataset}_M64_efbuild64_efassign10_topk100"
        f"_efbase{ef_base}_pprobe{p_probe}_init{init}"
        f"_theta1800_budget{budget}_tau0.5_leisurec{leisure}{suffix}"
    )


def _timestamp(path: Path) -> str:
    match = TIMESTAMP_RE.search(path.name)
    return match.group(1) if match else ""


def _candidates(data_dir: Path, stem: str, *, qps: bool = False) -> list[Path]:
    paths = []
    for path in data_dir.glob(f"{stem}_th{PAPER_THREADS}.v*.csv"):
        if path.name.endswith(".qps.csv") == qps and _timestamp(path):
            paths.append(path)
    return sorted(paths, key=_timestamp, reverse=True)


def _row_count(path: Path) -> int:
    with path.open("rb") as csv_file:
        last_line = -1
        for last_line, _ in enumerate(csv_file):
            pass
    return max(0, last_line)


def _latest_complete(data_dir: Path, stem: str, expected_rows: int) -> Path | None:
    for path in _candidates(data_dir, stem):
        with path.open(newline="") as csv_file:
            columns = next(csv.reader(csv_file), [])
        if "cummulative_wall_time_ms" not in columns:
            print(f"Ignoring legacy run without measured wall time: {path.name}")
            continue
        rows = _row_count(path)
        if rows == expected_rows:
            return path
        print(f"Ignoring incomplete run ({rows:,}/{expected_rows:,} rows): {path.name}")
    return None


def _latest_qps(data_dir: Path, stem: str) -> Path | None:
    candidates = _candidates(data_dir, stem, qps=True)
    return candidates[0] if candidates else None


def _paper_smoothed_latency(values: pd.Series, window_size: float = 0.2) -> pd.Series:
    """Apply the forward, logarithmic-window smoothing used by the paper plot."""
    series = values.reset_index(drop=True).astype(np.float64)
    if series.empty:
        return series

    smoothed = pd.Series(np.nan, index=series.index, dtype=np.float64)
    base = 4
    log_upper = max(1, int(np.ceil(np.log(len(series) + 1) / np.log(base))))
    for exponent in range(log_upper):
        left = int(base**exponent)
        right = min(int(base ** (exponent + 1)), len(series))
        if left >= right:
            continue

        interval_length = right - left
        window = max(1, int(np.ceil(interval_length * window_size)))
        desired_length = interval_length + window - 1
        segment = series.iloc[left : min(len(series), right + window - 1)].copy()
        if len(segment) < desired_length:
            padding_length = desired_length - len(segment)
            tail = segment.iloc[-window:] if len(segment) >= window else segment
            padding_value = tail.mean() if not tail.empty else 0.0
            padding = pd.Series(
                padding_value,
                index=pd.RangeIndex(segment.index[-1] + 1, segment.index[-1] + 1 + padding_length),
                dtype=np.float64,
            )
            segment = pd.concat([segment, padding])

        forward_mean = segment.iloc[::-1].rolling(window=window, min_periods=1).mean().iloc[::-1]
        smoothed.iloc[left:right] = forward_mean.iloc[:interval_length].to_numpy()

    return smoothed.fillna(series)


def _downsample(curve: pd.DataFrame, requested_points: int = 2_500) -> pd.DataFrame:
    if len(curve) <= requested_points:
        return curve
    positions = np.unique(np.geomspace(1, len(curve), requested_points).astype(np.int64) - 1)
    positions = np.unique(np.concatenate(([0], positions, [len(curve) - 1])))
    return curve.iloc[positions].reset_index(drop=True)


def _raw_curve(path: Path, *, recall: bool, graph: bool) -> pd.DataFrame:
    columns = ["query_idx", "cummulative_wall_time_ms", "latency_ms"]
    if recall:
        columns.append("recall")
    if graph:
        columns.append("idx_graph_points_cnt")
    raw = pd.read_csv(path, usecols=columns)
    if not np.array_equal(raw["query_idx"].to_numpy(), np.arange(len(raw))):
        raise ValueError(f"Noncontiguous query_idx in {path}")
    wall_time = raw["cummulative_wall_time_ms"].to_numpy(dtype=np.float64)
    if not np.isfinite(wall_time).all() or (wall_time <= 0).any() or (np.diff(wall_time) < 0).any():
        raise ValueError(f"Invalid cumulative wall time in {path}")

    curve = pd.DataFrame(
        {
            "num_queries": raw["query_idx"] + 1,
            "cumulative_wall_time_s": wall_time / 1_000,
            "query_latency_ms": _paper_smoothed_latency(raw["latency_ms"]),
        }
    )
    if recall:
        curve["recall"] = raw["recall"].expanding().mean()
    if graph:
        curve["graph_points_count"] = raw["idx_graph_points_cnt"]
    return _downsample(curve)


def _replace(curves: pd.DataFrame, dataset: str, method: str, replacement: pd.DataFrame) -> pd.DataFrame:
    keep = ~((curves["dataset"] == dataset) & (curves["method"] == method))
    return pd.concat([curves.loc[keep], replacement], ignore_index=True)


def load_acc_time_curves(data_dir: Path) -> pd.DataFrame:
    curves = _load_released_table(data_dir, "acc_time_curves.csv")
    datasets_with_new_runs = set()

    for dataset, (slug, ef_base, p_probe, expected_rows) in MAIN_DATASETS.items():
        methods = {
            "DRIP": (_stem(slug, ef_base, p_probe, init=2, budget=1200, leisure=1000), "-"),
            "HNSW": (_stem(slug, ef_base, p_probe, init=0, budget=0, leisure=-1), "--"),
        }
        for method, (stem, linestyle) in methods.items():
            path = _latest_complete(data_dir, stem, expected_rows)
            if path is None:
                continue
            replacement = _raw_curve(path, recall=True, graph=False)
            replacement.insert(0, "linestyle", linestyle)
            replacement.insert(0, "method", method)
            replacement.insert(0, "dataset", dataset)
            curves = _replace(curves, dataset, method, replacement)
            datasets_with_new_runs.add(dataset)
            print(f"Using latest {dataset} {method}: {path.name}")

    # The release launcher uses 1M normal Wiki queries.  Once a new Wiki run is
    # present, do not extend archived historical baselines to the old 10M range.
    if "Wiki" in datasets_with_new_runs:
        curves = curves[(curves["dataset"] != "Wiki") | (curves["num_queries"] <= 1_000_000)]
    return curves


def load_qps_curves(data_dir: Path) -> pd.DataFrame:
    curves = _load_released_table(data_dir, "qps_curves.csv")
    for dataset, (slug, ef_base, p_probe, _) in MAIN_DATASETS.items():
        methods = {
            "DRIP-start": _stem(slug, ef_base, p_probe, init=2, budget=0, leisure=-1),
            "DRIP-end": _stem(slug, ef_base, p_probe, init=2, budget=1200, leisure=1000),
            "HNSW": _stem(slug, ef_base, p_probe, init=0, budget=0, leisure=-1),
        }
        for method, stem in methods.items():
            path = _latest_qps(data_dir, stem)
            if path is None:
                continue
            raw = pd.read_csv(path, usecols=["avg_recall", "qps"])
            replacement = pd.DataFrame(
                {"dataset": dataset, "method": method, "recall": raw["avg_recall"], "qps": raw["qps"]}
            ).dropna()
            curves = _replace(curves, dataset, method, replacement)
            print(f"Using latest {dataset} {method}: {path.name}")
    return curves


def load_skew_curves(data_dir: Path) -> pd.DataFrame:
    curves = _load_released_table(data_dir, "skew_curves.csv")
    for dataset, (slug, ef_base, p_probe, expected_rows) in SKEW_DATASETS.items():
        methods = {
            "DRIP": (
                _stem(
                    slug,
                    ef_base,
                    p_probe,
                    init=2,
                    budget=1200,
                    leisure=-1,
                    query_type=SKEW_QUERY_TYPE,
                ),
                "-",
            ),
            "DRIP (w/o adaptive refinement)": (
                _stem(
                    slug,
                    ef_base,
                    p_probe,
                    init=2,
                    budget=1200,
                    leisure=1000,
                    query_type=SKEW_QUERY_TYPE,
                    suffix="_noadaptive",
                ),
                ":",
            ),
            "HNSW": (
                _stem(
                    slug,
                    ef_base,
                    p_probe,
                    init=0,
                    budget=0,
                    leisure=-1,
                    query_type=SKEW_QUERY_TYPE,
                ),
                "--",
            ),
            "HNSW (probe centroids)": (
                _stem(
                    slug,
                    ef_base,
                    p_probe,
                    init=0,
                    budget=0,
                    leisure=-1,
                    query_type=SKEW_QUERY_TYPE,
                    suffix="_probecen",
                ),
                "-",
            ),
        }
        for method, (stem, linestyle) in methods.items():
            path = _latest_complete(data_dir, stem, expected_rows)
            if path is None:
                continue
            replacement = _raw_curve(path, recall=False, graph=True)
            replacement.insert(0, "linestyle", linestyle)
            replacement.insert(0, "method", method)
            replacement.insert(0, "dataset", dataset)
            curves = _replace(curves, dataset, method, replacement)
            print(f"Using latest {dataset} {method}: {path.name}")
    return curves
