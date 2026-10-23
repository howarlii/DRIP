#!/usr/bin/env bash
set -uo pipefail

# Paper experiment launcher. Commands are grouped by released figure:
#   Figure 6: accumulated time / latency / recall (enabled by default)
#   Figure 7: aggregate QPS-recall (disabled by default)
#   Figure 8: Zipf-skewed workloads (disabled by default)
#
# The in-repository test_drip binary produces the DRIP/HNSW series. Plot-ready
# data for baselines implemented in other repositories is checked in under
# results/paper_data/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${DRIP_THREADS:-24}"
ENABLE_NUMA_CONCURRENCY="${ENABLE_NUMA_CONCURRENCY:-1}"
read -r -a NUMA_NODES <<< "${DRIP_NUMA_NODES:-0 1 2 3}"

if (( ${#NUMA_NODES[@]} == 0 )); then
    echo "DRIP_NUMA_NODES must contain at least one NUMA node." >&2
    exit 1
fi

paper_defaults="--M=64 --ef_build=64 --ef_assign=10 --top_k=100 --num_threads=$THREADS --theta1=800 --early_drip_tau=0.5 --random_seed=42"

# Figure 6: main DRIP experiment. Keep the largest dataset on the last NUMA
# node by default; check free memory when selecting DRIP_NUMA_NODES.
# Every dataset serves 1M logical queries. MSMARCO provides 1M distinct
# queries; shorter source query sets are replayed by test_drip.
MAIN_DRIP_COMMANDS=(
    "./bin/test_drip --dataset=sift10m --ef_base=241 --p_probe=0.025 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=1000"
    "./bin/test_drip --dataset=wiki10m --query_type=skew --ef_base=350 --p_probe=0.04 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=1000"
    "./bin/test_drip --dataset=laion10m --ef_base=768 --p_probe=0.03 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=1000"
    "./bin/test_drip --dataset=msmarco10M-1M --ef_base=276 --p_probe=0.04 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=1000"
)
MAIN_HNSW_COMMANDS=(
    "./bin/test_drip --dataset=sift10m --ef_base=241 --p_probe=0.025 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=wiki10m --query_type=skew --ef_base=350 --p_probe=0.04 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=laion10m --ef_base=768 --p_probe=0.03 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=msmarco10M-1M --ef_base=276 --p_probe=0.04 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1"
)

# Figure 7: 24-thread aggregate QPS-recall. DRIP-start/HNSW need only one
# warm-up query; DRIP-end first serves the full Figure 6 workload.
QPS_COMMANDS=(
    "./bin/test_drip --dataset=sift10m --ef_base=241 --p_probe=0.025 $paper_defaults --target_numq=1 --qps_test_qsr=1.0 --init=2 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=sift10m --ef_base=241 --p_probe=0.025 $paper_defaults --target_numq=1000000 --qps_test_qsr=1.0 --init=2 --budget=1200 --leisure_cracking_after=1000"
    "./bin/test_drip --dataset=sift10m --ef_base=241 --p_probe=0.025 $paper_defaults --target_numq=1 --qps_test_qsr=1.0 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=msmarco10M-1M --ef_base=276 --p_probe=0.04 $paper_defaults --target_numq=1 --qps_test_qsr=0.05 --init=2 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=msmarco10M-1M --ef_base=276 --p_probe=0.04 $paper_defaults --target_numq=1000000 --qps_test_qsr=0.05 --init=2 --budget=1200 --leisure_cracking_after=1000"
    "./bin/test_drip --dataset=msmarco10M-1M --ef_base=276 --p_probe=0.04 $paper_defaults --target_numq=1 --qps_test_qsr=0.05 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=laion10m --ef_base=768 --p_probe=0.03 $paper_defaults --target_numq=1 --qps_test_qsr=0.2 --init=2 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=laion10m --ef_base=768 --p_probe=0.03 $paper_defaults --target_numq=1000000 --qps_test_qsr=0.2 --init=2 --budget=1200 --leisure_cracking_after=1000"
    "./bin/test_drip --dataset=laion10m --ef_base=768 --p_probe=0.03 $paper_defaults --target_numq=1 --qps_test_qsr=0.2 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=wiki10m --query_type=skew --ef_base=350 --p_probe=0.04 $paper_defaults --target_numq=1 --qps_test_qsr=0.05 --init=2 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=wiki10m --query_type=skew --ef_base=350 --p_probe=0.04 $paper_defaults --target_numq=1000000 --qps_test_qsr=0.05 --init=2 --budget=1200 --leisure_cracking_after=1000"
    "./bin/test_drip --dataset=wiki10m --query_type=skew --ef_base=350 --p_probe=0.04 $paper_defaults --target_numq=1 --qps_test_qsr=0.05 --init=0 --budget=0 --leisure_cracking_after=-1"
)

# Figure 8: skewed 1M-query workloads (the generated source query set contains
# 10K distinct vectors and is replayed). The no-adaptive variant uses the same
# budget with fixed PID-order refinement.
SKEW_COMMANDS=(
    "./bin/test_drip --dataset=sift10m --query_type=1nearest_zipf1.26_10k --ef_base=98 --p_probe=0.01 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=sift10m --query_type=1nearest_zipf1.26_10k --ef_base=98 --p_probe=0.01 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=1000 --adaptive_refinement=false"
    "./bin/test_drip --dataset=sift10m --query_type=1nearest_zipf1.26_10k --ef_base=98 --p_probe=0.01 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=sift10m --query_type=1nearest_zipf1.26_10k --ef_base=98 --p_probe=0.01 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1 --probe_cen=true"
    "./bin/test_drip --dataset=msmarco10M-1M --query_type=1nearest_zipf1.26_10k --ef_base=102 --p_probe=0.012 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=msmarco10M-1M --query_type=1nearest_zipf1.26_10k --ef_base=102 --p_probe=0.012 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=2 --budget=1200 --leisure_cracking_after=1000 --adaptive_refinement=false"
    "./bin/test_drip --dataset=msmarco10M-1M --query_type=1nearest_zipf1.26_10k --ef_base=102 --p_probe=0.012 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1"
    "./bin/test_drip --dataset=msmarco10M-1M --query_type=1nearest_zipf1.26_10k --ef_base=102 --p_probe=0.012 $paper_defaults --target_numq=1000000 --qps_test_qsr=0 --init=0 --budget=0 --leisure_cracking_after=-1 --probe_cen=true"
)

ALL_COMMANDS=()
ALL_COMMANDS+=("${MAIN_DRIP_COMMANDS[@]}")
if [[ "${DRIP_INCLUDE_HNSW:-0}" == "1" ]]; then
    ALL_COMMANDS+=("${MAIN_HNSW_COMMANDS[@]}")
fi
# ALL_COMMANDS+=("${QPS_COMMANDS[@]}")
# ALL_COMMANDS+=("${SKEW_COMMANDS[@]}")

echo "Total commands to execute: ${#ALL_COMMANDS[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "${ALL_COMMANDS[@]}"
    exit 0
fi
if ! command -v numactl >/dev/null 2>&1; then
    echo "numactl is required for the paper's NUMA-pinned runs." >&2
    exit 1
fi
if ! command -v lscpu >/dev/null 2>&1; then
    echo "lscpu is required to select physical cores on each NUMA node." >&2
    exit 1
fi

cd "$ROOT_DIR"

run_on_node() {
    local node="$1"
    local command="$2"
    local -a argv
    local -a physical_cpus
    local cpu_list
    read -r -a argv <<< "$command"
    mapfile -t physical_cpus < <(
        lscpu -p=CPU,NODE,SOCKET,CORE |
            awk -F, -v node="$node" -v limit="$THREADS" \
                '$1 !~ /^#/ && $2 == node && !seen[$3 ":" $4]++ { print $1; if (++count == limit) exit }'
    )
    if (( ${#physical_cpus[@]} != THREADS )); then
        echo "NUMA node $node has fewer than $THREADS available physical cores." >&2
        return 1
    fi
    cpu_list=$(IFS=,; printf '%s' "${physical_cpus[*]}")
    echo "[NUMA $node] Pinned to physical CPUs $cpu_list with local memory"
    numactl --physcpubind="$cpu_list" --membind="$node" "${argv[@]}"
}

if [[ "$ENABLE_NUMA_CONCURRENCY" == "1" ]]; then
    declare -A RUNNING_NODE
    declare -A RUNNING_COMMAND

    cleanup() {
        echo "Stopping running jobs..." >&2
        for pid in "${!RUNNING_NODE[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null || true
        exit 1
    }
    trap cleanup INT TERM

    command_index=0
    for node in "${NUMA_NODES[@]}"; do
        if (( command_index >= ${#ALL_COMMANDS[@]} )); then
            break
        fi
        command="${ALL_COMMANDS[$command_index]}"
        echo "==========> [NUMA $node] Starting: $command"
        run_on_node "$node" "$command" &
        pid=$!
        RUNNING_NODE[$pid]="$node"
        RUNNING_COMMAND[$pid]="$command"
        ((command_index += 1))
    done

    while (( ${#RUNNING_NODE[@]} > 0 )); do
        if wait -n -p finished_pid; then
            exit_code=0
        else
            exit_code=$?
        fi
        node="${RUNNING_NODE[$finished_pid]}"
        command="${RUNNING_COMMAND[$finished_pid]}"
        unset 'RUNNING_NODE[$finished_pid]'
        unset 'RUNNING_COMMAND[$finished_pid]'

        if (( exit_code != 0 )); then
            echo "Command failed with exit code $exit_code on NUMA $node: $command" >&2
            cleanup
        fi
        echo "==========> [NUMA $node] Completed: $command"

        if (( command_index < ${#ALL_COMMANDS[@]} )); then
            command="${ALL_COMMANDS[$command_index]}"
            echo "==========> [NUMA $node] Starting: $command"
            run_on_node "$node" "$command" &
            pid=$!
            RUNNING_NODE[$pid]="$node"
            RUNNING_COMMAND[$pid]="$command"
            ((command_index += 1))
        fi
    done
    trap - INT TERM
else
    node="${NUMA_NODES[0]}"
    for command in "${ALL_COMMANDS[@]}"; do
        echo "==========> [NUMA $node] Running: $command"
        if ! run_on_node "$node" "$command"; then
            echo "Command failed: $command" >&2
            exit 1
        fi
    done
fi

echo "All selected paper commands completed successfully."
