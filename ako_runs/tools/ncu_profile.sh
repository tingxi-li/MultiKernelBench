#!/bin/bash
# ncu_profile.sh <workspace-dir>   e.g. ako_runs/layer_norm/cuda_noptx
#
# Profiles the workspace's CURRENT solution/ under Nsight Compute and prints a
# direction-picking report (bytes/passes/structural). This is the SEPARATE,
# expensive path — run it at BASELINE and at STALLS, never every iteration
# (per-iter direction from RUNTIME; ncu only when the loop needs to know *why*).
#
# Replay mode is per-op: layer_norm/group_norm need cross-launch L2 reuse, which
# a per-kernel cache flush would destroy -> application replay + --cache-control
# none. Everything else -> kernel replay (default) + --cache-control all (cold
# before each pass = the bench's per-trial L2 clear).
set -eo pipefail

WS="$(realpath "$1")"
[ -d "$WS" ] || { echo "usage: ncu_profile.sh <workspace-dir>"; exit 1; }
DSL="$(basename "$WS")"
OP="$(basename "$(dirname "$WS")")"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
NCU="${NCU:-/usr/local/cuda-13.1/bin/ncu}"

# --- env (mirrors bench.sh; GPU pin is overridable by an orchestrator) --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="/usr/local/cuda-13.1/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export TORCH_EXTENSIONS_DIR="$WS/.torch_ext"   # reuse cached build -> no recompile

# --- derive ref/solution from the committed bench.sh (single source of truth) -
BENCH="$WS/scripts/bench.sh"
REF="$(grep -oP -- '--ref \K[^ ]+' "$BENCH")"
SOL="$(grep -oP -- '--solution \K[^ ]+' "$BENCH")"   # e.g. solution/<op>.py (rel)
[ -n "$REF" ] && [ -n "$SOL" ] || { echo "could not parse --ref/--solution from $BENCH"; exit 1; }

# --- per-op replay mode -------------------------------------------------------
if [[ "$OP" == "layer_norm" || "$OP" == "group_norm" ]]; then
    MODE=(--replay-mode application --cache-control none)
    MODE_NOTE="application-replay + cache-control none (cross-launch L2 reuse preserved)"
else
    MODE=(--cache-control all)   # kernel replay is ncu default
    MODE_NOTE="kernel-replay + cache-control all (cold before each pass)"
fi

METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"

TRAJ="$WS/trajectory/$(date +%Y%m%d_%H%M%S)_ncu"
mkdir -p "$TRAJ"
echo "[ncu_profile] op=$OP dsl=$DSL gpu=$CUDA_VISIBLE_DEVICES"
echo "[ncu_profile] mode: $MODE_NOTE"
echo "[ncu_profile] -> $TRAJ"

cd "$WS"
set +e
timeout 1200 "$NCU" --profile-from-start off --target-processes all "${MODE[@]}" \
    --metrics "$METRICS" --csv --log-file "$TRAJ/ncu.csv" \
    python "$TOOLS/ncu_driver.py" run \
        --ref "$REF" --solution "$WS/$SOL" \
        --build-dir "$WS/.torch_ext" --meta-out "$TRAJ/meta.json"
NCU_EXIT=$?
set -e
if [ $NCU_EXIT -ne 0 ]; then
    echo "[ncu_profile] WARNING: ncu exited $NCU_EXIT (see $TRAJ/ncu.csv)"
fi

python "$TOOLS/ncu_driver.py" parse --csv "$TRAJ/ncu.csv" --meta "$TRAJ/meta.json" \
    | tee "$TRAJ/report.txt"
echo "[ncu_profile] report saved: $TRAJ/report.txt"
