#!/bin/bash
# AKO4ALL bench wrapper — op=matmul_with_transposed_b dsl=cuda_noptx
set -eo pipefail
cd "$(dirname "$0")/.."
# GPU pin is OVERRIDABLE: orchestrator may pre-set CUDA_VISIBLE_DEVICES to fan
# benches across GPUs; fall back to the per-workspace default 0 otherwise.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="/usr/local/cuda-13.1/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
# per-workspace build dir (script already cd'd into the workspace) so the
# load_inline extensions never collide on build lock under parallel benches
export TORCH_EXTENSIONS_DIR="$(pwd)/.torch_ext"

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

set +e
python /home/lxt230026/MultiKernelBench/AKO4ALL/bench/kernelbench/bench.py --ref /home/lxt230026/MultiKernelBench/reference/matmul/matmul_with_transposed_b.py --solution solution/matmul_with_transposed_b.py --num-warmup 200 --verbose 2>&1 | tee _bench_output.txt
BENCH_EXIT=$?
set -e

if [ -n "$LABEL" ]; then TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"; else TRAJ_DIR="trajectory/${TIMESTAMP}"; fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"
exit $BENCH_EXIT
