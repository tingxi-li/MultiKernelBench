#!/bin/bash
# AKO4ALL bench wrapper — op=conv_depthwise_separable_2d dsl=tilelang
set -eo pipefail
cd "$(dirname "$0")/.."
# GPU pin is OVERRIDABLE: orchestrator may pre-set CUDA_VISIBLE_DEVICES to fan
# benches across GPUs; fall back to the per-workspace default 2 otherwise.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

set +e
python /home/lxt230026/MultiKernelBench/AKO4ALL/bench/kernelbench/bench.py --ref /home/lxt230026/MultiKernelBench/reference/convolution/conv_depthwise_separable_2d.py --solution solution/conv_depthwise_separable_2d.py --num-warmup 200 --verbose 2>&1 | tee _bench_output.txt
BENCH_EXIT=$?
set -e

if [ -n "$LABEL" ]; then TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"; else TRAJ_DIR="trajectory/${TIMESTAMP}"; fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"
exit $BENCH_EXIT
