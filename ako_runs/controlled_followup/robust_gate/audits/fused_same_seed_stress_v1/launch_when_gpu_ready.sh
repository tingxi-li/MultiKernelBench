#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
CAMPAIGN="ako_runs/controlled_followup/robust_gate/audits/fused_same_seed_stress_v1"
LOG="$ROOT/$CAMPAIGN/results/launcher.log"
LOCK="$ROOT/$CAMPAIGN/results/launcher.lock"

mkdir -p "$ROOT/$CAMPAIGN/results"
if ! ( set -o noclobber; : > "$LOCK" ) 2>/dev/null; then
  echo "launcher already active: $LOCK" >&2
  exit 2
fi
trap 'rm -f "$LOCK"' EXIT

exec >>"$LOG" 2>&1
echo "launcher_started_utc=$(date -u +%FT%TZ)"
while true; do
  if nvidia-smi -L >/dev/null 2>&1; then
    echo "gpu_driver_ready_utc=$(date -u +%FT%TZ)"
    export CUDA_VISIBLE_DEVICES=2
    export PYTHONDONTWRITEBYTECODE=1
    exec python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v1.run
  fi
  echo "gpu_driver_unavailable_utc=$(date -u +%FT%TZ); retry_seconds=60"
  sleep 60
done
