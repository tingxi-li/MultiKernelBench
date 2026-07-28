#!/bin/bash
# Everything that still needs GPU 0, in order, after sdpa_cross finishes.
#
# Waits on the running campaign's PID first: driver2's preflight aborts on a
# busy card, and two campaigns on one GPU would invalidate both anyway.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True

CROSS_PID="${1:-}"
if [ -n "$CROSS_PID" ]; then
  echo "=== waiting for sdpa_cross (pid $CROSS_PID) ==="
  while [ -e "/proc/$CROSS_PID" ]; do sleep 20; done
fi

# Two records were lost when runner2.py was edited mid-campaign (the driver
# spawns a fresh process per job, so it picked the file up mid-edit). Deleting
# them makes driver2 re-run exactly those: run_job() returns the cached file
# when it exists and re-runs when it does not.
echo "=== repairing 2 records damaged by a mid-campaign edit ==="
rm -fv results/sdpa_cross/raw/tilelang__FLASH__x_d256_x_pdtypefp16_x_sdtypefp16__rand__rep2.json
rm -fv results/sdpa_cross/raw/cuda_unlimited__FLASH__x_d1024_x_pdtypefp16_x_sdtypefp16__rand__rep1.json
python driver2.py --op sdpa --jobs jobs/sdpa_cross.json --gpu 0 --reps 5 \
    --tag sdpa_cross --trials 50 --warmup-s 2.0 --timeout 3600

echo "=== sdpa_abstraction ==="
python driver2.py --op sdpa --jobs jobs/sdpa_abstraction.json --gpu 0 --reps 5 \
    --tag sdpa_abstraction --trials 50 --warmup-s 2.0 --timeout 3600

# Counters only. ncu durations sit at the cold-clock transient and are never
# used as runtimes; this run is for score-tensor DRAM traffic, registers and
# occupancy per kernel, which is what the spec asks for.
echo "=== ncu per-kernel census (sdpa) ==="
python ncu_collect2.py --op sdpa --jobs jobs/ncu_sdpa.json --gpu 0

echo "=== ALL GPU WORK DONE ==="
