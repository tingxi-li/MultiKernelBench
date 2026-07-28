#!/bin/bash
# Phase-1 measurement campaigns, back to back, serialized on ONE idle GPU.
#
# Protocol frozen here (see results/stability/ for how it was chosen):
#   --warmup-s 2.0   fixed warmup TIME, not iteration count. These cards idle at
#                    210 MHz, boost, then thermally soak; per-process median for a
#                    ~1 ms GEMM rises 0.886 -> 1.023 ms as warmup goes 50 -> 1000
#                    iters. Variants differ ~4.5x in runtime, so a fixed iteration
#                    count delivers 4.5x more heat before the slow variant is
#                    measured. Fixed time equalizes that. Yields ~3.5% spread
#                    across independent processes.
#   --trials 100     cuda-event timed, L2 thrashed before each
#   --reps           independent PROCESSES; the reported number is the median of
#                    per-process medians
#   randomized order across (job x rep), fixed order-seed, reproducible
set -uo pipefail
cd "$(dirname "$0")"
GPU="${GPU:-0}"
W="--warmup-s 2.0 --trials 100"

run () {  # run <jobfile> <tag> <reps>
  echo "=================================================================="
  echo "== $2  ($(python -c "import json;print(len(json.load(open('jobs/$1.json'))))") jobs x $3 reps)  $(date +%H:%M:%S)"
  echo "=================================================================="
  python driver.py --jobs "jobs/$1.json" --gpu "$GPU" --reps "$3" --tag "$2" $W
}

# headline table first, while the machine is freshest
run matched      matched      5
# sub-studies: each contains its own matched-table twin as an internal anchor,
# so any drift between campaigns is detectable rather than assumed absent
run kc_sweep     kc_sweep     3
run casting      casting      5
run pipeline     pipeline     5
run abstraction  abstraction  5
# the H1-vs-M2 depth control. Separate campaign because tilelang_abstraction.py
# refuses an off-spec pipeline depth unless x_depth_control=1 is passed -- the
# spec pins each arm's depth and accidental drift would void the study.
run abstraction_depth abstraction_depth 5
# equal-budget native tuning: a SEARCH, so 2 reps to rank, winners re-measured after
run native_tuned native_tuned 2

echo "=================================================================="
echo "== confirm  (native-tuning winners at the full protocol)  $(date +%H:%M:%S)"
echo "=================================================================="
# A number selected as the MINIMUM over ~19 two-process medians is partly
# selection noise. Re-measure the top two per DSL at 5 processes; a rank flip
# between the two means the search resolved below the noise floor.
python confirm_winners.py --emit && run confirm confirm 5

echo "=================================================================="
echo "== compile_cost  (cold, isolated caches)  $(date +%H:%M:%S)"
echo "=================================================================="
# The `compile s` column in the timing tables is WARM-cache and inverts the true
# ordering. This is the comparable number.
python compile_cost.py --gpu "$GPU" --variants A,D

echo "ALL CAMPAIGNS DONE $(date +%H:%M:%S)"
