#!/bin/bash
# SDPA campaigns, serialized on GPU 0.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True
for t in sdpa_cross sdpa_abstraction; do
  echo "=== $t ==="
  python driver2.py --op sdpa --jobs jobs/$t.json --gpu 0 --reps 5 \
      --tag $t --trials 50 --warmup-s 2.0 --timeout 3600
done
