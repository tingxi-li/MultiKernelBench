#!/bin/bash
# Remaining fused campaigns, serialized on GPU 0. Reported timings must all come
# from one card under one protocol; the other three GPUs are used only for
# development smoke tests, never for numbers that reach a table.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.
for t in fused_native fused_epilogue fused_cast fused_abstraction; do
  echo "=== $t ==="
  python driver2.py --op fused --jobs jobs/$t.json --gpu 0 --reps 5 \
      --tag $t --trials 100 --warmup-s 2.0
done
