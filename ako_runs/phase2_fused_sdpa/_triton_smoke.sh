#!/bin/bash
# scratch driver for the triton sdpa lane smoke tests (GPU 2)
cd /home/lxt230026/MultiKernelBench/ako_runs/phase2_fused_sdpa
ALGOS="${ALGOS:-FLASH K3}"
DIMS="${DIMS:-128 256 1024}"
PAIRS="${PAIRS:-fp32:fp32 fp32:fp16 fp16:fp16}"
for a in $ALGOS; do
for d in $DIMS; do
for p in $PAIRS; do
  sd="${p%%:*}"; pd="${p##*:}"
  out=$(PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 PYTORCH_ALLOC_CONF=expandable_segments:True \
    timeout 2400 python runner2.py --op sdpa --dsl triton --variant "$a" \
    --set x_d=$d,x_sdtype=$sd,x_pdtype=$pd --trials "${TRIALS:-10}" --warmup-s "${WS:-0.5}" 2>&1)
  echo "$out" | python -c "
import sys, json
t = sys.stdin.read()
i = t.find('###JSON###')
if i < 0:
    print('$a d=$d $sd/$pd  NO-JSON: ' + t.strip().splitlines()[-1][:200]); raise SystemExit
r = json.loads(t[i+10:])
if not r.get('ok'):
    print('$a d=$d $sd/$pd  FAIL: ' + r.get('error_msg','?')[:250]); raise SystemExit
e = r.get('error') or {}
print('$a d=$d $sd/$pd  gate=%s max_abs=%.3e pctfail=%.4g  med=%.3f ms  compile=%.1fs' % (
    e.get('gate_pass'), e.get('max_abs_err', float('nan')),
    e.get('pct_elems_failing_gate', float('nan')), r['timing']['median_ms'], r['compile_s']))
"
done; done; done
