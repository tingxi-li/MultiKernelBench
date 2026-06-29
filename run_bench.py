#!/usr/bin/env python3
"""Verdict-quality bench for given ops, 3-at-a-time pinned to GPU 0/1/2.
Usage: python run_bench.py [op1 op2 ...]   (default: all 12)
"""
import subprocess, re, os, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = "/home/lxt230026/MultiKernelBench"
BENCH = f"{ROOT}/AKO4ALL/bench/kernelbench/bench.py"
CAT = {"relu":"activation","sigmoid":"activation","hardsigmoid":"activation","elu":"activation",
       "gelu":"activation","swish":"activation","layer_norm":"normalization","group_norm":"normalization",
       "gather":"index","scatter":"index","cumsum":"math","lstm":"arch"}
WARM = {"gather":50, "scatter":50}  # tiny kernels: clocks won't saturate anyway

def bench(op, gpu):
    ref = f"{ROOT}/reference/{CAT[op]}/{op}.py"
    sol = f"{ROOT}/ako_runs/{op}/solution/{op}.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run(["python", BENCH, "--ref", ref, "--solution", sol,
                        "--num-perf-trials", "100", "--num-warmup", str(WARM.get(op,200)),
                        "--num-correct-trials", "5"],
                       env=env, text=True, capture_output=True, timeout=900)
    out = p.stdout + p.stderr
    def g(k):
        m = re.search(rf"^{k}: (.+)$", out, re.M); return m.group(1) if m else "?"
    err = ""
    if g("CORRECT") != "True":
        me = re.search(r"(compilation_error|runtime_error|correctness_issue|OutOfMemory|excessive)[^\n]*", out)
        if me: err = "  | " + me.group(0)[:140]
    return f"{op:12s} gpu{gpu}  COMPILED={g('COMPILED'):5s} CORRECT={g('CORRECT'):5s} SPEEDUP={g('SPEEDUP'):9s} RUN={g('RUNTIME'):8s} REF={g('REF_RUNTIME')}{err}"

ops = sys.argv[1:] or list(CAT)
waves = [ops[i:i+3] for i in range(0, len(ops), 3)]
for wave in waves:
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(bench, op, j) for j, op in enumerate(wave)]
        for f in futs: print(f.result(), flush=True)
print("=== DONE ===")
