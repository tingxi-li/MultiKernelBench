#!/usr/bin/env python3
"""Re-bench the 4 anti-hack-fixed ops, each on its own GPU."""
import subprocess, re, os
from concurrent.futures import ThreadPoolExecutor
ROOT = "/home/lxt230026/MultiKernelBench"
BENCH = f"{ROOT}/AKO4ALL/bench/kernelbench/bench.py"
CAT = {"layer_norm":"normalization","group_norm":"normalization","cumsum":"math","lstm":"arch"}
JOBS = [("layer_norm",0),("group_norm",1),("cumsum",2),("lstm",3)]
def bench(op, gpu):
    ref = f"{ROOT}/reference/{CAT[op]}/{op}.py"
    sol = f"{ROOT}/ako_runs/{op}/solution/{op}.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run(["python", BENCH, "--ref", ref, "--solution", sol,
                        "--num-perf-trials","100","--num-warmup","200","--num-correct-trials","5"],
                       env=env, text=True, capture_output=True, timeout=900)
    out = p.stdout + p.stderr
    def g(k):
        m = re.search(rf"^{k}: (.+)$", out, re.M); return m.group(1) if m else "?"
    err=""
    if g("CORRECT")!="True":
        me=re.search(r"(compilation_error|runtime_error|correctness_issue|OutOfMemory|excessive|Traceback|Error)[^\n]*", out)
        if me: err="  | "+me.group(0)[:200]
    return f"{op:12s} gpu{gpu} COMPILED={g('COMPILED'):5s} CORRECT={g('CORRECT'):6s} SPEEDUP={g('SPEEDUP'):9s} RUN={g('RUNTIME'):8s} REF={g('REF_RUNTIME')}{err}"
with ThreadPoolExecutor(max_workers=4) as ex:
    futs=[ex.submit(bench,op,gpu) for op,gpu in JOBS]
    for f in futs: print(f.result(), flush=True)
