#!/usr/bin/env python3
"""Bounded-concurrency bench runner for the cross-DSL ako_dsl_runs/ workspaces.

Runs `scripts/bench.sh` for a list of <op>/<dsl> jobs, pinning each to a GPU
(round-robin) with a concurrency cap (default 4 = one heavy op per GPU on this
3+1 RTX6000-Ada host), parses the structured COMPILED/CORRECT/RUNTIME/SPEEDUP
lines, prints a table, and writes a JSON summary.

Usage:
    python bench_dsl.py --label iter-1 relu/cuda_noptx swish/tilelang ...
    python bench_dsl.py --label iter-1 --concurrency 4 --all-activations
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

ROOT = "/home/lxt230026/MultiKernelBench"
RUNS = f"{ROOT}/ako_dsl_runs"
NGPU = 4

ACTIVATIONS = ["relu", "sigmoid", "hardsigmoid", "gelu", "swish", "elu"]
BESPOKE = ["layer_norm", "group_norm", "gather", "scatter", "cumsum", "lstm"]
DSLS = ["cuda_noptx", "cuda_unlimited", "tilelang"]

PAT = {k: re.compile(rf"^{k}:\s*(.+)$", re.M) for k in
       ["COMPILED", "CORRECT", "RUNTIME", "REF_RUNTIME", "SPEEDUP"]}


def parse(out):
    r = {}
    for k, p in PAT.items():
        m = p.search(out)
        r[k] = m.group(1).strip() if m else "?"
    return r


GPU_POOL = Queue()


def run_job(job, label):
    op, dsl = job.split("/")
    ws = f"{RUNS}/{op}/{dsl}"
    sh = f"{ws}/scripts/bench.sh"
    if not os.path.isfile(sh):
        return job, {"COMPILED": "NO_SH", "CORRECT": "?", "SPEEDUP": "?", "RUNTIME": "?", "REF_RUNTIME": "?"}, ""
    # Acquire an exclusive GPU from the pool so two heavy jobs never share a GPU
    # (static index%4 let concurrent jobs collide -> OOM on 20GB+ activations).
    gpu = GPU_POOL.get()
    try:
        return _run_on_gpu(job, gpu, label, sh, ws)
    finally:
        GPU_POOL.put(gpu)


def _run_on_gpu(job, gpu, label, sh, ws):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    args = ["bash", sh] + ([label] if label else [])
    t0 = time.time()
    try:
        p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1800)
        out = p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired:
        return job, {"COMPILED": "TIMEOUT", "CORRECT": "?", "SPEEDUP": "?", "RUNTIME": "?", "REF_RUNTIME": "?"}, ""
    r = parse(out)
    r["secs"] = round(time.time() - t0, 1)
    return job, r, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", help="<op>/<dsl> ...")
    ap.add_argument("--label", default="")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--all-activations", action="store_true")
    ap.add_argument("--all-bespoke", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    jobs = list(a.jobs)
    if a.all_activations:
        jobs += [f"{op}/{d}" for op in ACTIVATIONS for d in DSLS]
    if a.all_bespoke:
        jobs += [f"{op}/{d}" for op in BESPOKE for d in DSLS]
    # de-dup, keep order
    seen = set(); jobs = [j for j in jobs if not (j in seen or seen.add(j))]
    if not jobs:
        ap.error("no jobs")

    results = {}
    # GPU pool: exactly `concurrency` distinct GPUs, one per running job
    conc = min(a.concurrency, NGPU)
    for g in range(conc):
        GPU_POOL.put(g)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(run_job, job, a.label): job for job in jobs}
        for fut in list(futs):
            job, r, out = fut.result()
            results[job] = r

    # print table
    print(f"\n{'job':28s} {'COMPILED':9s} {'CORRECT':8s} {'RUNTIME':>9s} {'REF':>9s} {'SPEEDUP':>9s} {'s':>5s}")
    print("-" * 86)
    for job in jobs:
        r = results[job]
        print(f"{job:28s} {r['COMPILED']:9s} {r['CORRECT']:8s} {str(r['RUNTIME']):>9s} "
              f"{str(r['REF_RUNTIME']):>9s} {str(r['SPEEDUP']):>9s} {str(r.get('secs','?')):>5s}")
    nok = sum(1 for j in jobs if results[j]["CORRECT"] == "True")
    print(f"\nCORRECT: {nok}/{len(jobs)}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", a.json)
    sys.exit(0 if nok == len(jobs) else 1)


if __name__ == "__main__":
    main()
