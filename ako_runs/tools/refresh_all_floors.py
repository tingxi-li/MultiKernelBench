#!/usr/bin/env python3
"""
refresh_all_floors.py — agent-free 4-lane re-bench of every committed cell to
give check_gate.py a truer, same-host floor than the RESULTS.md seed.

4 GPU lanes (one cell per GPU at a time -> no OOM on the 49 GB cards); each cell
runs its OWN scripts/bench.sh (single source of truth for per-op flags). Cells
are round-robined across lanes so an op's 4 DSLs land on 4 different GPUs. The
CSV is written ONCE at the end (concurrent --write would race), and a cell that
fails to re-bench KEEPS its prior floor rather than losing it.

Usage:  python refresh_all_floors.py            # all cells in committed_baseline.csv
        python refresh_all_floors.py --ops layer_norm group_norm   # subset
"""
import argparse
import csv
import os
import re
import subprocess
import threading
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
AKO = os.path.dirname(TOOLS)
BASELINE = os.path.join(TOOLS, "committed_baseline.csv")
NGPU = 4


def run_cell(op, dsl, gpu):
    bench = os.path.join(AKO, op, dsl, "scripts", "bench.sh")
    if not os.path.exists(bench):
        return None, False, "no bench.sh"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    try:
        out = subprocess.run(
            ["bash", bench, "floor_refresh"],
            env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None, False, "timeout"
    text = out.stdout + out.stderr
    m = re.search(r"^SPEEDUP:\s*([0-9.]+)x", text, re.M)
    c = re.search(r"^CORRECT:\s*(True|False)", text, re.M)
    sp = float(m.group(1)) if m else None
    ok = (c.group(1) == "True") if c else False
    return sp, ok, text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ops", nargs="*", default=None,
                    help="restrict to these ops (default: all in the CSV)")
    args = ap.parse_args()

    with open(BASELINE) as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    cells = [r for r in rows if args.ops is None or r["op"] in args.ops]

    lanes = [[] for _ in range(NGPU)]
    for i, r in enumerate(cells):
        lanes[i % NGPU].append(r)

    results = {}
    lock = threading.Lock()
    t_start = time.time()

    def worker(lane, gpu):
        for r in lane:
            op, dsl = r["op"], r["dsl"]
            t0 = time.time()
            sp, ok, _ = run_cell(op, dsl, gpu)
            with lock:
                results[(op, dsl)] = (sp, ok)
                tag = "OK  " if (ok and sp) else "FAIL"
                print(f"[gpu{gpu}] {tag} {op}/{dsl}: "
                      f"{sp if sp else '-'}x correct={ok} ({time.time()-t0:.0f}s)",
                      flush=True)

    threads = [threading.Thread(target=worker, args=(lanes[g], g))
               for g in range(NGPU)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # write once; keep prior floor on failure
    n_ok = 0
    for r in rows:
        key = (r["op"], r["dsl"])
        if key not in results:
            continue
        sp, ok = results[key]
        if sp and ok:
            r["committed_speedup"] = f"{sp:.4f}"
            r["source"] = "rebench"
            n_ok += 1
        else:
            r["source"] = r.get("source", "RESULTS.md").split("|")[0] + "|rebench-failed"
    with open(BASELINE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDONE in {time.time()-t_start:.0f}s: {n_ok}/{len(cells)} refreshed "
          f"(failures kept prior floor). -> {BASELINE}")


if __name__ == "__main__":
    main()
