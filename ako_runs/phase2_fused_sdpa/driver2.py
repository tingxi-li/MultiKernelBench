#!/usr/bin/env python3
"""Phase-2 campaign driver: randomized-order, multi-process, serial on one GPU.

Same protocol as Phase 1's driver.py -- shuffled but reproducible job order, one
subprocess per (variant, repeat), a preflight that refuses to start on a busy
card -- extended with `--op`. Randomizing order matters for the same reason it
did in Phase 1: these cards ramp 210 MHz to ~2.5 GHz and drift with temperature,
so a fixed order hands whichever variant runs last a systematically different
clock state.

usage:
  python driver2.py --op fused --jobs jobs/fused_matched.json --gpu 0 --reps 5 \
      --tag fused_matched
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
import common  # noqa: E402


def job_name(job, args, rep):
    n = f"{job['dsl']}__{job['variant']}"
    if job.get("set"):
        n += "__" + job["set"].replace("=", "").replace(",", "_")
    return n + f"__{args.dist}__rep{rep}"


def run_job(job, rep, gpu, args, outdir) -> dict:
    out = os.path.join(outdir, job_name(job, args, rep) + ".json")
    if os.path.exists(out) and not args.force:
        with open(out) as f:
            return json.load(f)

    cmd = [sys.executable, os.path.join(HERE, "runner2.py"),
           "--op", args.op, "--dsl", job["dsl"], "--variant", job["variant"],
           "--dist", args.dist, "--seed", str(args.seed), "--rep", str(rep),
           "--trials", str(args.trials), "--warmup-s", str(args.warmup_s),
           "--out", out]
    if job.get("set"):
        cmd += ["--set", job["set"]]
    if args.time_only:
        cmd += ["--time-only"]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           cwd=HERE, timeout=args.timeout)
        stdout, stderr, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = f"TIMEOUT after {args.timeout}s"
        rc = -9
    dur = time.time() - t0

    rec = None
    if "###JSON###" in stdout:
        try:
            rec = json.loads(stdout.split("###JSON###", 1)[1].strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            rec = None
    if rec is None:
        rec = {"ok": False, "op": args.op, "dsl": job["dsl"],
               "variant": job["variant"], "rep": rep,
               "key": job_name(job, args, rep),
               "cfg": {"extra": _set_to_extra(job.get("set", ""))},
               "error_msg": "runner produced no JSON",
               "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:]}
        common.write_json(out, rec)
    rec["wall_s"] = dur
    rec["returncode"] = rc
    if not rec.get("ok"):
        rec.setdefault("stderr_tail", stderr[-4000:])
    return rec


def _set_to_extra(s):
    d = {}
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip().startswith("x_"):
                d[k.strip()[2:]] = v.strip()
    return d


def preflight(gpu: int, strict: bool = True):
    problems = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader", "-i", str(gpu)],
            capture_output=True, text=True, timeout=15).stdout.strip()
        others = [l for l in out.splitlines() if l.strip()]
        if others:
            problems.append(f"GPU {gpu} already has {len(others)} compute "
                            f"process(es): {others}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"could not query GPU {gpu}: {e}")

    # Phase-2 CUDA lanes build into the Phase-1 extension tree (they import its
    # modules), so that is where stale ninja locks accumulate.
    stale = []
    for extdir in (os.path.join(HERE, ".torch_ext"),
                   os.path.join(os.path.dirname(HERE), "phase1_matmul", ".torch_ext")):
        for root, _d, files in os.walk(extdir):
            for f in files:
                if f == "lock":
                    stale.append(os.path.join(root, f))
                    try:
                        os.remove(os.path.join(root, f))
                    except OSError:
                        pass
    if stale:
        print(f"[preflight] cleared {len(stale)} stale ninja build lock(s)")
    if problems:
        print("[preflight] " + "\n[preflight] ".join(problems))
        if strict:
            print("[preflight] ABORTING. Timings from a shared GPU are not "
                  "recoverable after the fact. Pass --allow-busy to override.")
            sys.exit(3)
    else:
        print(f"[preflight] GPU {gpu} idle, build locks clear -- ok to measure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=("fused", "sdpa"))
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dist", default="rand", choices=common.DISTRIBUTIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup-s", dest="warmup_s", type=float, default=2.0)
    ap.add_argument("--order-seed", type=int, default=20260728)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args()

    preflight(args.gpu, strict=not args.allow_busy)
    with open(args.jobs) as f:
        jobs = json.load(f)

    outdir = os.path.join(common2.RESULTS_DIR, args.tag, "raw")
    os.makedirs(outdir, exist_ok=True)

    plan = [(j, r) for j in jobs for r in range(args.reps)]
    random.Random(args.order_seed).shuffle(plan)
    print(f"[driver2] op={args.op} {len(jobs)} jobs x {args.reps} reps = "
          f"{len(plan)} processes, shuffled with seed {args.order_seed}")

    t0 = time.time()
    nok = nfail = 0
    for i, (job, rep) in enumerate(plan, 1):
        rec = run_job(job, rep, args.gpu, args, outdir)
        ok = rec.get("ok")
        nok += bool(ok)
        nfail += (not ok)
        ms = rec.get("timing", {}).get("median_ms")
        tag = f"{rec.get('dsl')}/{rec.get('variant')}"
        extra = (rec.get("cfg", {}) or {}).get("extra", {})
        if extra:
            tag += "[" + ",".join(f"{k}={v}" for k, v in sorted(extra.items())) + "]"
        msg = f"{ms:8.4f} ms" if ms else (rec.get("error_msg", "?")[:70])
        print(f"[{i:4d}/{len(plan)}] rep{rep} {tag:<58s} {'ok ' if ok else 'FAIL'} {msg}",
              flush=True)

    print(f"[driver2] done in {time.time()-t0:.0f}s: {nok} ok, {nfail} failed "
          f"-> {outdir}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
