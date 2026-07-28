#!/usr/bin/env python3
"""Campaign driver: randomized-order, multi-process, serial-on-one-GPU timing.

Builds the full job list (variant x repeat), shuffles it with a fixed seed so
the order is random but reproducible, then runs each job in its own subprocess
one at a time on one pinned GPU. Nothing else may be running on that GPU.

Randomizing order matters because these cards ramp 210 MHz -> ~2.5 GHz and drift
with temperature; a fixed order silently gives whichever variant runs last a
systematically hotter (slower) or better-ramped (faster) card.

usage:
  python driver.py --jobs jobs/matched.json --gpu 0 --reps 5 --tag matched
  python driver.py --grid              --gpu 0 --reps 5 --tag matched
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
import common  # noqa: E402


def matched_grid(dsls=None, variants=None, geoms=("primary",)) -> list[dict]:
    dsls = dsls or list(common.DSLS)
    variants = variants or ["A", "B", "C", "D"]
    jobs = []
    for g in geoms:
        for d in dsls:
            for v in variants:
                jobs.append({"dsl": d, "variant": v, "geom": g, "set": ""})
    # the denominator
    for g in geoms[:1]:
        jobs.append({"dsl": "torch", "variant": "A", "geom": g, "set": ""})
        jobs.append({"dsl": "torch", "variant": "B", "geom": g, "set": ""})
    return jobs


def run_job(job, rep, gpu, args, outdir) -> dict:
    name = f"{job['dsl']}__{job['variant']}__{job['geom']}"
    if job.get("set"):
        name += "__" + job["set"].replace("=", "").replace(",", "_")
    name += f"__{args.dist}__rep{rep}"
    out = os.path.join(outdir, name + ".json")
    if os.path.exists(out) and not args.force:
        with open(out) as f:
            return json.load(f)

    cmd = [sys.executable, os.path.join(HERE, "runner.py"),
           "--dsl", job["dsl"], "--variant", job["variant"],
           "--geom", job["geom"], "--dist", args.dist,
           "--seed", str(args.seed), "--rep", str(rep),
           "--trials", str(args.trials), "--warmup", str(args.warmup),
           "--warmup-s", str(args.warmup_s),
           "--out", out]
    if job.get("set"):
        cmd += ["--set", job["set"]]
    if args.time_only:
        cmd += ["--time-only"]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=HERE, timeout=args.timeout)
    dur = time.time() - t0

    rec = None
    if "###JSON###" in p.stdout:
        try:
            rec = json.loads(p.stdout.split("###JSON###", 1)[1].strip().splitlines()[0])
        except Exception:
            rec = None
    if rec is None:
        rec = {"ok": False, "dsl": job["dsl"], "variant": job["variant"],
               "geom": job["geom"], "rep": rep, "key": name,
               "error_msg": "runner produced no JSON",
               "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-4000:]}
        common.write_json(out, rec)
    rec["wall_s"] = dur
    rec["returncode"] = p.returncode
    if not rec.get("ok"):
        rec.setdefault("stderr_tail", p.stderr[-4000:])
    return rec


def preflight(gpu: int, strict: bool = True):
    """Refuse to start a 'serialized' campaign on a busy GPU.

    A concurrent process on the target card silently contaminates every timing
    in the run, and the result looks perfectly well-formed afterwards -- there
    is no way to detect it from the numbers. Also clears stale ninja build locks,
    which otherwise deadlock a load_inline compile forever.
    """
    problems = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader",
             "-i", str(gpu)], capture_output=True, text=True, timeout=15).stdout.strip()
        others = [l for l in out.splitlines() if l.strip()]
        if others:
            problems.append(f"GPU {gpu} already has {len(others)} compute process(es): {others}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"could not query GPU {gpu}: {e}")

    stale = []
    extdir = os.path.join(HERE, ".torch_ext")
    for root, _dirs, files in os.walk(extdir):
        for f in files:
            if f == "lock":
                p = os.path.join(root, f)
                stale.append(p)
                try:
                    os.remove(p)
                except OSError:
                    pass
    if stale:
        print(f"[preflight] cleared {len(stale)} stale ninja build lock(s)")

    if problems:
        print("[preflight] " + "\n[preflight] ".join(problems))
        if strict:
            print("[preflight] ABORTING. Timings from a shared GPU are not "
                  "recoverable after the fact. Re-run when the card is idle, or "
                  "pass --allow-busy if you accept contaminated numbers.")
            sys.exit(3)
    else:
        print(f"[preflight] GPU {gpu} is idle, build locks clear -- ok to measure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="", help="JSON file: list of {dsl,variant,geom,set}")
    ap.add_argument("--grid", action="store_true", help="use the built-in matched grid")
    ap.add_argument("--dsls", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--geoms", default="primary")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dist", default="rand", choices=common.DISTRIBUTIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--warmup-s", dest="warmup_s", type=float, default=0.0)
    ap.add_argument("--order-seed", type=int, default=20260727)
    ap.add_argument("--tag", default="matched")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--allow-busy", action="store_true",
                    help="measure anyway on a GPU that already has compute processes")
    args = ap.parse_args()

    preflight(args.gpu, strict=not args.allow_busy)

    if args.jobs:
        with open(args.jobs) as f:
            jobs = json.load(f)
    else:
        jobs = matched_grid(
            dsls=[s for s in args.dsls.split(",") if s] or None,
            variants=[s for s in args.variants.split(",") if s] or None,
            geoms=tuple(s for s in args.geoms.split(",") if s))

    outdir = os.path.join(common.RESULTS_DIR, args.tag, "raw")
    os.makedirs(outdir, exist_ok=True)

    plan = [(j, r) for j in jobs for r in range(args.reps)]
    random.Random(args.order_seed).shuffle(plan)

    print(f"[driver] tag={args.tag} gpu={args.gpu} jobs={len(jobs)} reps={args.reps} "
          f"-> {len(plan)} processes, order-seed={args.order_seed}, dist={args.dist}")

    recs = []
    t0 = time.time()
    for i, (job, rep) in enumerate(plan, 1):
        label = f"{job['dsl']}/{job['variant']}/{job['geom']}" + (f" [{job['set']}]" if job.get('set') else "")
        print(f"[{i}/{len(plan)}] rep{rep} {label} ...", flush=True)
        try:
            rec = run_job(job, rep, args.gpu, args, outdir)
        except subprocess.TimeoutExpired:
            rec = {"ok": False, "dsl": job["dsl"], "variant": job["variant"],
                   "geom": job["geom"], "rep": rep, "error_msg": "TIMEOUT"}
        recs.append(rec)
        if rec.get("ok"):
            t = rec["timing"]
            err = rec.get("error", {})
            gate = "n/a" if not err else ("PASS" if err.get("gate_pass") else "FAIL")
            extra = "" if not err else "  maxerr={:.4g}".format(err.get("max_abs_err", 0))
            print(f"      median={t['median_ms']:.4f} ms  "
                  f"({t['tflops_at_median']:.1f} TF/s)  compile={rec.get('compile_s', 0):.1f}s  "
                  f"gate={gate}{extra}", flush=True)
        else:
            print(f"      FAILED: {str(rec.get('error_msg'))[:200]}", flush=True)

    summary_path = os.path.join(common.RESULTS_DIR, args.tag, "summary.json")
    common.write_json(summary_path, {
        "tag": args.tag, "args": vars(args), "n_processes": len(plan),
        "wall_s": time.time() - t0, "records": recs,
    })
    print(f"[driver] done in {time.time()-t0:.0f}s -> {summary_path}")


if __name__ == "__main__":
    main()
