#!/usr/bin/env python3
"""Convergence-log summarizer for the 6-op cross-DSL redo.

The per-cell convergence.csv is APPENDED BY timed_bench.sh (one row per benched
variant). This tool only READS it, post-hoc, to compute the convergence metric:
  time_to_ceiling = cum_compute_s of the first variant reaching >=0.95*ceiling.

Ceiling is read off the finished curve (max speedup seen) unless an external
reference is passed with --ceiling (cuBLAS/cuDNN/torch-SDPA/known layer_norm),
in which case ceiling = max(curve_best, external). Never circular: only run
after a cell's search has stopped.

Schema (see CONVERGENCE_PROTOCOL.md):
  iter,cum_compute_s,variant_desc,runtime_ms,speedup,ncu_key,kept,agent_s

Usage:
  convergence_log.py summarize <cell_dir_or_csv> [--ceiling F] [--frac 0.95]
  convergence_log.py summarize-all <root> [--frac 0.95]   # every convergence.csv under root
"""
import argparse
import csv
import os
import sys


def _find_csv(path):
    if os.path.isdir(path):
        cand = os.path.join(path, "convergence.csv")
        return cand if os.path.isfile(cand) else None
    return path if os.path.isfile(path) else None


def _rows(csv_path):
    with open(csv_path, newline="") as f:
        r = list(csv.DictReader(f))
    out = []
    for d in r:
        try:
            out.append({
                "iter": int(float(d["iter"])),
                "cum_compute_s": float(d["cum_compute_s"]),
                "variant_desc": d.get("variant_desc", ""),
                "runtime_ms": float(d["runtime_ms"]) if d.get("runtime_ms") else float("nan"),
                "speedup": float(d["speedup"]) if d.get("speedup") else float("nan"),
                "ncu_key": d.get("ncu_key", ""),
                "kept": str(d.get("kept", "")).strip() in ("1", "true", "True"),
            })
        except (KeyError, ValueError) as e:
            print(f"  ! skipping malformed row {d}: {e}", file=sys.stderr)
    return out


def summarize_one(csv_path, external_ceiling=None, frac=0.95):
    rows = _rows(csv_path)
    if not rows:
        return None
    speeds = [r["speedup"] for r in rows if r["speedup"] == r["speedup"]]  # drop NaN
    if not speeds:
        return None
    curve_best = max(speeds)
    ceiling = max(curve_best, external_ceiling) if external_ceiling else curve_best
    target = frac * ceiling
    hit = next((r for r in rows if r["speedup"] == r["speedup"] and r["speedup"] >= target), None)
    total_compute = rows[-1]["cum_compute_s"]
    n_variants = len(rows)
    n_kept = sum(1 for r in rows if r["kept"])
    return {
        "csv": csv_path,
        "n_variants": n_variants,
        "n_kept": n_kept,
        "curve_best": curve_best,
        "ceiling": ceiling,
        "external_ceiling": external_ceiling,
        "target_frac": frac,
        "time_to_ceiling_s": hit["cum_compute_s"] if hit else None,
        "iter_to_ceiling": hit["iter"] if hit else None,
        "reached": hit is not None,
        "total_compute_s": total_compute,
        "final_speedup": rows[-1]["speedup"],
    }


def _print(s):
    if s is None:
        print("  (no usable rows)")
        return
    cap = f"{s['ceiling']:.4f}"
    ext = f" (ext {s['external_ceiling']:.4f})" if s["external_ceiling"] else ""
    ttc = f"{s['time_to_ceiling_s']:.1f}s @ iter {s['iter_to_ceiling']}" if s["reached"] else "NOT REACHED"
    print(f"  variants={s['n_variants']} kept={s['n_kept']}  "
          f"best={s['curve_best']:.4f}  ceiling={cap}{ext}")
    print(f"  time_to_{int(s['target_frac']*100)}%_ceiling = {ttc}   "
          f"total_compute={s['total_compute_s']:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("summarize")
    p1.add_argument("path")
    p1.add_argument("--ceiling", type=float, default=None)
    p1.add_argument("--frac", type=float, default=0.95)
    p2 = sub.add_parser("summarize-all")
    p2.add_argument("root")
    p2.add_argument("--frac", type=float, default=0.95)
    a = ap.parse_args()

    if a.cmd == "summarize":
        csv_path = _find_csv(a.path)
        if not csv_path:
            print(f"no convergence.csv at {a.path}", file=sys.stderr)
            sys.exit(1)
        print(csv_path)
        _print(summarize_one(csv_path, a.ceiling, a.frac))
    else:
        for dirpath, _dirs, files in os.walk(a.root):
            if "convergence.csv" in files:
                cp = os.path.join(dirpath, "convergence.csv")
                print(cp)
                _print(summarize_one(cp, None, a.frac))
                print()


if __name__ == "__main__":
    main()
