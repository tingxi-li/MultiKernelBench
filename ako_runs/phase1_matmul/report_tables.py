#!/usr/bin/env python3
"""Emit the non-timing tables: precision control, KC sweep, generated-code census.

Timing tables come from analyze.py. These are the ones that do not depend on a
quiet GPU, because they are arithmetic and static-code facts.

usage: python report_tables.py > results/TABLES_static.md
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DSL_ORDER = ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"]


def load_accuracy():
    out = []
    for p in sorted(glob.glob(os.path.join(common.RESULTS_DIR, "accuracy", "*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def accuracy_table(recs):
    L = ["## Precision control — 20 seeds × 2 RMS-matched input distributions\n",
         "Both distributions have `RMS = 1/√3`, so the operands carry identical",
         "energy. Only the mean differs. `pass` is the fraction of the 20 seeds where",
         "**every** element satisfies the harness gate `|ref−got| ≤ 1e-4 + 1e-4·|ref|`.\n",
         "| variant | KC | dist | seeds passing | max abs err | gate budget | % elems failing | signed bias | ‖C‖ |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for d in recs:
        kc = d["cfg"]["kc"]
        for dist, s in d["by_dist"].items():
            rows.append((d["variant"], kc, dist, s))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    for v, kc, dist, s in rows:
        kcs = "full-K" if kc == 0 else str(kc)
        L.append(f"| {v} | {kcs} | `{dist}` | {s['trials_passing_gate']}/{s['n_seeds']} | "
                 f"{s['max_abs_err_worst']:.4g} | {s['budget_mean']:.4g} | "
                 f"{s['pct_elems_failing_gate_mean']:.3f}% | {s['signed_mean_err_mean']:+.4g} | "
                 f"{s['ref_abs_mean']:.1f} |")
    return "\n".join(L)


def kc_scaling_table(recs):
    """Is the split-K error a random walk or a systematic drift?

    Random walk => bias ~ sqrt(KC), i.e. x1.41 per doubling.
    Systematic  => bias ~ KC,       i.e. x2.00 per doubling.
    """
    pts = []
    for d in recs:
        if d["variant"] not in ("C", "B"):
            continue
        kc = d["cfg"]["kc"] or common.K
        s = d["by_dist"].get("rand")
        if s:
            pts.append((kc, abs(s["signed_mean_err_mean"]), s["max_abs_err_worst"],
                        s["trials_passing_gate"], s["n_seeds"]))
    pts = sorted(set(pts))
    L = ["\n## KC sweep — split-K is an accuracy lever, and the scaling law says why\n",
         "| KC | \\|bias\\| | ratio vs previous | max abs err | seeds passing (`rand`) |",
         "|---|---|---|---|---|"]
    prev = None
    for kc, bias, mx, p, n in pts:
        r = f"{bias / prev:.2f}×" if prev else "—"
        L.append(f"| {kc} | {bias:.4g} | {r} | {mx:.4g} | {p}/{n} |")
        prev = bias
    L.append("\nA random walk of independent round-offs would grow as √KC (**1.41× per "
             "doubling**); a systematic drift grows as KC (**2.00× per doubling**). "
             "The measured ratios decide which mechanism is at work.")
    return "\n".join(L)


def oracle_table(recs):
    L = ["\n## Is the fp32 oracle itself accurate? (vs an fp64 ground truth)\n",
         "| dist | oracle max err vs fp64 | oracle bias vs fp64 | gate budget |",
         "|---|---|---|---|"]
    seen = set()
    for d in recs:
        for dist, s in d["by_dist"].items():
            if dist in seen or "oracle_max_err_vs_fp64_mean" not in s:
                continue
            seen.add(dist)
            L.append(f"| `{dist}` | {s['oracle_max_err_vs_fp64_mean']:.4g} | "
                     f"{s['oracle_bias_vs_fp64_mean']:+.3g} | {s['budget_mean']:.4g} |")
    L.append("\nThe reference the benchmark scores against is not exact. Its own error "
             "sets the floor below which 'kernel error' cannot be measured.")
    return "\n".join(L)


def code_census():
    p = os.path.join(common.RESULTS_DIR, "code_inspection.json")
    if not os.path.exists(p):
        return ""
    recs = json.load(open(p))["records"]
    L = ["\n## Generated-code census (SASS) — what the compilers actually emitted\n",
         "Static instruction counts from `cuobjdump -sass`. **These are static, not "
         "dynamic**: a fully unrolled loop shows a large count against a rolled loop "
         "with a large trip count, for identical arithmetic. The bit-identical outputs "
         "across all four DSLs prove the arithmetic is the same regardless.\n",
         "| DSL | variant | HMMA | LDGSTS<br><sub>cp.async</sub> | LDSM<br><sub>ldmatrix</sub> | FFMA | regs | spill B | smem B |",
         "|---|---|---|---|---|---|---|---|---|"]
    by = {(r["dsl"], r["variant"]): r for r in recs if r.get("sass_counts")}
    for d in DSL_ORDER:
        for v in ("A", "B", "C", "D"):
            r = by.get((d, v))
            if not r:
                continue
            c = r["sass_counts"]
            smem = r.get("static_smem_bytes") or r.get("reported_shared_bytes", "—")
            L.append(f"| {d} | {v} | {c.get('HMMA', '—')} | {c.get('LDGSTS', '—')} | "
                     f"{c.get('LDSM', '—')} | {c.get('FFMA', '—')} | "
                     f"{r.get('n_regs') or r.get('reported_n_regs', '—')} | "
                     f"{r.get('spill_store_bytes', 0)} | {smem} |")
    return "\n".join(L)


def main():
    recs = load_accuracy()
    parts = ["# Phase-1 static tables (precision + generated code)\n",
             "*Independent of GPU load: these are arithmetic and static-code facts.*\n"]
    if recs:
        parts += [accuracy_table(recs), kc_scaling_table(recs), oracle_table(recs)]
    parts.append(code_census())
    print("\n".join(parts))


if __name__ == "__main__":
    main()
