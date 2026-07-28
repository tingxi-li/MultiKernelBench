#!/usr/bin/env python3
"""Tables for the Phase-1 sub-studies: KC sweep, casting, pipeline depth,
equal-budget native tuning, and the TileLang abstraction study.

Every sub-study contains a point that also exists in the matched table (kc=2048,
cast=precast, stages=3, geom=primary). That shared point is an ANCHOR: comparing
it across campaigns detects drift between runs instead of assuming there is none.

usage: python sweep_tables.py --which kc_sweep casting pipeline abstraction native_tuned
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from analyze import aggregate, load  # noqa: E402

DSL_ORDER = ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"]


def cfgval(key_cfgstr, field):
    parts = dict(p.split("=", 1) for p in (key_cfgstr or "").split(";") if "=" in p)
    return parts.get(field)


def rows(tag):
    recs = load(tag)
    if not recs:
        return {}
    return aggregate(recs)


def _ms(a):
    return a["median_of_medians_ms"] if a and a.get("status") == "OK" else None


def table_kc(agg):
    L = ["## KC sweep — runtime cost of the chunk flush\n",
         "Absolute median ms. Pair with the error column from the precision study: "
         "error falls ~2× per halving of KC while runtime moves comparatively little, "
         "which is what makes split-K an accuracy lever rather than a performance one.\n"]
    for variant, note in (("C", "pipeline OFF (stages=1)"), ("D", "pipeline ON (stages=3)")):
        kcs = sorted({int(cfgval(k[4], "kc")) for k in agg
                      if k[1] == variant and cfgval(k[4], "kc")})
        if not kcs:
            continue
        L.append(f"\n**Variant {variant} — {note}**\n")
        L.append("| DSL | " + " | ".join(f"KC={k}" for k in kcs) + " | KC=512 ÷ KC=8192 |")
        L.append("|" + "---|" * (len(kcs) + 2))
        for d in DSL_ORDER:
            cells, vals = [], {}
            for kc in kcs:
                a = next((x for k, x in agg.items()
                          if k[0] == d and k[1] == variant and cfgval(k[4], "kc") == str(kc)), None)
                m = _ms(a)
                vals[kc] = m
                cells.append(f"{m:.3f}" if m else "—")
            span = "—"
            if vals.get(kcs[0]) and vals.get(kcs[-1]):
                span = f"{vals[kcs[0]] / vals[kcs[-1]]:.2f}×"
            L.append(f"| {d} | " + " | ".join(cells) + f" | {span} |")
    return "\n".join(L)


def table_casting(agg):
    L = ["\n## Casting control — fp16 compute vs the cost of getting to fp16\n",
         "Same fp16 kernel, three input paths. `precast` = operands already fp16 "
         "(conversion outside the timed region). `in_region` = `.half()` executed "
         "inside the timed region, which is what the incumbent solution does. "
         "`on_load` = kernel reads fp32 global and converts into shared memory.\n",
         "| DSL | precast | in_region | on_load | in_region − precast | on_load ÷ precast |",
         "|---|---|---|---|---|---|"]
    for d in DSL_ORDER + ["torch"]:
        got = {}
        for c in common.CAST_MODES:
            a = next((x for k, x in agg.items()
                      if k[0] == d and cfgval(k[4], "cast") == c and cfgval(k[4], "BN") == "128"), None)
            got[c] = _ms(a)
        if not any(got.values()):
            continue
        d1 = (f"+{got['in_region'] - got['precast']:.3f} ms"
              if got.get("in_region") and got.get("precast") else "—")
        d2 = (f"{got['on_load'] / got['precast']:.2f}×"
              if got.get("on_load") and got.get("precast") else "—")
        cells = [f"{got[c]:.3f}" if got.get(c) else "—" for c in common.CAST_MODES]
        L.append(f"| {d} | " + " | ".join(cells) + f" | {d1} | {d2} |")

    inc = [(k, x) for k, x in agg.items() if k[2] == "incumbent"]
    if inc:
        L.append("\n**The incumbent solution's own shape** (`BM=128 BN=256 BK=64`, "
                 "`stages=2`, `KC=2048`) — its published form casts inside `forward()`:\n")
        L.append("| cast | median ms |")
        L.append("|---|---|")
        for k, x in sorted(inc, key=lambda kv: cfgval(kv[0][4], "cast") or ""):
            m = _ms(x)
            L.append(f"| {cfgval(k[4], 'cast')} | {m:.3f} |" if m else
                     f"| {cfgval(k[4], 'cast')} | fail |")
    return "\n".join(L)


def table_pipeline(agg):
    depths = sorted({int(cfgval(k[4], "stages")) for k in agg if cfgval(k[4], "stages")})
    L = ["\n## Pipeline depth\n",
         "Everything else held at the matched configuration. For the CUDA lanes "
         "`stages=1` is a synchronous single buffer and `stages>1` is async multi-buffering.\n",
         "| DSL | " + " | ".join(f"stages={s}" for s in depths) + " | best ÷ stages=1 |",
         "|" + "---|" * (len(depths) + 2)]
    for d in DSL_ORDER:
        vals, cells = {}, []
        for s in depths:
            a = next((x for k, x in agg.items()
                      if k[0] == d and cfgval(k[4], "stages") == str(s)), None)
            m = _ms(a)
            vals[s] = m
            cells.append(f"{m:.3f}" if m else "—")
        ok = [v for v in vals.values() if v]
        gain = f"{vals[1] / min(ok):.2f}×" if ok and vals.get(1) else "—"
        L.append(f"| {d} | " + " | ".join(cells) + f" | {gain} |")
    return "\n".join(L)


def table_abstraction(agg):
    order = ["H1", "H2", "M1", "M2", "S1"]
    L = ["\n## TileLang abstraction study (reported separately from the cross-DSL study)\n",
         "| variant | level | inner-K implementation | median ms | vs H1 |",
         "|---|---|---|---|---|"]
    h1 = _ms(next((x for k, x in agg.items() if k[1] == "H1" and not cfgval(k[4], "stages_override")), None))
    for v in order:
        spec = common.ABSTRACTION_SPECS[v]
        a = next((x for k, x in agg.items() if k[1] == v), None)
        m = _ms(a)
        rel = f"{m / h1:.2f}×" if m and h1 else "—"
        L.append(f"| {v} | {spec['level']} | {spec['label']} | "
                 f"{m:.3f} |" .replace("None", "—") + f" {rel} |")
    # The depth control (H1@stages=2) lives in its own campaign and is reported by
    # abstraction_rule.py, which is injected directly below this table. Repeating
    # it here would duplicate the numbers and risk the two disagreeing.
    return "\n".join(L)


def table_native(agg):
    from analyze import canonical
    matched = rows("matched")
    torch_ms = None
    tr = canonical(matched, "torch", "A", "primary", "rand") if matched else None
    if tr:
        torch_ms = tr["median_of_medians_ms"]

    L = ["\n## Equal-budget native tuning — the same configuration grid offered to every DSL\n",
         "Each DSL searches an identical 19-point grid with an identical budget; its "
         "best point becomes its row. This is the counterpart to the matched table: "
         "it asks who exploits a shared search space best, rather than who is fastest "
         "at one imposed configuration. The search ran at 2 processes per point "
         "because it only has to rank; the winners are re-measured at the full "
         "protocol below.\n",
         "| DSL | best configuration | median ms | vs its matched-D | vs torch fp32 | points that ran |",
         "|---|---|---|---|---|---|"]
    best = {}
    counts = defaultdict(int)
    for k, x in agg.items():
        if x.get("status") != "OK":
            continue
        counts[k[0]] += 1
        m = _ms(x)
        if m and (k[0] not in best or m < best[k[0]][0]):
            best[k[0]] = (m, k[4])
    vals = []
    for d in DSL_ORDER:
        if d not in best:
            continue
        m, cfgstr = best[d]
        vals.append(m)
        parts = dict(p.split("=", 1) for p in cfgstr.split(";") if "=" in p)
        desc = (f"BM={parts.get('BM')} BN={parts.get('BN')} BK={parts.get('BK')} "
                f"stages={parts.get('stages')}")
        md = canonical(matched, d, "D", "primary", "rand") if matched else None
        gain = f"{md['median_of_medians_ms'] / m:.2f}×" if md else "—"
        spd = f"{torch_ms / m:.2f}×" if torch_ms else "—"
        L.append(f"| {d} | {desc} | {m:.3f} | {gain} | {spd} | {counts[d]} |")
    if len(vals) > 1:
        L.append(f"\nSpread across DSLs at each one's own tuned best: "
                 f"**{max(vals) / min(vals):.2f}×**.")

    # The grid was equal on paper. Report where it was not equal in practice --
    # a silently smaller search space for one lane would otherwise read as that
    # lane simply searching worse.
    short = {d: n for d, n in counts.items() if n < max(counts.values() or [0])}
    if short:
        L.append("")
        for d, n in sorted(short.items()):
            L.append(f"**The grid was not equally available.** `{d}` reached "
                     f"{n} of {max(counts.values())} points; the rest failed to "
                     f"build. Points a DSL cannot compile are a real property of "
                     f"the toolchain, but they also mean its row is the best of a "
                     f"smaller search — see the note under the table.")
    return "\n".join(L)


BUILDERS = {"kc_sweep": table_kc, "casting": table_casting, "pipeline": table_pipeline,
            "abstraction": table_abstraction, "native_tuned": table_native}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="*", default=list(BUILDERS))
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    out = []
    for tag in args.which:
        agg = rows(tag)
        if not agg:
            out.append(f"\n*(no results yet for `{tag}`)*")
            continue
        try:
            out.append(BUILDERS[tag](agg))
        except Exception as e:  # noqa: BLE001
            out.append(f"\n*(could not build `{tag}` table: {type(e).__name__}: {e})*")
    md = "\n".join(out) + "\n"
    print(md)
    if args.md:
        os.makedirs(os.path.dirname(args.md), exist_ok=True)
        with open(args.md, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
