#!/usr/bin/env python3
"""Turn raw per-process records into the Phase-1 tables.

Aggregation contract (measurement discipline):
  * each process contributes ONE number: the median of its 100 timed trials
  * the reported value is the median of those per-process medians
  * the reported interval is the full [min,max] range of per-process medians
    plus a t-based 95% CI on their mean; both are shown, neither is dressed up
  * absolute runtime is primary; every ratio is derived from medians and is
    labelled as derived

usage:
  python analyze.py --tag matched
  python analyze.py --tag matched --md results/matched/TABLES.md
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

VARIANT_ORDER = ["A", "B", "C", "D"]
DSL_ORDER = ["torch", "tilelang", "triton", "cuda_noptx", "cuda_unlimited"]


def load(tag: str) -> list[dict]:
    pat = os.path.join(common.RESULTS_DIR, tag, "raw", "*.json")
    recs = []
    for p in sorted(glob.glob(pat)):
        try:
            with open(p) as f:
                recs.append(json.load(f))
        except Exception:
            pass
    return recs


def group_key(r):
    cfg = r.get("cfg", {})
    extra = []
    for k in ("kc", "stages", "cast", "BM", "BN", "BK"):
        if k in cfg:
            extra.append(f"{k}={cfg[k]}")
    return (r.get("dsl"), r.get("variant"), r.get("geom"), r.get("dist"), ";".join(extra))


def aggregate(recs):
    """(dsl,variant,geom,dist,cfgstr) -> aggregate dict."""
    by = defaultdict(list)
    for r in recs:
        if not r.get("ok"):
            by[group_key(r)].append(r)
            continue
        by[group_key(r)].append(r)
    out = {}
    for k, rs in by.items():
        ok = [r for r in rs if r.get("ok")]
        agg = {"dsl": k[0], "variant": k[1], "geom": k[2], "dist": k[3], "cfg": k[4],
               "n_procs_total": len(rs), "n_procs_ok": len(ok)}
        if not ok:
            agg["status"] = "FAILED"
            agg["error_msg"] = rs[0].get("error_msg", "?")
            out[k] = agg
            continue
        meds = [r["timing"]["median_ms"] for r in ok]
        agg.update(common.median_ci(meds))
        m = agg["median_of_medians_ms"]
        agg["tflops"] = common.FLOPS / (m * 1e-3) / 1e12 if m else 0.0
        agg["compile_s_median"] = sorted(r.get("compile_s", 0.0) for r in ok)[len(ok) // 2]
        errs = [r.get("error") for r in ok if r.get("error")]
        if errs:
            agg["max_abs_err"] = max(e["max_abs_err"] for e in errs)
            agg["mean_abs_err"] = sum(e["mean_abs_err"] for e in errs) / len(errs)
            agg["pct_elems_failing_gate"] = max(e["pct_elems_failing_gate"] for e in errs)
            agg["gate_pass"] = all(e["gate_pass"] for e in errs)
            agg["signed_mean_err"] = sum(e["signed_mean_err"] for e in errs) / len(errs)
        agg["status"] = "OK"
        out[k] = agg
    return out


def canonical(agg, dsl, variant, geom, dist):
    """The record for the variant's OWN spec, not any swept relative of it.

    Without this, a `kc=512` run sitting in the same tag could silently be
    picked up as if it were variant C, which would make the C-vs-B step measure
    a chunk size it never used.
    """
    spec = common.VARIANT_SPECS.get(variant) or common.ABSTRACTION_SPECS.get(variant, {})
    g = common.GEOMS.get(geom, {})
    want = {"kc": spec.get("kc"), "stages": spec.get("stages"),
            "BM": g.get("BM"), "BN": g.get("BN"), "BK": g.get("BK"),
            "cast": "precast"}
    best = None
    for k, a in agg.items():
        if (k[0], k[1], k[2], k[3]) != (dsl, variant, geom, dist):
            continue
        cfgstr = k[4] or ""
        parts = dict(p.split("=", 1) for p in cfgstr.split(";") if "=" in p)
        ok = True
        for f, v in want.items():
            if v is None or f not in parts:
                continue
            if str(parts[f]) != str(v):
                ok = False
                break
        if ok:
            # prefer an exact/short cfg string over a longer overridden one
            if best is None or len(cfgstr) < len(best[0]):
                best = (cfgstr, a)
    return best[1] if best else None


def fmt_cell(a):
    if a is None:
        return "—"
    if a.get("status") != "OK":
        return "**fail**"
    s = f"{a['median_of_medians_ms']:.3f}"
    if a.get("gate_pass") is False:
        s += " ✗"
    elif a.get("gate_pass") is True:
        s += " ✓"
    return s


def table_matched(agg, geom="primary", dist="rand"):
    lines = []
    lines.append(f"### Matched configuration — geom={geom}, inputs={dist}\n")
    lines.append("Absolute median runtime in ms (median of per-process medians). "
                 "✓/✗ = passes / fails the harness gate `|ref−got| ≤ 1e-4 + 1e-4·|ref|`.\n")
    hdr = "| DSL | " + " | ".join(f"{v}<br><sub>{common.VARIANT_SPECS[v]['label']}</sub>" for v in VARIANT_ORDER) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(VARIANT_ORDER) + 1))
    for d in DSL_ORDER:
        cells = [fmt_cell(canonical(agg, d, v, geom, dist)) for v in VARIANT_ORDER]
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_tflops(agg, geom="primary", dist="rand"):
    lines = ["\n### Achieved TFLOP/s (2·M·N·K ÷ median runtime)\n"]
    lines.append("| DSL | " + " | ".join(VARIANT_ORDER) + " |")
    lines.append("|" + "---|" * (len(VARIANT_ORDER) + 1))
    for d in DSL_ORDER:
        cells = []
        for v in VARIANT_ORDER:
            a = canonical(agg, d, v, geom, dist)
            cells.append(f"{a['tflops']:.1f}" if a and a.get("status") == "OK" else "—")
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_decomposition(agg, geom="primary", dist="rand"):
    """The whole point: B/A, C/B, D/C, and the residual spread at D."""
    lines = ["\n### Decomposition — each step isolates one factor\n"]
    lines.append("Speedup of the later variant over the earlier one, derived from the "
                 "medians above. >1 means the step made it faster.\n")
    lines.append("| DSL | B/A<br><sub>fp16 tensor cores</sub> | C/B<br><sub>split-K chunk flush</sub> | "
                 "D/C<br><sub>software pipeline</sub> | D/A<br><sub>total</sub> |")
    lines.append("|---|---|---|---|---|")
    for d in DSL_ORDER:
        got = {}
        for v in VARIANT_ORDER:
            a = canonical(agg, d, v, geom, dist)
            if a and a.get("status") == "OK":
                got[v] = a["median_of_medians_ms"]

        def ratio(num, den):
            if num in got and den in got and got[num]:
                return f"{got[den] / got[num]:.2f}×"
            return "—"
        row = [ratio("B", "A"), ratio("C", "B"), ratio("D", "C"), ratio("D", "A")]
        if all(c == "—" for c in row):
            continue
        lines.append(f"| {d} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_spread(agg, geom="primary", dist="rand"):
    lines = ["\n### Measurement quality — per-process spread\n"]
    lines.append("| key | n proc | median ms | min–max ms | 95% CI ms | spread % | compile s |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in sorted(agg, key=lambda x: (DSL_ORDER.index(x[0]) if x[0] in DSL_ORDER else 99, x[1])):
        a = agg[k]
        if a.get("status") != "OK" or a["geom"] != geom or a["dist"] != dist:
            continue
        ci = (f"{a['ci95_lo_ms']:.3f}–{a['ci95_hi_ms']:.3f}" if "ci95_lo_ms" in a else "—")
        lines.append(f"| {a['dsl']}/{a['variant']} | {a['n_procs_ok']} | "
                     f"{a['median_of_medians_ms']:.4f} | {a['min_ms']:.4f}–{a['max_ms']:.4f} | {ci} | "
                     f"{a.get('rel_spread_pct', 0):.1f} | {a.get('compile_s_median', 0):.1f} |")
    return "\n".join(lines)


def table_error(agg, geom="primary", dist="rand"):
    lines = [f"\n### Error against the fp32 oracle — inputs={dist}\n"]
    lines.append("| DSL | variant | max abs err | gate budget | % elems failing | signed bias | gate |")
    lines.append("|---|---|---|---|---|---|---|")
    budget = 0.205 if dist == "rand" else 0.0025
    for k in sorted(agg, key=lambda x: (DSL_ORDER.index(x[0]) if x[0] in DSL_ORDER else 99, x[1])):
        a = agg[k]
        if a.get("status") != "OK" or a["geom"] != geom or a["dist"] != dist:
            continue
        if "max_abs_err" not in a:
            continue
        lines.append(f"| {a['dsl']} | {a['variant']} | {a['max_abs_err']:.4g} | ~{budget} | "
                     f"{a['pct_elems_failing_gate']:.4f}% | {a['signed_mean_err']:+.4g} | "
                     f"{'PASS' if a['gate_pass'] else 'FAIL'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="matched")
    ap.add_argument("--geom", default="primary")
    ap.add_argument("--dist", default="rand")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    recs = load(args.tag)
    if not recs:
        print(f"no records under results/{args.tag}/raw/")
        return
    agg = aggregate(recs)

    parts = [f"# Phase-1 tables — tag `{args.tag}`\n",
             f"*{len(recs)} process records, "
             f"{sum(1 for r in recs if r.get('ok'))} successful.*\n",
             table_matched(agg, args.geom, args.dist),
             table_tflops(agg, args.geom, args.dist),
             table_decomposition(agg, args.geom, args.dist),
             table_error(agg, args.geom, args.dist),
             table_spread(agg, args.geom, args.dist)]
    md = "\n".join(parts) + "\n"
    print(md)

    common.write_json(os.path.join(common.RESULTS_DIR, args.tag, "aggregate.json"),
                      {"tag": args.tag, "aggregate": {str(k): v for k, v in agg.items()}})
    if args.md:
        os.makedirs(os.path.dirname(args.md), exist_ok=True)
        with open(args.md, "w") as f:
            f.write(md)
        print(f"-> {args.md}")


if __name__ == "__main__":
    main()
