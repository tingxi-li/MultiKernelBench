#!/usr/bin/env python3
"""Aggregate Phase-2 raw records and render the tables.

Reuses Phase 1's aggregation verbatim (`median_ci` over per-process medians,
t-based 95% CI) so a Phase-2 number and a Phase-1 number are the same kind of
object. Only the grouping key differs: Phase 2 keys on the factors this study
varies -- arm, weight mode, epilogue, and for SDPA the algorithm, head dim and
the two dtypes.

usage:
  python analyze2.py --tag fused_matched --table ladder
  python analyze2.py --tag fused_matched --table wcache
  python analyze2.py --tags fused_matched,fused_native,fused_epilogue --table full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
import common  # noqa: E402

ARM_ORDER = ["G", "GB", "GBG", "GBGS"]
ABS_ORDER = ["F1", "F2", "F3", "F4", "F2c", "F3c", "F4c"]
DSL_ORDER = ["tilelang", "triton", "cuda_unlimited", "cuda_noptx",
             "torch:fp16", "torch:fp32"]


def _recover_cfg_from_name(r, path):
    """Backfill the factor values for a record that has no `cfg`.

    `runner2` builds the config and the kernel in one call, so a record whose
    *build* failed carries `cfg: None` -- it never got far enough to record
    what it was. The factors are still in the filename, which the driver
    composed from the job's `--set` string, e.g.

        tilelang_abs__S3-MP__x_d256_x_pdtypefp16_x_sdtypefp32__rand__rep1.json

    Without this, a build failure lands in a `d=0` bucket and a table shows a
    phantom head-dim column instead of attributing the failure to the head dim
    that caused it -- which for `S3-MP` is the entire result (SPEC2.md §3.4).
    """
    name = os.path.basename(path)
    extra = {}
    for k, v in re.findall(r"x_([A-Za-z_]+?)([0-9]+|fp16|fp32|cached|uncached"
                           r"|native|smem|regs|precast|in_region)(?=_x_|__|\.)",
                           name):
        extra[k] = v
    if extra:
        r["cfg"] = {"extra": extra, "recovered_from_filename": True}
    return r


def load(tags) -> list[dict]:
    recs = []
    for tag in tags:
        for p in sorted(glob.glob(os.path.join(common2.RESULTS_DIR, tag,
                                               "raw", "*.json"))):
            try:
                with open(p) as f:
                    r = json.load(f)
                r["_tag"] = tag
                r["_path"] = p
                if not r.get("cfg"):
                    _recover_cfg_from_name(r, p)
                recs.append(r)
            except Exception:  # noqa: BLE001
                pass
    return recs


def lane(r):
    """DSL identity for a row. torch is split by arithmetic, because its two
    arithmetics are two different denominators, not one lane."""
    d = r.get("dsl")
    if d == "torch":
        return "torch:" + (r.get("cfg", {}) or {}).get("arith", "?")
    return d


def key(r):
    cfg = r.get("cfg", {}) or {}
    e = cfg.get("extra", {}) or {}
    # Every factor this study varies must be in the key, or cells that differ
    # get pooled into one median whose CI spans both populations.
    #   `soft_only` selects what the timer brackets -- the whole fused op
    #     (~1.5 ms) or the softmax kernel alone (~0.09 ms), a 17x difference.
    #   `d`/`sdtype`/`pdtype` are the SDPA factors; without them all nine
    #     (head dim x dtype pair) cells of a lane collapse into one.
    return (lane(r), r.get("variant"), e.get("wcache", "cached"),
            e.get("epilogue", "-"), cfg.get("cast", "precast"), r.get("dist"),
            str(e.get("soft_only", "0")),
            str(e.get("d", "-")), e.get("sdtype", "-"), e.get("pdtype", "-"))


def aggregate(recs):
    by = defaultdict(list)
    for r in recs:
        by[key(r)].append(r)
    out = {}
    for k, rs in by.items():
        ok = [r for r in rs if r.get("ok")]
        a = {"lane": k[0], "variant": k[1], "wcache": k[2], "epilogue": k[3],
             "cast": k[4], "dist": k[5], "soft_only": k[6],
             "d": k[7], "sdtype": k[8], "pdtype": k[9],
             "n_procs_total": len(rs), "n_procs_ok": len(ok)}
        if not ok:
            a["status"] = "FAILED"
            a["error_msg"] = (rs[0].get("error_msg") or "?")[:200]
            out[k] = a
            continue
        meds = [r["timing"]["median_ms"] for r in ok]
        a.update(common.median_ci(meds))
        a["compile_s_median"] = sorted(r.get("compile_s", 0.0)
                                       for r in ok)[len(ok) // 2]
        errs = [r.get("error") for r in ok if r.get("error")]
        if errs:
            a["max_abs_err"] = max(e["max_abs_err"] for e in errs)
            a["gate_pass"] = all(e["gate_pass"] for e in errs)
            a["pct_elems_failing_gate"] = max(e["pct_elems_failing_gate"]
                                              for e in errs)
        a["status"] = "OK"
        out[k] = a
    return out


def cell(a):
    if a is None:
        return "--"
    if a.get("status") != "OK":
        return "FAIL"
    s = f"{a['median_of_medians_ms']:.3f}"
    if not a.get("gate_pass", True):
        s += "*"
    return s


def _lanes_present(agg):
    seen = {a["lane"] for a in agg.values()}
    return [d for d in DSL_ORDER if d in seen] + sorted(seen - set(DSL_ORDER))


def table_ladder(agg, wcache="cached", arms=None, epilogue="-", cast="precast"):
    arms = arms or ARM_ORDER
    lanes = _lanes_present(agg)
    lines = [f"Fused ladder, absolute ms (median of per-process medians), "
             f"wcache={wcache}, cast={cast}",
             "`*` = fails the 1e-4 gate.", ""]
    head = "| lane | " + " | ".join(
        f"{a} ({common2.FUSED_ARMS[a]['label']})" if a in common2.FUSED_ARMS else a
        for a in arms) + " | GBGS-G |"
    lines += [head, "|" + "---|" * (len(arms) + 2)]
    for d in lanes:
        row = [d]
        vals = {}
        for a in arms:
            r = agg.get((d, a, wcache, epilogue, cast, "rand", "0", "-", "-", "-"))
            vals[a] = r
            row.append(cell(r))
        g, s = vals.get(arms[0]), vals.get(arms[-1])
        if g and s and g.get("status") == "OK" and s.get("status") == "OK":
            row.append(f"+{s['median_of_medians_ms'] - g['median_of_medians_ms']:.3f}")
        else:
            row.append("--")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_wcache(agg, arm="GBGS", cast="precast"):
    lanes = _lanes_present(agg)
    modes = [m for m in common2.WCACHE_MODES]
    lines = [f"Weight factor at arm {arm}, absolute ms", "",
             "| lane | " + " | ".join(modes) + " | cache worth | native gap |",
             "|" + "---|" * (len(modes) + 3)]
    for d in lanes:
        row, v = [d], {}
        for m in modes:
            r = agg.get((d, arm, m, "-", cast, "rand", "0", "-", "-", "-"))
            v[m] = r
            row.append(cell(r))
        c, u, n = (v.get(m) or {} for m in modes)
        if c.get("status") == "OK" and u.get("status") == "OK":
            row.append(f"{u['median_of_medians_ms'] - c['median_of_medians_ms']:+.3f}")
        else:
            row.append("--")
        if c.get("status") == "OK" and n.get("status") == "OK":
            row.append(f"{n['median_of_medians_ms'] - c['median_of_medians_ms']:+.3f}")
        else:
            row.append("--")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "`cache worth` = uncached - cached: what re-converting the "
                  "weight every call costs.",
              "`native gap`   = native  - cached: what a kernel that never "
              "materializes a transposed copy gives up. It is the honest floor "
              "for 'uncached'; the difference between the two columns is work "
              "the cache is credited with but that a competent uncached kernel "
              "simply does not do."]
    return "\n".join(lines)


def table_speedup(agg, arm="GBGS", wcache="cached", cast="precast"):
    lanes = _lanes_present(agg)
    d32 = agg.get(("torch:fp32", arm, wcache, "-", cast, "rand", "0", "-", "-", "-"))
    d16 = agg.get(("torch:fp16", arm, wcache, "-", cast, "rand", "0", "-", "-", "-"))
    if not d32 or d32.get("status") != "OK":
        return "(no fp32 denominator in this tag)"
    b32 = d32["median_of_medians_ms"]
    b16 = d16["median_of_medians_ms"] if d16 and d16.get("status") == "OK" else None
    lines = [f"Speedup at arm {arm}, wcache={wcache}: which denominator you "
             f"divide by decides the answer", "",
             "| lane | ms | vs torch fp32 | vs torch fp16 |", "|---|---|---|---|"]
    for d in lanes:
        a = agg.get((d, arm, wcache, "-", cast, "rand", "0", "-", "-", "-"))
        if not a or a.get("status") != "OK":
            continue
        m = a["median_of_medians_ms"]
        lines.append(f"| {d} | {m:.3f} | {b32/m:.2f}x | "
                     + (f"{b16/m:.2f}x |" if b16 else "-- |"))
    return "\n".join(lines)


ABS_LABEL = {
    "F1":  "T.reduce_max/sum (high)",
    "F2":  "manual smem tree",
    "F3":  "manual warp shuffle",
    "F4":  "warp shuffle + cached exp",
    "F2c": "manual smem tree, coalesced",
    "F3c": "manual warp shuffle, coalesced",
    "F4c": "warp shuffle + cached exp, coalesced",
}


def table_abstraction(agg, cast="precast"):
    """The softmax-abstraction ladder in all three regimes at once.

    Reading only the full-op columns would credit the abstraction level with
    ~1% and stop there. The softmax-only column is what actually isolates the
    factor, and it is the one that shows the abstraction level is worth ~0
    while the *indexing* choice the manual form exposes is worth 2.7x.
    """
    lines = ["Softmax abstraction ladder (TileLang), absolute ms. Identical GEMM "
             "kernel in every row; only the softmax level changes.", "",
             "| arm | softmax level | softmax kernel alone | full op, cached W | "
             "full op, uncached W | softmax share of full op |",
             "|---|---|---|---|---|---|"]
    base = None
    for a in ABS_ORDER:
        so = agg.get(("tilelang_abs", a, "cached", "-", cast, "rand", "1", "-", "-", "-"))
        fc = agg.get(("tilelang_abs", a, "cached", "-", cast, "rand", "0", "-", "-", "-"))
        fu = agg.get(("tilelang_abs", a, "uncached", "-", cast, "rand", "0", "-", "-", "-"))
        share = "--"
        if (so and so.get("status") == "OK") and (fc and fc.get("status") == "OK"):
            share = (f"{100.0 * so['median_of_medians_ms'] / fc['median_of_medians_ms']:.1f}%")
        if so and so.get("status") == "OK" and base is None:
            base = so["median_of_medians_ms"]
        lines.append(f"| {a} | {ABS_LABEL.get(a, '')} | {cell(so)} | {cell(fc)} | "
                     f"{cell(fu)} | {share} |")
    if base is not None:
        worst = max((v["median_of_medians_ms"]
                     for k, v in agg.items()
                     if k[6] == "1" and v.get("status") == "OK"), default=base)
        best = min((v["median_of_medians_ms"]
                    for k, v in agg.items()
                    if k[6] == "1" and v.get("status") == "OK"), default=base)
        lines += ["", f"Softmax-kernel spread across the whole ladder: "
                      f"{best:.4f} - {worst:.4f} ms ({worst/best:.2f}x). "
                      f"F1 (highest level) = {base:.4f} ms."]
    return "\n".join(lines)


def table_raw(agg):
    rows = sorted((a for a in agg.values()),
                  key=lambda a: (a["lane"], a["variant"], a["wcache"],
                                 a["epilogue"], a["cast"], a.get("soft_only", "0"),
                                 int(str(a.get("d", "")).lstrip("-") or 0),
                                 a.get("sdtype", "-"), a.get("pdtype", "-")))
    # The SDPA factor columns are only rendered when some row actually carries
    # them, so the fused table keeps its original shape and an SDPA table does
    # not silently show two different cells as identical rows.
    sdpa = any(a.get("d", "-") != "-" for a in agg.values())
    extra = " d | scores/probs |" if sdpa else ""
    lines = ["| lane | arm |" + extra +
             " wcache | epi | cast | timed | ms | ci95 | n | gate | maxerr |",
             "|---|---|" + ("---|---|" if sdpa else "") + "---|---|---|---|---|---|---|---|---|"]
    for a in rows:
        timed = "softmax" if a.get("soft_only") == "1" else "full"
        ext = (" %s | %s/%s |" % (a.get("d", "-"), a.get("sdtype", "-"),
                                  a.get("pdtype", "-"))) if sdpa else ""
        if a.get("status") != "OK":
            lines.append(f"| {a['lane']} | {a['variant']} |{ext} {a['wcache']} | "
                         f"{a['epilogue']} | {a['cast']} | {timed} | FAILED | | "
                         f"{a['n_procs_ok']}/{a['n_procs_total']} | | "
                         f"{a.get('error_msg','')[:60]} |")
            continue
        # `median_ci` only emits a CI when n > 1, so a cell that lost four of
        # its five processes must still render rather than crash the report.
        ci = ("%.4f-%.4f" % (a["ci95_lo_ms"], a["ci95_hi_ms"])
              if "ci95_lo_ms" in a else "n/a")
        err = a.get("max_abs_err")
        lines.append(
            "| %s | %s |%s %s | %s | %s | %s | %.4f | %s | %d/%d | %s | %s |"
            % (a["lane"], a["variant"], ext, a["wcache"], a["epilogue"], a["cast"],
               timed, a["median_of_medians_ms"], ci,
               a["n_procs_ok"], a["n_procs_total"],
               "pass" if a.get("gate_pass") else "FAIL",
               "%.2e" % err if err is not None else "--"))
    return "\n".join(lines)


TABLES = {
    "ladder": lambda agg, a: table_ladder(agg, wcache=a.wcache),
    "ladder_uncached": lambda agg, a: table_ladder(agg, wcache="uncached"),
    "abstraction": lambda agg, a: table_abstraction(agg),
    "wcache": lambda agg, a: table_wcache(agg, arm=a.arm),
    "speedup": lambda agg, a: table_speedup(agg, arm=a.arm, wcache=a.wcache),
    "raw": lambda agg, a: table_raw(agg),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--table", default="raw", choices=sorted(TABLES) + ["all"])
    ap.add_argument("--arm", default="GBGS")
    ap.add_argument("--wcache", default="cached")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    tags = [t for t in (a.tags.split(",") if a.tags else [a.tag]) if t]
    recs = load(tags)
    agg = aggregate(recs)
    print(f"# {len(recs)} raw records, {len(agg)} cells, tags={tags}\n")
    names = sorted(TABLES) if a.table == "all" else [a.table]
    for n in names:
        print(f"## {n}\n")
        print(TABLES[n](agg, a))
        print()
    if a.json_out:
        common2.write_json(a.json_out,
                           {"tags": tags,
                            "cells": [dict(v, key=list(k)) for k, v in agg.items()]})


if __name__ == "__main__":
    main()
