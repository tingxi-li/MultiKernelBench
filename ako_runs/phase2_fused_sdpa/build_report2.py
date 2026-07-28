#!/usr/bin/env python3
"""Fill PHASE2_REPORT.md's placeholders from the campaign results.

Same arrangement as Phase 1's build_report.py: the report is prose with
`<!--PLACEHOLDER-->` markers, every table is generated from `results/`, and the
build fails loudly if a marker is left unfilled or a table comes out empty. A
table that silently renders as a bare header is the failure mode this exists to
prevent.

usage: python build_report2.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
import analyze2  # noqa: E402

SRC = os.path.join(HERE, "PHASE2_REPORT.md")


def _demote(s: str) -> str:
    """Injected tables carry their own `## ` headings; the report's own outline
    owns that level, so push injected headings one deeper."""
    return re.sub(r"(?m)^## ", "### ", s)


def _load(tags):
    return analyze2.aggregate(analyze2.load(tags))


def _json(name, default=None):
    p = os.path.join(common2.RESULTS_DIR, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ tables ---
def t_ladder():
    agg = _load(["fused_matched"])
    return (analyze2.table_ladder(agg, wcache="cached") + "\n\n"
            + analyze2.table_ladder(agg, wcache="uncached"))


def t_speedup():
    agg = _load(["fused_matched"])
    return analyze2.table_speedup(agg, arm="GBGS", wcache="cached")


def t_wcache():
    agg = _load(["fused_matched", "fused_native"])
    return analyze2.table_wcache(agg, arm="GBGS")


def t_epilogue():
    agg = _load(["fused_epilogue"])
    rows = [a for a in agg.values() if a["lane"] == "cuda_unlimited"
            and a.get("status") == "OK"]
    if not rows:
        return "(campaign not run)"
    by = {(a["variant"], a["epilogue"]): a for a in rows}
    out = ["| arm | epilogue=smem (matched) | epilogue=regs (native) | regs advantage |",
           "|---|---|---|---|"]
    for arm in analyze2.ARM_ORDER:
        s, r = by.get((arm, "smem")), by.get((arm, "regs"))
        if not s or not r:
            continue
        out.append("| %s | %.3f | %.3f | %+.3f |"
                   % (arm, s["median_of_medians_ms"], r["median_of_medians_ms"],
                      s["median_of_medians_ms"] - r["median_of_medians_ms"]))
    return "\n".join(out) if len(out) > 2 else "(no paired cells)"


def t_cast():
    agg = _load(["fused_cast", "fused_matched"])
    out = ["| lane | cast=precast | cast=in_region | activation cast cost |",
           "|---|---|---|---|"]
    # Select by field rather than by literal tuple: the aggregation key gains a
    # slot whenever the study gains a factor, and a hard-coded tuple then stops
    # matching *silently* -- every cell reads None and the table degrades to
    # "(campaign not run)" while the records are sitting on disk.
    def pick(lane, cast):
        for a in agg.values():
            if (a["lane"] == lane and a["variant"] == "GBGS"
                    and a["wcache"] == "cached" and a["epilogue"] == "-"
                    and a["cast"] == cast and a["dist"] == "rand"
                    and a.get("soft_only", "0") == "0"):
                return a
        return None

    for d in analyze2._lanes_present(agg):
        p, i = pick(d, "precast"), pick(d, "in_region")
        if not p or not i or p.get("status") != "OK" or i.get("status") != "OK":
            continue
        out.append("| %s | %.3f | %.3f | %+.3f |"
                   % (d, p["median_of_medians_ms"], i["median_of_medians_ms"],
                      i["median_of_medians_ms"] - p["median_of_medians_ms"]))
    return "\n".join(out) if len(out) > 2 else "(campaign not run)"


ABS_LABEL = {
    "F1": "T.reduce_max / T.reduce_sum",
    "F2": "manual smem tree",
    "F3": "manual warp shuffle, exp recomputed",
    "F4": "warp shuffle + cached exp (incumbent)",
    "F2c": "manual smem tree, coalesced",
    "F3c": "manual warp shuffle, coalesced",
    "F4c": "warp shuffle + cached exp, coalesced",
}
ABS_PATTERN = {"F1": "n/a (T.copy)", "F2": "thread-contiguous",
               "F3": "thread-contiguous", "F4": "thread-contiguous",
               "F2c": "coalesced", "F3c": "coalesced", "F4c": "coalesced"}


def t_fabs():
    agg = _load(["fused_abstraction"])
    rows = [a for a in agg.values() if a.get("status") == "OK"]
    if not rows:
        return "(campaign not run)"
    # Split the two timing regimes explicitly. `soft_only` selects what the timer
    # brackets -- the whole fused op (~1.5 ms) or the softmax kernel alone
    # (~0.09 ms) -- so pooling them would collapse a 17x difference into one
    # meaningless median. (`analyze2.key` now carries the flag too; this stays
    # independent of it so the two paths cross-check each other.)
    raw = analyze2.load(["fused_abstraction"])
    so, fu = {}, {}
    for r in raw:
        if not r.get("ok"):
            continue
        e = (r.get("cfg", {}) or {}).get("extra", {}) or {}
        if e.get("wcache", "cached") != "cached":
            continue
        (so if str(e.get("soft_only", "")) in ("1", "True", "true") else fu) \
            .setdefault(r["variant"], []).append(r["timing"]["median_ms"])
    import common as c1
    out = ["| arm | reduction | access pattern | softmax kernel alone (ms) | whole op (ms) |",
           "|---|---|---|---|---|"]
    for arm in ("F1", "F2", "F3", "F4", "F2c", "F3c", "F4c"):
        s = c1.median_ci(so[arm])["median_of_medians_ms"] if arm in so else None
        f = c1.median_ci(fu[arm])["median_of_medians_ms"] if arm in fu else None
        out.append("| %s | %s | %s | %s | %s |"
                   % (arm, ABS_LABEL[arm], ABS_PATTERN[arm],
                      f"{s:.4f}" if s else "--", f"{f:.3f}" if f else "--"))
    return "\n".join(out)


def _ms_ci(r, field):
    """`field` ms with its across-process CI, or a bare figure with a marker if
    the record predates the repeated-process aggregation."""
    v = r.get(field)
    if not isinstance(v, (int, float)):
        return "—", "—"
    ci = r.get(field + "_ci95")
    n = r.get(field + "_n")
    if ci:
        return "%.3f" % v, "[%.3f, %.3f]" % (ci[0], ci[1])
    return "%.3f" % v, ("n=1" if n in (None, 1) else "—")


def _procs_note(d):
    n = d.get("n_procs")
    if not n or n < 2:
        return ("\nSingle process, single measurement — no interval, and the "
                "orderings below are not resolvable.")
    return ("\n%s. Intervals that overlap mean the ordering of those two rows "
            "is **not resolved** by this measurement."
            % d.get("aggregation", f"{n} independent processes"))


def t_incumbent_fused():
    d = _json("fused_incumbent_check.json")
    if not d:
        return "(fused_incumbent_check.py not run)"
    out = ["| what | ms (no L2 flush) | 95% CI | vs torch fp32 | vs torch fp16 | gate |",
           "|---|---|---|---|---|---|"]
    for r in d["records"]:
        if r.get("error"):
            out.append(f"| {r['who']} | FAILED | | | | {str(r['error'])[-60:]} |")
            continue
        if "ms" in r:
            ms, ci = _ms_ci(r, "ms")
            out.append(f"| {r['who']} (flush_l2={r.get('flush_l2')}) | "
                       f"{ms} | {ci} | | | |")
            continue
        ms, ci = _ms_ci(r, "ms_flush0")
        out.append("| %s | %s | %s | %.2fx | %.2fx | %s |"
                   % (r["who"], ms, ci, r["vs_fp32"], r["vs_fp16"],
                      "pass" if r["gate_pass"] else
                      f"**FAIL** ({r['max_abs_err']:.1e})"))
    return "\n".join(out) + "\n" + _procs_note(d)


def t_gate():
    d = _json("fused_incumbent_check.json")
    if not d or "gate_sensitivity" not in d:
        return "(fused_incumbent_check.py not run)"
    gs = dict(d["gate_sensitivity"])
    sc = gs.pop("_ref_scale", {})
    out = [f"Softmax output scale: mean {sc.get('mean', 0):.3e}, "
           f"max {sc.get('max', 0):.3e}; median gate tolerance "
           f"{sc.get('median_tol', 0):.3e}.", "",
           "| substitute for the true answer | % of elements inside tolerance | gate |",
           "|---|---|---|"]
    label = {"uniform_1_over_N": "a constant tensor, every element = 1/8192",
             "row_mean": "each row replaced by its own mean",
             "zeros": "all zeros",
             "shuffled_rows": "**the reference with its rows reversed** "
                              "(every row's answer assigned to the wrong row)"}
    for k, v in gs.items():
        out.append("| %s | %.2f%% | %s |"
                   % (label.get(k, k), v["pct_pass"],
                      "pass" if v["gate_pass"] else "fail"))
    return "\n".join(out)


def t_sdpa_audit():
    d = _json("sdpa_reference_audit.json")
    if not d:
        return "(sdpa_reference_audit.py not run)"
    out = ["| head dim | dtype | default (ms) | attributed to | flash | mem_efficient | math |",
           "|---|---|---|---|---|---|---|"]
    for r in d["records"]:
        b = r.get("backends", {})
        def c(n):
            x = b.get(n, {})
            return f"{x['ms']:.2f}" if x.get("ok") else "unavailable"
        out.append("| %d | %s | %.2f | %s | %s | %s | %s |"
                   % (r["head_dim"], r["dtype"], r["default_ms"],
                      r.get("default_matches", "?"), c("flash"),
                      c("mem_efficient"), c("math")))
    return "\n".join(out)


def t_sdpa_incumbent():
    d = _json("sdpa_incumbent_check.json")
    if not d:
        return "(sdpa_incumbent_check.py not run)"
    out = ["| what | ms | 95% CI | vs torch fp32 | vs torch fp16 | gate |",
           "|---|---|---|---|---|---|"]
    for r in d["records"]:
        ms, ci = _ms_ci(r, "ms")
        if r.get("error"):
            out.append(f"| {r['who']} | FAILED | | | | |")
        elif "vs_fp32_ref" in r:
            out.append("| %s | %s | %s | %.2fx | %.2fx | %s |"
                       % (r["who"], ms, ci, r["vs_fp32_ref"], r["vs_fp16_ref"],
                          "pass" if r["gate_pass"] else "**FAIL**"))
        else:
            out.append(f"| {r['who']} | {ms} | {ci} | | | _{r.get('note','')}_ |")
    return "\n".join(out) + "\n" + _procs_note(d)


def t_sdpa_cross():
    agg = _load(["sdpa_cross"])
    raw = analyze2.load(["sdpa_cross"])
    if not raw:
        return "(campaign not run)"
    import common as c1
    from collections import defaultdict
    by = defaultdict(list)
    err = {}
    for r in raw:
        if not r.get("ok"):
            continue
        e = (r.get("cfg", {}) or {}).get("extra", {}) or {}
        k = (r["dsl"], r["variant"], int(e.get("d", 0)),
             e.get("sdtype", "?"), e.get("pdtype", "?"))
        by[k].append(r["timing"]["median_ms"])
        if r.get("error"):
            err[k] = r["error"]
    dims = sorted({k[2] for k in by})
    pairs = [p for p in common2.SDPA_DTYPES]
    # torch's variants are backends, not algorithms, so they have no K3/FLASH
    # column to sit in. They are the denominators and are rendered separately.
    TORCH = {"TORCH_F32": "torch fp32 (published denominator)",
             "TORCH_F16": "torch fp16 (precision-matched)",
             "TORCH_MATH": "torch, math backend forced"}
    out = []
    for d in dims:
        out += [f"**head_dim = {d}**", "",
                "| lane | " + " | ".join(f"{a} {s}/{p}" for a in ("K3", "FLASH")
                                         for s, p in pairs) + " |",
                "|" + "---|" * (1 + 2 * len(pairs))]
        lanes = sorted({k[0] for k in by if k[2] == d and k[0] != "torch"})
        for ln in lanes:
            row = [ln]
            for a in ("K3", "FLASH"):
                for s, p in pairs:
                    k = (ln, a, d, s, p)
                    if k not in by:
                        row.append("--")
                        continue
                    m = c1.median_ci(by[k])["median_of_medians_ms"]
                    cell = f"{m:.2f}"
                    if err.get(k) and not err[k].get("gate_pass"):
                        cell += "*"
                    row.append(cell)
            out.append("| " + " | ".join(row) + " |")
        # The three denominators at this head dim, and the best custom kernel
        # measured against each -- which is the whole point of carrying two.
        den = {}
        for v, lab in TORCH.items():
            got = [by[k] for k in by if k[0] == "torch" and k[1] == v and k[2] == d]
            if got:
                den[v] = c1.median_ci(got[0])["median_of_medians_ms"]
        best = min((c1.median_ci(v)["median_of_medians_ms"]
                    for k, v in by.items() if k[2] == d and k[0] != "torch"
                    and not (err.get(k) and not err[k].get("gate_pass"))),
                   default=None)
        if den:
            out += ["", "| denominator | ms | best gate-passing custom kernel vs it |",
                    "|---|---|---|"]
            for v, lab in TORCH.items():
                if v not in den:
                    continue
                out.append("| %s | %.2f | %s |"
                           % (lab, den[v],
                              f"{den[v] / best:.2f}x" if best else "--"))
        out.append("")
    out += ["`*` = fails the 1e-4 gate.",
            "", "The `best gate-passing custom kernel` column uses the fastest "
            "cell in the table above at that head dim that actually passes the "
            "gate, so a number that only exists because it is wrong cannot "
            "become the speedup."]
    return "\n".join(out)


ABS_S = {
    "S3-H":  "single accumulator, `T.Pipelined`, `T.reduce_*`",
    "S3-M":  "one accumulator per d-tile, plain loop, `T.reduce_*`",
    "S3-MP": "`S3-M` + `T.Pipelined` on the same manual structure",
    "S3-L":  "`S3-M` + manual `T.shfl_down` reductions",
    "S1":    "three kernels, `S` materialized to global",
    "S2":    "two kernels, `S` never leaves registers",
}


def t_sdpa_abs():
    raw = analyze2.load(["sdpa_abstraction"])
    if not raw:
        return "(campaign not run)"
    import common as c1
    from collections import defaultdict
    by, failed = defaultdict(list), {}
    for r in raw:
        e = (r.get("cfg", {}) or {}).get("extra", {}) or {}
        k = (r["variant"], int(e.get("d", 0) or 0))
        if r.get("ok"):
            by[k].append(r["timing"]["median_ms"])
        else:
            failed[k] = (r.get("error_msg") or "?")
    dims = sorted({k[1] for k in list(by) + list(failed)})

    def med(arm, d):
        v = by.get((arm, d))
        if v:
            return c1.median_ci(v)["median_of_medians_ms"]
        return None

    def render(arms, title, note=""):
        o = [title, "",
             "| arm | what changes | " + " | ".join(f"d={d}" for d in dims) + " |",
             "|---|---|" + "---|" * len(dims)]
        for arm in arms:
            row = [arm, ABS_S.get(arm, "")]
            for d in dims:
                m = med(arm, d)
                if m is not None:
                    row.append(f"{m:.2f}")
                elif (arm, d) in failed:
                    row.append("**does not build**")
                else:
                    row.append("--")
            o.append("| " + " | ".join(row) + " |")
        if note:
            o += ["", note]
        return o

    out = render(
        ("S3-H", "S3-M", "S3-MP", "S3-L"),
        "**Within-kernel abstraction axis.** One algorithm, one tile "
        "(`block_M=64, block_N=64, D_TILE=128, threads=256`) at every head dim; "
        "only how the kernel is written changes.")
    # The honest sub-split: S3-M vs S3-L is the only pair that differs purely in
    # expression. S3-H differs in what the formulation *is*, because a single
    # accumulator forces an outer d-tile loop that redoes QK^T.
    o2 = ["", "Of those four, only **`S3-M` vs `S3-L`** isolates expression "
              "level alone — same loop structure, same accumulators, library "
              "reduction versus hand-written warp shuffles:", "",
          "| head dim | `S3-M` (`T.reduce_*`) | `S3-L` (manual shuffles) | manual costs |",
          "|---|---|---|---|"]
    for d in dims:
        a, b = med("S3-M", d), med("S3-L", d)
        o2.append("| %d | %s | %s | %s |"
                  % (d, f"{a:.2f}" if a else "--", f"{b:.2f}" if b else "--",
                     f"{100 * (b / a - 1):+.1f}%" if (a and b) else "--"))
    out += o2

    best = {}
    for d in dims:
        cand = [(med(a, d), a) for a in ("S3-H", "S3-M", "S3-MP", "S3-L")
                if med(a, d) is not None]
        if cand:
            best[d] = min(cand)
    out += [""] + render(
        ("S1", "S2", "S3-M"),
        "**Algorithmic decomposition axis — this is NOT an abstraction result.** "
        "These differ in kernel count, materialization, V re-reads and occupancy "
        "all at once.")
    if best:
        out += ["", "| head dim | best `S3` | best decomposed | fused wins by |",
                "|---|---|---|---|"]
        for d in dims:
            if d not in best:
                continue
            bm, ba = best[d]
            dec = [(med(a, d), a) for a in ("S1", "S2") if med(a, d) is not None]
            if not dec:
                continue
            dm, da = min(dec)
            out.append("| %d | %s (%.2f) | %s (%.2f) | %s |"
                       % (d, ba, bm, da, dm,
                          f"{dm / bm:.2f}x" if dm >= bm
                          else f"**loses**, {bm / dm:.2f}x slower"))
    return "\n".join(out)


def _ncu(name):
    d = _json(name)
    return (d or {}).get("records") or []


def t_ncu_fused():
    recs = _ncu("ncu_fused.json")
    if not recs:
        return "(ncu_collect2.py --op fused not run)"
    out = ["| lane | epilogue | kernel | regs | occupancy | smem/block | "
           "DRAM | grid |", "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        if not r.get("ok"):
            out.append(f"| {r['dsl']} | | FAILED | | | | | |")
            continue
        epi = "-"
        for tok in (r.get("set") or "").split(","):
            if tok.startswith("x_epilogue="):
                epi = tok.split("=", 1)[1]
        if r["variant"] != "GBGS":
            continue
        for k in r["kernels"]:
            if k["is_setup"]:
                continue
            smem = (k.get("smem_static_B", 0) or 0) + (k.get("smem_dyn_B", 0) or 0)
            out.append("| %s | %s | %s | %d | %.1f%% | %.0f B | %.2f GB | %d |"
                       % (r["dsl"], epi, k["kernel"][:28], int(k.get("regs", 0)),
                          k.get("occupancy_pct", 0.0), smem,
                          k["dram_total_GB"], int(k.get("grid", 0))))
    out += ["", "One-shot host-side weight conversion is excluded (it launches "
                "once, not once per call); it is reported separately in the "
                "JSON as `setup_dram_GB`."]
    return "\n".join(out)


def t_ncu_sdpa():
    recs = _ncu("ncu_sdpa.json")
    if not recs:
        return "(ncu_collect2.py --op sdpa not run)"
    out = ["| lane | algo | d | scores/probs | kernel | regs | occupancy | "
           "DRAM | share of algo DRAM |", "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        if not r.get("ok"):
            out.append(f"| {r['dsl']} | {r['variant']} | | | FAILED | | | | |")
            continue
        st = dict(t.split("=", 1) for t in (r.get("set") or "").split(",") if "=" in t)
        tot = r.get("algo_dram_GB") or 0.0
        for k in r["kernels"]:
            if k["is_setup"]:
                continue
            out.append("| %s | %s | %s | %s/%s | %s | %d | %.1f%% | %.2f GB | %s |"
                       % (r["dsl"], r["variant"], st.get("x_d", "?"),
                          st.get("x_sdtype", "?"), st.get("x_pdtype", "?"),
                          k["kernel"][:24], int(k.get("regs", 0)),
                          k.get("occupancy_pct", 0.0), k["dram_total_GB"],
                          f"{100.0 * k['dram_total_GB'] / tot:.0f}%" if tot else "--"))
    return "\n".join(out)


ALL_TAGS = ["fused_matched", "fused_native", "fused_epilogue", "fused_cast",
            "fused_abstraction", "sdpa_cross", "sdpa_abstraction"]

# Which lanes actually compile cold every time. Only TileLang calls
# `tilelang.disable_cache()`; Triton keeps its own on-disk cache and the two
# CUDA lanes are served by ninja out of a persistent TORCH_EXTENSIONS_DIR that
# Phase 1's `common.py` pins deliberately. A median over all builds therefore
# compares one cold lane against three warm ones.
CACHE_POLICY = {
    "tilelang": ("disabled", "`tilelang.disable_cache()` — every build is cold"),
    "tilelang_abs": ("disabled", "`tilelang.disable_cache()` — every build is cold"),
    "triton": ("kept", "Triton's own cache, keyed by source hash"),
    "cuda_noptx": ("kept", "ninja, persistent `TORCH_EXTENSIONS_DIR`"),
    "cuda_unlimited": ("kept", "ninja, persistent `TORCH_EXTENSIONS_DIR`"),
    "torch": ("n/a", "no compilation step"),
}
COLD_S = 5.0     # a build this slow cannot be a cache hit


def _fmt_ci(st, unit=" s"):
    if not st:
        return "—"
    m = st.get("median_of_medians_ms")
    if m is None:
        return "—"
    if "ci95_lo_ms" in st:
        return "%.2f%s [%.2f, %.2f]" % (m, unit, st["ci95_lo_ms"],
                                        st["ci95_hi_ms"])
    return "%.2f%s" % (m, unit)


def t_compile():
    """Two blocks: the controlled cold/warm measurement, then the campaign
    census that shows why the campaign numbers alone cannot be ranked."""
    import statistics as stats
    from collections import defaultdict

    out = []
    doc = _json("compile_cold.json")
    if doc:
        out += ["**Controlled** — arm `%s`, %d repetitions, a fresh empty cache "
                "directory per cold repetition (`compile_cold.py`):"
                % (doc.get("arm", "G"), doc.get("reps", 0)), "",
                "| lane | cold build | warm build | cold ÷ warm |",
                "|---|---|---|---|"]
        lanes = doc.get("lanes", {})
        order = sorted(lanes, key=lambda k: -(
            (lanes[k].get("cold_stats") or {}).get("median_of_medians_ms") or 0))
        for lane in order:
            r = lanes[lane]
            cs = r.get("cold_stats")
            ws = r.get("warm_stats")
            ratio = "—"
            if cs and ws and ws.get("median_of_medians_ms"):
                ratio = "%.0f×" % (cs["median_of_medians_ms"]
                                   / ws["median_of_medians_ms"])
            warm = _fmt_ci(ws) if ws else ("n/a — " + r.get("warm_note", ""))
            out += ["| %s | %s | %s | %s |" % (lane, _fmt_ci(cs), warm, ratio)]
        out += [""]

    recs = [r for r in analyze2.load(ALL_TAGS)
            if r.get("ok") and isinstance(r.get("compile_s"), (int, float))]
    if not recs:
        return "\n".join(out) or "(no campaigns run)"
    by = defaultdict(list)
    seen, first_slow, rep_slow = set(), defaultdict(int), defaultdict(int)
    for r in sorted(recs, key=lambda r: r.get("t_start") or 0):
        by[r["dsl"]].append(r["compile_s"])
        # Identity of the *source* being compiled, not of the timing cell: a
        # cache hit is possible exactly when this key has been built before.
        k = (r["dsl"], r.get("variant"), json.dumps(
            (r.get("cfg") or {}).get("extra") or {}, sort_keys=True))
        tgt = first_slow if k not in seen else rep_slow
        if r["compile_s"] > COLD_S:
            tgt[r["dsl"]] += 1
        seen.add(k)
    out += ["**Campaign census** — every build the campaigns performed, "
            "which is *not* a like-for-like ranking:", "",
            "| lane | cache policy | builds | median | builds > %gs | of those, "
            "first-time source | repeat source |" % COLD_S,
            "|---|---|---|---|---|---|---|"]
    for lane in sorted(by, key=lambda k: -stats.median(by[k])):
        v = sorted(by[lane])
        n_slow = sum(1 for x in v if x > COLD_S)
        out += ["| %s | %s | %d | %.2f s | %d | %d | %d |"
                % (lane, CACHE_POLICY.get(lane, ("?", ""))[0], len(v),
                   stats.median(v), n_slow, first_slow[lane], rep_slow[lane])]
    out += ["", "Cache policy: "
            + "; ".join(f"**{k}** — {v[1]}" for k, v in CACHE_POLICY.items()
                        if k in by) + "."]
    return "\n".join(out)


def t_anchor():
    import anchor_check2
    return anchor_check2.table(anchor_check2.collect())


def t_raw():
    return _demote(analyze2.table_raw(_load(
        ["fused_matched", "fused_native", "fused_epilogue", "fused_cast",
         "fused_abstraction"])))


def t_raw_sdpa():
    return _demote(analyze2.table_raw(_load(["sdpa_cross", "sdpa_abstraction"])))


SUBS = {
    "TABLE_LADDER": t_ladder,
    "TABLE_SPEEDUP": t_speedup,
    "TABLE_WCACHE": t_wcache,
    "TABLE_EPILOGUE": t_epilogue,
    "TABLE_CAST": t_cast,
    "TABLE_FABS": t_fabs,
    "TABLE_INCUMBENT_FUSED": t_incumbent_fused,
    "TABLE_GATE": t_gate,
    "TABLE_SDPA_AUDIT": t_sdpa_audit,
    "TABLE_SDPA_INCUMBENT": t_sdpa_incumbent,
    "TABLE_SDPA_CROSS": t_sdpa_cross,
    "TABLE_SDPA_ABS": t_sdpa_abs,
    "TABLE_NCU_FUSED": t_ncu_fused,
    "TABLE_NCU_SDPA": t_ncu_sdpa,
    "TABLE_COMPILE": t_compile,
    "TABLE_ANCHOR": t_anchor,
    "TABLE_RAW": t_raw,
    "TABLE_RAW_SDPA": t_raw_sdpa,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only verify that every marker resolves")
    ap.add_argument("--out", default=os.path.join(HERE, "PHASE2_REPORT.built.md"))
    a = ap.parse_args()

    with open(SRC) as f:
        text = f.read()

    missing, empty = [], []
    for name, fn in SUBS.items():
        marker = f"<!--{name}-->"
        if marker not in text:
            missing.append(name)
            continue
        try:
            body = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            body = (f"_table {name} failed to build: {type(e).__name__}: {e}_\n\n"
                    f"```\n{traceback.format_exc(limit=3)}\n```")
            empty.append(name)
        # Flag a degraded table, not just an empty one. A stale lookup renders a
        # perfectly well-formed "(campaign not run)" that a reader takes at face
        # value, so any body that is a bare parenthetical counts as a failure --
        # `--check` is supposed to catch precisely that.
        b = body.strip()
        if (b.count("\n") <= 2 and "|" in b) or (b.startswith("(") and b.endswith(")")):
            empty.append(name)
        text = text.replace(marker, body)

    left = re.findall(r"<!--([A-Z_]+)-->", text)
    print(f"[build_report2] {len(SUBS)} tables; markers unresolved: {left or 'none'}; "
          f"markers absent from source: {missing or 'none'}; "
          f"empty/failed: {empty or 'none'}")

    if not a.check:
        with open(a.out, "w") as f:
            f.write(text)
        print(f"-> {a.out}")
    return 1 if (left or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
