#!/usr/bin/env python3
"""Inject generated tables into PHASE1_REPORT.md placeholders.

Transcribing numbers by hand into a report is how a study acquires errors that
nobody can trace later. Every table in the report is generated from the raw
per-process JSON and substituted here, so the document and the data cannot drift.

Prose placeholders (DECISION_TABLE, LIMITATIONS, ABSTRACTION_*) are left alone if
they have already been written; they require judgement, not generation.

usage: python build_report.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

REPORT = os.path.join(HERE, "PHASE1_REPORT.md")


def run_py(script, args):
    p = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                       capture_output=True, text=True, cwd=HERE, timeout=900)
    return p.stdout


def matched_tables():
    out = run_py("analyze.py", ["--tag", "matched"])
    # drop the generated H1 title, keep the tables
    lines = out.splitlines()
    keep = [l for l in lines if not l.startswith("# Phase-1 tables")]
    body = "\n".join(keep).strip()
    sec = run_py("analyze.py", ["--tag", "matched", "--geom", "secondary"])
    sec_lines = [l for l in sec.splitlines() if not l.startswith("# Phase-1 tables")]
    # only the matched + decomposition part of the secondary geometry
    txt = "\n".join(sec_lines)
    start = txt.find("### Matched configuration")
    end = txt.find("### Error against")
    secondary = txt[start:end].strip() if start >= 0 and end > start else ""
    if secondary:
        body += ("\n\n### Secondary geometry (BM=128 BN=256 BK=32)\n\n"
                 "Carried so no conclusion is hostage to one tile shape.\n\n"
                 + secondary.split("\n", 1)[1].strip())
    return body


def _demote(md):
    """Sub-study tables emit `## Title`, but they are injected *under* a numbered
    `##` section. Left alone they break the document outline."""
    return "\n".join(("#" + l) if l.startswith("## ") else l for l in md.splitlines())


def sass_table():
    out = run_py("report_tables.py", [])
    i = out.find("## Generated-code census")
    return _demote(out[i:].strip()) if i >= 0 else "*(no SASS census)*"


def ncu_table():
    p = os.path.join(common.RESULTS_DIR, "ncu_matched.json")
    if not os.path.exists(p):
        return "*(no ncu data)*"
    recs = json.load(open(p))["records"]
    by = {(r["dsl"], r["variant"]): r["metrics"] for r in recs if r.get("ok")}
    L = ["| DSL | variant | ncu ms | tensor-pipe % | occupancy % | regs | smem B | DRAM MiB | L2 hit % | achieved TC TFLOP/s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for d in ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"]:
        for v in "ABCD":
            m = by.get((d, v))
            if not m:
                continue
            smem = (m.get("smem_static_B", 0) or 0) + (m.get("smem_dyn_B", 0) or 0)
            dram = ((m.get("dram_rd_B", 0) or 0) + (m.get("dram_wr_B", 0) or 0)) / 2**20
            L.append(f"| {d} | {v} | {m.get('measured_ms', 0):.3f} | "
                     f"{m.get('tensor_pipe_pct', 0):.1f} | {m.get('occupancy_pct', 0):.1f} | "
                     f"{m.get('regs', 0):.0f} | {smem:.0f} | {dram:.0f} | "
                     f"{m.get('l2_hit_pct', 0):.1f} | {m.get('tensor_tflops_achieved', 0):.1f} |")
    return "\n".join(L)


def sweeps():
    body = _demote(run_py("sweep_tables.py",
                  ["--which", "kc_sweep", "casting", "pipeline", "native_tuned"]).strip())
    # The native-tuning winners are re-measured at the full 5-process protocol;
    # a number selected as the minimum over 19 points at 2 processes is partly
    # selection noise and must not be reported as if it were not.
    conf = run_py("confirm_winners.py", ["--table"]).strip()
    # A markdown table with a header and no rows reads as "we measured this and
    # found nothing", which is the opposite of "this campaign has not run yet".
    # Emit it only once it has at least one data row.
    if sum(1 for l in conf.splitlines() if l.startswith("| ") and "---" not in l) > 1:
        body += "\n" + _demote(conf)
    return body


def abstraction():
    return _demote(run_py("sweep_tables.py", ["--which", "abstraction"]).strip())


def abstraction_rule():
    return run_py("abstraction_rule.py", []).strip()


def loc_table():
    return run_py("loc.py", []).strip()


def anchor():
    return run_py("anchor_check.py", []).strip()


def compile_table():
    """Cold compile cost -- the other half of the exploration-cost argument.

    The matched table's `compile s` column is warm-cache and inverts the true
    ordering, so it must not be the number quoted for search cost."""
    p = os.path.join(common.RESULTS_DIR, "compile_cost.json")
    if not os.path.exists(p):
        return "*(no cold-compile data)*"
    recs = [r for r in json.load(open(p))["records"] if "error" not in r]
    if not recs:
        return "*(no cold-compile data)*"
    by = {}
    for r in recs:
        by.setdefault(r["dsl"], {})[r["variant"]] = r["cold_compile_s"]
    fastest = min(min(v.values()) for v in by.values())
    L = ["**Cold compile cost — empty cache to launchable kernel.** The "
         "`compile s` column in §3 is *warm-cache* time and inverts this "
         "ordering (CUDA 0.2 s is a `.so` cache hit, TileLang 4–10 s is a "
         "partial miss). Measured in isolated, empty `TORCH_EXTENSIONS_DIR` / "
         "`TRITON_CACHE_DIR` / `TILELANG_CACHE_DIR`, one process each.\n",
         "| DSL | variant A | variant D | ÷ fastest | cost of a 19-point grid search |",
         "|---|---|---|---|---|"]
    for d in ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"]:
        v = by.get(d)
        if not v:
            continue
        a, dd = v.get("A"), v.get("D")
        ref = dd if dd else a
        grid = (ref or 0) * 19
        span = f"~{grid:.0f} s" if grid < 90 else f"~{grid / 60:.1f} min"
        L.append(f"| {d} | {f'{a:.1f} s' if a else '—'} | "
                 f"{f'{dd:.1f} s' if dd else '—'} | {ref / fastest:.0f}× | "
                 f"{span} |")
    L.append("\nThe last column is the point: an equal *budget* in points is not "
             "an equal budget in time. Compiling the same 19 configurations costs "
             "Triton around twenty seconds and the CUDA lanes around twelve "
             "minutes — before a single measurement is taken.")
    return "\n".join(L)


SUBS = {
    "TABLES_MATCHED": matched_tables,
    "TABLE_SASS": sass_table,
    "TABLE_NCU": ncu_table,
    "TABLES_SWEEPS": sweeps,
    "TABLE_ABSTRACTION": abstraction,
    "ABSTRACTION_RULE": abstraction_rule,
    "TABLE_LOC": loc_table,
    "ANCHOR_CHECK": anchor,
    "TABLE_COMPILE": compile_table,
}


def main():
    with open(REPORT) as f:
        doc = f.read()
    for name, fn in SUBS.items():
        marker = f"<!--{name}-->"
        if marker not in doc:
            print(f"  (placeholder {name} already filled or absent)")
            continue
        try:
            doc = doc.replace(marker, fn())
            print(f"  filled {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
    with open(REPORT, "w") as f:
        f.write(doc)
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
