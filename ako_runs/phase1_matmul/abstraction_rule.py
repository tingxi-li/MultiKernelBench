#!/usr/bin/env python3
"""Apply the abstraction study's decision rule mechanically.

The rule was specified in advance with hard thresholds:

  H1 ~= M2 within 3% and similar SASS -> abstraction is mainly ergonomic
  H1 > M2 by more than 10%            -> compiler-managed pipeline materially better
  M2 > H1 by more than 10%            -> the high-level abstraction leaves performance on the table
  H1 >> H2                            -> the pipeline, not T.gemm alone, is the important feature

Two reasons this is a script and not a paragraph:

1. A pre-registered threshold applied by hand after seeing the numbers is not a
   pre-registered threshold. Encoding it means the verdict is a function of the
   data, not of which comparison looked interesting afterwards.

2. `ABSTRACTION_SPECS` gives H1 three pipeline stages and M2 two, so the literal
   H1-vs-M2 comparison confounds "compiler-generated vs hand-written pipeline"
   with pipeline depth. Removing that confound means moving one arm to the
   other's depth. Only one of them can move: `num_stages` is an integer to the
   compiler-managed pipeline, while M2's buffer parity is hand-unrolled at depth
   two and a third stage is a rewrite. So the matched-depth comparison is
   H1@stages=2 vs M2, run as a separate campaign (`abstraction_depth`) because
   the arm's own guard rejects an off-spec depth unless explicitly opted into.

   If more than one matched depth were available and the verdicts disagreed, the
   confound would be load-bearing and the rule could not be applied -- that case
   is handled and reported as a null result rather than resolved by picking the
   flattering pair.

The 3% band is also checked against the measured noise floor: with ~3.5%
between-process spread, "within 3%" is not distinguishable from "equal", so an
"ergonomic" verdict is reported as *consistent with* rather than *demonstrating*
equivalence.

usage: python abstraction_rule.py [--tags abstraction abstraction_depth]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import aggregate, load  # noqa: E402

ERGONOMIC_BAND = 0.03
MATERIAL_BAND = 0.10


def cfgparts(s):
    return dict(p.split("=", 1) for p in (s or "").split(";") if "=" in p)


def index(tags=("abstraction", "abstraction_depth")):
    """{(variant, stages): aggregate-record} for the tilelang abstraction arms.

    Reads the depth-control campaign alongside the main one: H1 at 2 stages had
    to be run separately because the arm's own guard rejects an off-spec depth
    unless it is explicitly opted into."""
    out = {}
    for tag in ([tags] if isinstance(tags, str) else tags):
        for k, x in aggregate(load(tag)).items():
            if x.get("status") != "OK":
                continue
            st = cfgparts(k[4]).get("stages")
            out[(k[1], int(st) if st else None)] = x
    return out


def ms(rec):
    return rec["median_of_medians_ms"] if rec else None


def spread(rec):
    return rec.get("rel_spread_pct") if rec else None


def verdict(h1, m2):
    """Return (label, rel) where rel = (M2 - H1)/H1; positive means H1 faster."""
    if h1 is None or m2 is None:
        return "insufficient data", None
    rel = (m2 - h1) / h1
    if abs(rel) <= ERGONOMIC_BAND:
        return "abstraction mainly ergonomic", rel
    if rel > MATERIAL_BAND:
        return "compiler-managed pipeline materially better", rel
    if rel < -MATERIAL_BAND:
        return "high-level abstraction leaves performance on the table", rel
    return "between the 3% and 10% bands -- rule gives no verdict", rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=["abstraction", "abstraction_depth"])
    ap.add_argument("--md", action="store_true", help="emit markdown for the report")
    a = ap.parse_args()
    ix = index(a.tags)
    if not ix:
        print(f"*(no abstraction results yet)*")
        return 1

    H1_3, H1_2 = ix.get(("H1", 3)), ix.get(("H1", 2))
    M2_2, M2_3 = ix.get(("M2", 2)), ix.get(("M2", 3))
    H2, M1, S1 = ix.get(("H2", 1)), ix.get(("M1", 1)), ix.get(("S1", 1))

    noise = max([s for s in (spread(r) for r in ix.values()) if s] or [0.0])

    L = []
    L.append("**Decision rule, applied at both matched pipeline depths.** "
             "`ABSTRACTION_SPECS` gives H1 3 stages and M2 2, so the literal "
             "H1-vs-M2 pair confounds compiler-vs-hand with depth. Both depths "
             "are addressed below:\n")
    L.append("| depth | H1 (compiler pipeline) | M2 (hand-written pipeline) | (M2−H1)/H1 | rule verdict |")
    L.append("|---|---|---|---|---|")
    # M2 at 3 stages is not a missing measurement -- it is not expressible. M2's
    # double buffer is hand-written with a static parity unroll, so a third stage
    # is a rewrite. Say that, rather than leaving a dash that reads like an
    # oversight.
    NOT_EXPRESSIBLE = ("*not a parameter* — M2's buffer parity is hand-unrolled "
                       "at depth 2; a third stage is a rewrite")
    rows = []
    for depth, h1r, m2r in ((2, H1_2, M2_2), (3, H1_3, M2_3)):
        lab, rel = verdict(ms(h1r), ms(m2r))
        rows.append((depth, lab, rel))
        m2cell = f"{ms(m2r):.3f} ms" if m2r else (NOT_EXPRESSIBLE if depth == 3 else "—")
        note = lab if rel is not None else ("no verdict — see left" if depth == 3
                                            else "insufficient data")
        L.append(f"| stages={depth} | "
                 f"{f'{ms(h1r):.3f} ms' if h1r else '—'} | {m2cell} | "
                 f"{f'{rel:+.1%}' if rel is not None else '—'} | {note} |")

    got = [r for r in rows if r[2] is not None]
    L.append("")
    if len(got) == 2 and got[0][1] != got[1][1]:
        L.append(f"**The two matched-depth verdicts disagree** "
                 f"(stages=2 → *{got[0][1]}*; stages=3 → *{got[1][1]}*). Pipeline "
                 f"depth is therefore load-bearing in this comparison and the rule "
                 f"cannot be applied to a single H1-vs-M2 number. Reported as a "
                 f"null result rather than resolved by choosing a depth.")
    elif len(got) == 1:
        d, lab, rel = got[0]
        L.append(f"**Verdict at the one depth where both arms exist "
                 f"(stages={d}): {lab}** ({rel:+.1%}).")
        L.append(f"\nOnly one matched-depth comparison is possible, and the reason "
                 f"is itself a result: raising the compiler-managed pipeline to any "
                 f"depth is an integer, while raising the hand-written one requires "
                 f"re-deriving the buffer parity. The confound was removed by moving "
                 f"the arm that *can* move.")
        if abs(rel) <= ERGONOMIC_BAND:
            L.append(f"\nCaveat: the measured between-process spread in this "
                     f"campaign is {noise:.1f}%, so a ≤{ERGONOMIC_BAND:.0%} gap is "
                     f"*consistent with* equivalence rather than a demonstration of "
                     f"it. The apparatus cannot resolve differences this small.")
    elif got:
        L.append(f"**Verdict: {got[0][1]}** — consistent at every depth measured.")

    # H1 vs H2 -- is the pipeline, or T.gemm alone, the important feature?
    L.append("\n**H1 vs H2 — is the pipeline or `T.gemm` the important feature?**\n")
    L.append("| comparison | ms | ratio | isolates |")
    L.append("|---|---|---|---|")
    if H1_3 and H2:
        L.append(f"| H1 (3 stages) | {ms(H1_3):.3f} | — | — |")
        L.append(f"| H2 (1 stage) | {ms(H2):.3f} | **{ms(H2) / ms(H1_3):.2f}×** | "
                 f"compiler software pipelining, `T.gemm` held constant |")
    if H2 and M1:
        L.append(f"| M1 (hybrid, 1 stage) | {ms(M1):.3f} | {ms(M1) / ms(H2):.2f}× vs H2 | "
                 f"cost of the high-level interface with pipelining off |")
    if M1 and S1:
        L.append(f"| S1 (SIMT fp32 FMA) | {ms(S1):.3f} | {ms(S1) / ms(M1):.2f}× vs M1 | "
                 f"tensor cores — a hardware control, **not** abstraction overhead |")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
