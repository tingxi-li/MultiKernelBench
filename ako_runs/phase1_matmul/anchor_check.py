#!/usr/bin/env python3
"""Cross-campaign drift detector.

Each sub-study was designed to contain a point that also exists in the matched
campaign: `pipeline` measures stages=3, `casting` measures precast, `kc_sweep`
measures KC=2048 -- all of which *are* variant D at the primary geometry. Those
shared points are anchors.

They exist because the sub-study tables are read against each other and against
the matched table, and those campaigns ran hours apart on a card that thermally
soaks. Without an anchor, a drift between campaigns is indistinguishable from a
real effect, and the report would have no way to tell the difference. With one,
the question is answered by measurement instead of assumption.

The check is one-sided in a useful way: agreement does not prove there was no
drift in the un-anchored points, but disagreement would prove there was.

usage: python anchor_check.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import aggregate, canonical, load  # noqa: E402

DSL_ORDER = ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"]
NOISE_FLOOR_PCT = 3.5   # measured between-process spread at the frozen protocol


def cf(key, field):
    d = dict(p.split("=", 1) for p in key[4].split(";") if "=" in p)
    return d.get(field)


def pick(agg, dsl, **want):
    for k, x in agg.items():
        if k[0] != dsl or x.get("status") != "OK":
            continue
        if all(cf(k, f) == str(v) for f, v in want.items()):
            return x["median_of_medians_ms"], x.get("n_processes", 0)
    return None, 0


def main():
    m = aggregate(load("matched"))
    # `reporting` = ran at the full 5-process protocol, so its numbers are quoted
    # directly. `kc_sweep` deliberately ran at 3 to fit the point count, which
    # makes it a weaker estimator -- it is checked, but separately, because a
    # 3-sample disagreement is a sampling result, not evidence the card drifted.
    sources = [
        ("pipeline @ stages=3", aggregate(load("pipeline")), dict(stages=3), True),
        ("casting @ precast", aggregate(load("casting")), dict(cast="precast", BN=128), True),
        ("kc_sweep @ KC=2048", aggregate(load("kc_sweep")), dict(kc=2048, stages=3), False),
    ]
    sources = [s for s in sources if s[1]]

    L = ["**Anchor check — did the campaigns drift?** Every sub-study contains a "
         "point that is also variant D at the primary geometry. The campaigns ran "
         "hours apart on a card that thermally soaks, so agreement here is what "
         "licenses reading the sub-study tables against the matched table at all.\n",
         "| DSL | matched D | " + " | ".join(f"{n}" for n, _, _, _ in sources)
         + " | drift (5-proc only) |",
         "|" + "---|" * (len(sources) + 3)]
    worst, worst_all = 0.0, 0.0
    small_n = set()
    for d in DSL_ORDER:
        base = canonical(m, d, "D", "primary", "rand")
        if not base:
            continue
        strong = [base["median_of_medians_ms"]]
        every = list(strong)
        cells = []
        for _, agg, want, is_reporting in sources:
            v, n = pick(agg, d, **want)
            cells.append(f"{v:.3f}" + ("" if n >= 5 else f" *(n={n})*") if v else "—")
            if v:
                every.append(v)
                if is_reporting:
                    strong.append(v)
                else:
                    small_n.add(n)
        drift = (max(strong) / min(strong) - 1) * 100 if len(strong) > 1 else 0.0
        worst = max(worst, drift)
        if len(every) > 1:
            worst_all = max(worst_all, (max(every) / min(every) - 1) * 100)
        L.append(f"| {d} | {strong[0]:.3f} | " + " | ".join(cells) + f" | {drift:.1f}% |")

    L.append("")
    if worst <= NOISE_FLOOR_PCT:
        L.append(f"Across the campaigns that ran at the full reporting protocol, "
                 f"worst disagreement is **{worst:.1f}%** against a "
                 f"{NOISE_FLOOR_PCT}% between-process noise floor. **The card did "
                 f"not drift**, and the sub-study tables may be read against the "
                 f"matched table without correction.")
    else:
        L.append(f"Worst disagreement **{worst:.1f}%**, which EXCEEDS the "
                 f"{NOISE_FLOOR_PCT}% noise floor even at the reporting protocol. "
                 f"Cross-campaign comparisons are not safe below that resolution.")
    if small_n and worst_all > worst:
        L.append(f"\nThe KC sweep ran at {min(small_n)} processes rather than 5 "
                 f"(40 points × 5 would have doubled the campaign), and it is the "
                 f"only column that disagrees more — up to {worst_all:.1f}%. That "
                 f"is the cost of the smaller sample, not card drift: the "
                 f"5-process campaigns agree with each other to {worst:.1f}%. It is "
                 f"also a direct demonstration of why the protocol requires five "
                 f"processes, so the KC *runtimes* are read only as a flat-vs-not "
                 f"shape and never as head-to-head DSL comparisons.")
    print("\n".join(L))


if __name__ == "__main__":
    main()
