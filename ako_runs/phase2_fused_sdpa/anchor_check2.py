#!/usr/bin/env python3
"""Cross-campaign drift check.

Several configurations are measured in more than one campaign. Because the
campaigns run at different times on a card that thermally soaks, those repeats
are the only direct evidence for how much a number may be compared across
campaigns -- and therefore for which comparisons in the report are legitimate.

Two things make a cell an anchor:
  * `torch:fp16 / GBGS / cached` appears in both `fused_matched` and
    `fused_cast` (torch always casts in-region, so its "cast" job is a plain
    repeat of the matched cell).
  * `cuda_unlimited`'s default epilogue IS `smem`, so a `fused_matched` cell
    that never names it and a `fused_epilogue` cell that names it explicitly are
    the same configuration.

usage: python anchor_check2.py [--json-out results/anchor_check2.json]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
import common  # noqa: E402
import analyze2  # noqa: E402

TAGS = ("fused_matched", "fused_native", "fused_epilogue", "fused_cast")
# Phase 1's measured noise floor for the same protocol on the same card.
NOISE_FLOOR_PCT = 3.5


def norm(r):
    e = (r.get("cfg", {}) or {}).get("extra", {}) or {}
    epi = e.get("epilogue", "-")
    if analyze2.lane(r) == "cuda_unlimited" and epi == "-":
        epi = "smem"
    return (analyze2.lane(r), r["variant"], e.get("wcache", "cached"), epi,
            (r.get("cfg", {}) or {}).get("cast", "precast"))


def collect(tags=TAGS):
    by = defaultdict(list)
    for tag in tags:
        for r in analyze2.load([tag]):
            if r.get("ok"):
                by[(tag, norm(r))].append(r["timing"]["median_ms"])
    cells = defaultdict(dict)
    for (tag, k), v in by.items():
        cells[k][tag] = common.median_ci(v)["median_of_medians_ms"]
    return {k: d for k, d in cells.items() if len(d) > 1}


def table(cells, tags=TAGS):
    if not cells:
        return "(no configuration appears in more than one campaign)"
    out = ["| configuration | " + " | ".join(tags) + " | spread |",
           "|---|" + "---|" * (len(tags) + 1)]
    worst = 0.0
    for k, d in sorted(cells.items()):
        vals = list(d.values())
        sp = 100.0 * (max(vals) - min(vals)) / min(vals)
        worst = max(worst, sp)
        out.append("| %s | %s | %.2f%% |"
                   % ("/".join(map(str, k[:4])),
                      " | ".join(f"{d[t]:.4f}" if t in d else "--" for t in tags),
                      sp))
    verdict = ("within" if worst <= NOISE_FLOOR_PCT else "ABOVE")
    out += ["", f"{len(cells)} anchor cells; worst drift **{worst:.2f}%**, "
                f"{verdict} Phase 1's measured {NOISE_FLOOR_PCT}% noise floor.",
            "", "Consequence for reading this report: differences **within** one "
                "campaign are comparable at the CI shown; differences **between** "
                f"campaigns carry an additional ~{worst:.0f}% of uncertainty and "
                "any effect smaller than that is not resolvable across tables."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(
        common2.RESULTS_DIR, "anchor_check2.json"))
    a = ap.parse_args()
    cells = collect()
    print(table(cells))
    common2.write_json(a.json_out, {
        "noise_floor_pct": NOISE_FLOOR_PCT,
        "cells": [{"config": list(k), "by_tag": d} for k, d in cells.items()]})


if __name__ == "__main__":
    main()
