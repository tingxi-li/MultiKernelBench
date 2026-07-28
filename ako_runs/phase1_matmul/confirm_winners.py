#!/usr/bin/env python3
"""Re-measure the native-tuning winners at the full 5-process protocol.

The native-tuning search runs at 2 repeats because it is a search: 76 points at
5 repeats would cost more than the rest of the study combined, and a search only
needs to rank. But a *reported* number must not come from a 2-process median --
with a ~3.5% between-process spread, picking the minimum over 19 points at 2
repeats is exactly the setup where the winner is partly luck. The selected point
is therefore re-measured under the same protocol as every other reported number.

This also re-measures each DSL's runner-up. If the winner and runner-up swap
places under the full protocol, the search resolution was below the noise floor
and the report must say so instead of naming a winner.

usage:
  python confirm_winners.py --emit          # write jobs/confirm.json
  python confirm_winners.py --table         # after the confirm campaign runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
from analyze import aggregate, load  # noqa: E402

DSL_ORDER = ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"]
TOP_N = 2


def cfgparts(cfgstr):
    return dict(p.split("=", 1) for p in (cfgstr or "").split(";") if "=" in p)


def ranked(tag="native_tuned"):
    """{dsl: [(ms, setstr), ...]} sorted fastest-first."""
    agg = aggregate(load(tag))
    out = {d: [] for d in DSL_ORDER}
    for k, x in agg.items():
        if x.get("status") != "OK":
            continue
        dsl, cfgstr = k[0], k[4]
        p = cfgparts(cfgstr)
        setstr = (f"BM={p.get('BM')},BN={p.get('BN')},BK={p.get('BK')},"
                  f"stages={p.get('stages')},kc={p.get('kc')}")
        out.setdefault(dsl, []).append((x["median_of_medians_ms"], setstr, x))
    for d in out:
        out[d].sort(key=lambda t: t[0])
    return out


def emit(path):
    r = ranked()
    jobs = []
    for d in DSL_ORDER:
        for ms, setstr, _ in r.get(d, [])[:TOP_N]:
            jobs.append({"dsl": d, "variant": "D", "geom": "primary", "set": setstr})
    if not jobs:
        print("no native_tuned results yet -- nothing to confirm")
        return 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(jobs, f, indent=1)
    print(f"-> {path}  ({len(jobs)} points, top-{TOP_N} per DSL)")
    for j in jobs:
        print(f"   {j['dsl']:<16s} {j['set']}")
    return 0


def table():
    search = ranked("native_tuned")
    conf = aggregate(load("confirm"))
    conf_by = {}
    for k, x in conf.items():
        if x.get("status") == "OK":
            p = cfgparts(k[4])
            key = (k[0], f"BM={p.get('BM')},BN={p.get('BN')},BK={p.get('BK')},"
                         f"stages={p.get('stages')},kc={p.get('kc')}")
            conf_by[key] = x

    L = ["\n### Native-tuning winners re-measured at the full protocol\n",
         "The search ran at 2 processes per point (it only has to rank); the two "
         "best points per DSL are re-measured at 5 processes, like every other "
         "reported number. A rank flip between the two columns means the search "
         "resolved below the noise floor.\n",
         "| DSL | configuration | search ms (2 proc) | confirmed ms (5 proc) | 95% CI | rank held |",
         "|---|---|---|---|---|---|"]
    bests = {}
    for d in DSL_ORDER:
        top = search.get(d, [])[:TOP_N]
        confirmed = []
        for ms, setstr, _ in top:
            c = conf_by.get((d, setstr))
            cms = c["median_of_medians_ms"] if c else None
            ci = (f"[{c['ci95_lo_ms']:.3f}, {c['ci95_hi_ms']:.3f}]"
                  if c and c.get("ci95_lo_ms") is not None else "—")
            confirmed.append((ms, setstr, cms, ci))
        if len(confirmed) >= 2 and confirmed[0][2] and confirmed[1][2]:
            held = "yes" if confirmed[0][2] <= confirmed[1][2] else "**NO — flipped**"
        else:
            held = "—"
        for i, (ms, setstr, cms, ci) in enumerate(confirmed):
            L.append(f"| {d if i == 0 else ''} | `{setstr}` | {ms:.3f} | "
                     f"{f'{cms:.3f}' if cms else '—'} | {ci} | "
                     f"{held if i == 0 else ''} |")
        if confirmed and confirmed[0][2]:
            bests[d] = min(c[2] for c in confirmed if c[2])

    if len(bests) > 1:
        lo, hi = min(bests.values()), max(bests.values())
        order = sorted(bests.items(), key=lambda kv: kv[1])
        L.append(f"\n**Confirmed tuned ranking:** "
                 + " < ".join(f"{d} {v:.3f}" for d, v in order)
                 + f" — spread **{hi / lo:.2f}×**.")
        # Every confirmed number sits above its search number. Two causes push the
        # same way and neither is drift: taking the MINIMUM over ~19 two-process
        # medians selects downward, and the confirm campaign ran after a further
        # hour of continuous load. Both are reasons to quote the confirmed column.
        deltas = []
        for d in DSL_ORDER:
            for ms, setstr, cms, _ in [c for c in
                                       [(m, s, conf_by.get((d, s), {}).get("median_of_medians_ms"), None)
                                        for m, s, _ in search.get(d, [])[:TOP_N]] if c[2]]:
                deltas.append(cms / ms - 1)
        if deltas and min(deltas) > 0:
            L.append(f"\nEvery confirmed number is higher than its search number "
                     f"(+{min(deltas):.1%} to +{max(deltas):.1%}). That is expected: "
                     f"the search reports the *minimum* over ~19 two-process medians, "
                     f"which selects downward. The confirmed column is the one to "
                     f"quote.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "jobs", "confirm.json"))
    a = ap.parse_args()
    if a.emit:
        return emit(a.out)
    if a.table:
        print(table())
    else:
        for d, v in ranked().items():
            print(f"\n{d}")
            for ms, s, _ in v[:5]:
                print(f"  {ms:8.3f}  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
