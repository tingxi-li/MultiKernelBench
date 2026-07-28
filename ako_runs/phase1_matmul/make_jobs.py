#!/usr/bin/env python3
"""Emit the job lists for each Phase-1 sub-study.

Each sub-study is a JSON list of {dsl, variant, geom, set} that driver.py runs
in randomized order across independent processes.

usage:  python make_jobs.py            # writes jobs/*.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")
DSLS = list(common.DSLS)


def w(name, jobs):
    os.makedirs(JOBS, exist_ok=True)
    p = os.path.join(JOBS, name + ".json")
    with open(p, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"{name:<22s} {len(jobs):>4d} jobs -> {p}")
    return p


# ---------------------------------------------------------------- matched ---
# A/B/C/D x 4 DSLs at both geometries, plus the torch denominator.
matched = []
for geom in ("primary", "secondary"):
    for d in DSLS:
        for v in ("A", "B", "C", "D"):
            matched.append({"dsl": d, "variant": v, "geom": geom, "set": ""})
matched += [{"dsl": "torch", "variant": "A", "geom": "primary", "set": ""},
            {"dsl": "torch", "variant": "B", "geom": "primary", "set": ""}]
w("matched", matched)

# --------------------------------------------------------------- kc sweep ---
# Variant C's structure with the chunk size swept. kc=8192 IS variant B
# (one chain over all K), so the sweep self-anchors at its own no-flush end.
kc_sweep = []
for d in DSLS:
    for kc in (512, 1024, 2048, 4096, 8192):
        kc_sweep.append({"dsl": d, "variant": "C", "geom": "primary",
                         "set": f"kc={kc}"})
    # and the same sweep with the pipeline on, so the KC conclusion is not
    # only valid in the unpipelined regime
    for kc in (512, 1024, 2048, 4096, 8192):
        kc_sweep.append({"dsl": d, "variant": "D", "geom": "primary",
                         "set": f"kc={kc}"})
w("kc_sweep", kc_sweep)

# ---------------------------------------------------------------- casting ---
# Same fp16 kernel, three input paths. Isolates "fp16 compute" from
# "cost of getting to fp16".
casting = []
for d in DSLS:
    for cast in common.CAST_MODES:
        casting.append({"dsl": d, "variant": "D", "geom": "primary",
                        "set": f"cast={cast}"})
# The incumbent tilelang winner's exact shape (128x256x64, stages=2, KC=2048),
# both cast ways. Its published form casts .half() INSIDE forward(), i.e.
# cast=in_region -- so this pair measures how much of the incumbent's headline
# number was paid for conversion. Carried at geom=incumbent because 128x256x64
# cannot host a 3-stage pipeline on sm_89 and so is not a matched point.
for cast in ("precast", "in_region"):
    casting.append({"dsl": "tilelang", "variant": "D", "geom": "incumbent",
                    "set": f"cast={cast},kc=2048,stages=2"})
casting += [{"dsl": "torch", "variant": "B", "geom": "primary", "set": f"cast={c}"}
            for c in ("precast", "in_region")]
w("casting", casting)

# --------------------------------------------------------------- pipeline ---
# Depth 1/2/3/4 at otherwise identical config. For the CUDA lanes stages==1 is
# synchronous single-buffer and stages>1 is async multi-buffer.
pipeline = []
for d in DSLS:
    for s in (1, 2, 3, 4):
        pipeline.append({"dsl": d, "variant": "D", "geom": "primary",
                         "set": f"stages={s}"})
w("pipeline", pipeline)

# ------------------------------------------------------------ native tuned ---
# Equal-budget native tuning: the SAME configuration grid offered to every DSL.
# 12 points each; whoever exploits the shared grid best wins the second table.
GRID = [
    (128, 128, 32), (128, 128, 64), (128, 256, 32), (128, 256, 64),
    (256, 128, 32), (256, 128, 64), (64, 128, 64), (128, 64, 64),
]
STAGES = (2, 3, 4)
native = []
for d in DSLS:
    for (bm, bn, bk) in GRID:
        for s in STAGES:
            smem_bytes = (bm * bk + bk * bn) * 2 * s
            if smem_bytes > 96 * 1024:      # Ada opt-in smem ceiling
                continue
            native.append({"dsl": d, "variant": "D", "geom": "primary",
                           "set": f"BM={bm},BN={bn},BK={bk},stages={s},kc=2048"})
w("native_tuned", native)

# ------------------------------------------------------- abstraction arms ---
# The TileLang-only study. Kept in its own campaign and its own tables: mixing it
# with the cross-DSL study would confound abstraction level with algorithm and
# with hardware instruction path. Each arm's depth and arithmetic come from
# common.ABSTRACTION_SPECS, so no `set` string is needed here.
w("abstraction", [{"dsl": "tilelang_abs", "variant": v, "geom": "primary", "set": ""}
                  for v in ("H1", "H2", "M1", "M2", "S1")])

# ------------------------------------------------------ abstraction depth ---
# The abstraction spec gives H1 3 stages and M2 2, so the headline H1-vs-M2
# comparison confounds "compiler-generated vs hand-written pipeline" with depth.
# Only H1 can move: `num_stages` is an integer to the compiler-managed pipeline,
# while M2's buffer parity is hand-unrolled at depth 2 and a third stage is a
# rewrite, not a config change. `x_depth_control=1` is the explicit opt-in past
# the arm's own drift guard -- the guard stays strict by default because
# accidental depth drift would void the study.
w("abstraction_depth", [
    {"dsl": "tilelang_abs", "variant": "H1", "geom": "primary",
     "set": "stages=2,x_depth_control=1"},
])

print("\nnote: native_tuned is intentionally the SAME grid for every DSL "
      "(equal budget); each DSL's best point becomes its row in table 2.")
print("note: jobs/confirm.json is NOT generated here -- it is derived from the "
      "native_tuned results by `confirm_winners.py --emit`.")
