#!/usr/bin/env python3
"""Measure cold and warm compile time per lane, with the caches under control.

The campaign records `compile_s` for every build, but those numbers are not
comparable across lanes, because the lanes do not share a caching policy: only
TileLang calls `tilelang.disable_cache()`, so only TileLang recompiles on every
build. Triton keeps its own source-hash cache and the two CUDA lanes are served
by ninja out of the persistent `TORCH_EXTENSIONS_DIR` that Phase 1 pins. A
median over campaign builds therefore reads one cold lane against three warm
ones -- and it reads backwards: it makes TileLang look ~35x slower to compile
when, cold against cold, it is several times faster than nvcc.

This measures the two states separately and deliberately:

  cold -- a fresh, empty cache directory per repetition, so nothing can be
          reused. This is the cost of compiling a kernel the search has not
          seen before, which is the cost that matters to a search loop.
  warm -- the same build repeated against the cache the cold run just
          populated. This is the cost of re-encountering a known kernel.

TileLang has no warm mode here: its cache is disabled inside the shipped build
path, so `warm` would require editing the artifact under measurement. That is
reported as not-exercised rather than estimated.

Compile time is host-side work; it does not depend on which card is visible, so
this can run off the reserved GPU while GPU 0 is busy timing.

usage: python compile_cold.py --gpu 1 --reps 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
import common  # noqa: E402

# Arm G -- GEMM only -- is the one arm every lane implements identically, and
# it is the anchor to Phase 1 (SPEC2.md). Using the same arm everywhere keeps
# this a measurement of the toolchain rather than of how much code each arm is.
ARM = "G"

# Lanes whose cache the shipped build path disables. For these, `warm` is not a
# state the artifact can be in.
NO_WARM = ("tilelang",)

CHILD = r"""
import json, os, sys, time
sys.path.insert(0, %(here)r)
import common2
common2.setup_cuda_env()
import variants2
cfg = common2.make_fused_config(%(dsl)r, %(arm)r)
t0 = time.perf_counter()
built = variants2.build("fused", cfg)
wall = time.perf_counter() - t0
print("###JSON###")
print(json.dumps({"compile_s": built.compile_s, "wall_s": wall,
                  "notes": built.notes}))
"""


def one_build(dsl, cache_dir, gpu):
    """Build arm G once, in a fresh process, with every cache pointed at
    `cache_dir`. Returns the lane's own `compile_s` and the wall time around
    the whole build call (which includes codegen, not just the toolchain)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")
    # Override, not setdefault: `common.setup_cuda_env` uses setdefault, so a
    # value already in the environment wins -- which is exactly how the cache
    # gets redirected here without touching the shared one.
    env["TORCH_EXTENSIONS_DIR"] = os.path.join(cache_dir, "torch_ext")
    env["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
    env["TILELANG_CACHE_DIR"] = os.path.join(cache_dir, "tilelang")
    env["XDG_CACHE_HOME"] = os.path.join(cache_dir, "xdg")
    src = CHILD % {"here": HERE, "dsl": dsl, "arm": ARM}
    p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, env=env, cwd=HERE, timeout=1800)
    out = p.stdout or ""
    if "###JSON###" not in out:
        return {"error": (p.stderr or out)[-400:]}
    return json.loads(out.split("###JSON###", 1)[1].strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(common2.RESULTS_DIR,
                                                  "compile_cold.json"))
    a = ap.parse_args()

    doc = {"arm": ARM, "reps": a.reps, "gpu": a.gpu, "lanes": {}}
    jobtmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", ""), "tmp")
    root = tempfile.mkdtemp(prefix="compile_cold_",
                            dir=jobtmp if os.path.isdir(jobtmp) else None)
    try:
        for dsl in common2.DSLS:
            rec = {"cold": [], "warm": [], "cold_wall": [], "warm_wall": []}
            for r in range(a.reps):
                cdir = os.path.join(root, f"{dsl}_{r}")
                os.makedirs(cdir, exist_ok=True)
                cold = one_build(dsl, cdir, a.gpu)
                if "error" in cold:
                    rec.setdefault("errors", []).append(cold["error"])
                    print(f"  {dsl} rep{r} cold FAILED: {cold['error'][-200:]}",
                          flush=True)
                    continue
                rec["cold"].append(cold["compile_s"])
                rec["cold_wall"].append(cold["wall_s"])
                msg = f"  {dsl} rep{r}: cold {cold['compile_s']:.2f}s"
                if dsl not in NO_WARM:
                    # Same cache dir, fresh process: whatever the lane cached on
                    # the cold pass is now available to it.
                    warm = one_build(dsl, cdir, a.gpu)
                    if "error" not in warm:
                        rec["warm"].append(warm["compile_s"])
                        rec["warm_wall"].append(warm["wall_s"])
                        msg += f" / warm {warm['compile_s']:.2f}s"
                else:
                    rec["warm_note"] = ("cache disabled in the shipped build "
                                        "path; warm state not reachable")
                print(msg, flush=True)
                shutil.rmtree(cdir, ignore_errors=True)
            for f in ("cold", "warm", "cold_wall", "warm_wall"):
                if rec[f]:
                    rec[f + "_stats"] = common.median_ci(rec[f])
            doc["lanes"][dsl] = rec
    finally:
        shutil.rmtree(root, ignore_errors=True)
    common2.write_json(a.out, doc)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
