#!/usr/bin/env python3
"""Cold compile cost per DSL, measured with every cache isolated and empty.

The `compile s` column in the matched table is **warm-cache** time and is not
comparable across lanes: the CUDA lanes reuse a `.so` from `TORCH_EXTENSIONS_DIR`
(0.2 s), Triton reuses its JIT cache (0.4 s), and TileLang re-runs part of its
pipeline every process (4-10 s). Reading those three numbers as "TileLang
compiles 25x slower" would be comparing a cache hit against a partial miss.

This measures the number that is actually comparable: time from an empty cache to
a launchable kernel. Each build gets a fresh temporary `TORCH_EXTENSIONS_DIR`,
`TRITON_CACHE_DIR` and `TILELANG_CACHE_DIR`, in its own process so no in-process
JIT state survives.

The study's measurement discipline requires compile time to be reported
separately from execution time and never folded into it -- this is that number.

usage:
  python compile_cost.py --gpu 0                # all four lanes, variant D
  python compile_cost.py --gpu 0 --variants A,D
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
import common  # noqa: E402

CHILD = r'''
import json, os, sys, time
sys.path.insert(0, %(here)r)
import common
common.setup_cuda_env()
os.environ["TORCH_EXTENSIONS_DIR"] = %(ext)r
import importlib, torch
cfg = common.make_config(%(dsl)r, %(variant)r, %(geom)r)
mod = importlib.import_module("variants.%(dsl)s_gemm")
t0 = time.perf_counter()
built = mod.build(cfg)
wall = time.perf_counter() - t0
print("###JSON###" + json.dumps({
    "dsl": %(dsl)r, "variant": %(variant)r,
    "cold_compile_s": wall,
    "self_reported_s": getattr(built, "compile_s", None),
}))
'''


def one(dsl, variant, geom, gpu):
    tmp = tempfile.mkdtemp(prefix=f"coldc_{dsl}_{variant}_")
    ext = os.path.join(tmp, "torch_ext")
    os.makedirs(ext, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TORCH_EXTENSIONS_DIR": ext,
        "TRITON_CACHE_DIR": os.path.join(tmp, "triton"),
        "TILELANG_CACHE_DIR": os.path.join(tmp, "tilelang"),
        "TVM_CACHE_DIR": os.path.join(tmp, "tvm"),
        "PYTHONPATH": HERE + ":" + env.get("PYTHONPATH", ""),
    })
    src = CHILD % {"here": HERE, "ext": ext, "dsl": dsl,
                   "variant": variant, "geom": geom}
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                           text=True, env=env, cwd=HERE, timeout=1800)
        line = next((l for l in p.stdout.splitlines() if l.startswith("###JSON###")), None)
        if not line:
            return {"dsl": dsl, "variant": variant, "error": (p.stderr or p.stdout)[-600:]}
        return json.loads(line[len("###JSON###"):])
    except subprocess.TimeoutExpired:
        return {"dsl": dsl, "variant": variant, "error": "TIMEOUT"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--geom", default="primary")
    ap.add_argument("--variants", default="A,D")
    ap.add_argument("--dsls", default=",".join(common.DSLS))
    ap.add_argument("--out", default=os.path.join(common.RESULTS_DIR, "compile_cost.json"))
    a = ap.parse_args()

    recs = []
    for d in [s for s in a.dsls.split(",") if s and s != "torch"]:
        for v in [s for s in a.variants.split(",") if s]:
            r = one(d, v, a.geom, a.gpu)
            recs.append(r)
            if "error" in r:
                print(f"{d}/{v:<3s} FAILED: {r['error'].splitlines()[-1][:100]}")
            else:
                print(f"{d}/{v:<3s} cold compile {r['cold_compile_s']:7.2f} s"
                      f"   (self-reported {r.get('self_reported_s') or float('nan'):.2f} s)")
    common.write_json(a.out, {"records": recs, "geom": a.geom})
    print(f"-> {a.out}")

    ok = [r for r in recs if "error" not in r]
    if ok:
        print("\n| DSL | variant | cold compile s |")
        print("|---|---|---|")
        for r in ok:
            print(f"| {r['dsl']} | {r['variant']} | {r['cold_compile_s']:.1f} |")


if __name__ == "__main__":
    main()
