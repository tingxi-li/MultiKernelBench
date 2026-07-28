#!/usr/bin/env python3
"""Scratch tile sweep for the triton SDPA lane.

NOT part of the shipped module and NOT an autotuner: this is how the constants
in FLASH_TILES / K3_TILES were chosen once, by hand, offline.  The shipped
module reads a fixed dict.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa
common2.setup_cuda_env()
import torch, common  # noqa
from variants2 import sdpa_triton as M  # noqa


def bench(algo, d, sd, pd, tile, q, k, v, ref, trials=8, warm=0.4, table=None):
    tbl = M.FLASH_TILES if algo == "FLASH" else M.K3_TILES
    old = tbl[d]
    if tile is not None:
        tbl[d] = tile
    try:
        ex = dict(d=d, sdtype=sd, pdtype=pd)
        if os.environ.get("F32DOT"):
            ex["f32dot"] = os.environ["F32DOT"]
        cfg = common2.make_sdpa_config("triton", algo, extra=ex)
        b = M.build(cfg)
        got = b.run(q, k, v)
        torch.cuda.synchronize()
        st = common.gate_stats(ref, got.float())
        del got
        torch.cuda.empty_cache()
        t0, n = time.perf_counter(), 0
        while True:
            b.run(q, k, v); n += 1
            if n % 4 == 0:
                torch.cuda.synchronize()
                if time.perf_counter() - t0 >= warm:
                    break
        torch.cuda.synchronize()
        ts = []
        for _ in range(trials):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); b.run(q, k, v); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        return st, ts[len(ts) // 2]
    finally:
        tbl[d] = old


if __name__ == "__main__":
    algo = sys.argv[1]
    d = int(sys.argv[2])
    pairs = [tuple(p.split(":")) for p in sys.argv[3].split(",")]
    cands = json.loads(sys.argv[4]) if len(sys.argv) > 4 else [None]
    q, k, v = common2.sdpa_inputs(d)
    with torch.no_grad():
        ref = common2.sdpa_reference(q, k, v, torch.float32)
    torch.cuda.empty_cache()
    for tile in cands:
        for sd, pd in pairs:
            try:
                st, ms = bench(algo, d, sd, pd, tile, q, k, v, ref)
                print(f"{algo} d={d} {sd}/{pd} {tile} -> gate={st['gate_pass']} "
                      f"max_abs={st['max_abs_err']:.3e} med={ms:.3f} ms", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{algo} d={d} {sd}/{pd} {tile} -> ERR {type(e).__name__}: "
                      f"{str(e)[:180]}", flush=True)
            torch.cuda.empty_cache()
