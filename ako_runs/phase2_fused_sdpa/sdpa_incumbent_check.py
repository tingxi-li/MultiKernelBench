#!/usr/bin/env python3
"""Re-measure the incumbent SDPA solutions against a precision-matched reference.

The published cell reports `tilelang 2.78x`. That denominator is
`F.scaled_dot_product_attention` on **fp32** inputs. The incumbent tilelang
kernel casts to **fp16** (`kernel(Q.half(), K.half(), V.half())`). So the ratio
mixes a code-generation difference with an arithmetic difference, which is the
same confound Phase 1 found in the GEMM cell.

`sdpa_reference_audit.py` established the other half: at D=1024 the reference is
NOT falling back to the naive `math` backend -- it runs mem-efficient attention,
a real tiled kernel. So "the SDPA gain is just a PyTorch fallback" is false as
stated. The question is what is left once precision is matched.

This times, in ONE process under the Phase-1 protocol so the numbers are
directly comparable:

  * torch SDPA, fp32 in  -- the published denominator
  * torch SDPA, fp16 in  -- the precision-matched denominator
  * the incumbent solution for each DSL, as shipped

and reports both ratios side by side.

usage: python sdpa_incumbent_check.py [--gpu 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
common2.setup_cuda_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

AKO = os.path.join(os.path.dirname(HERE), "scaled_dot_product_attention")
DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")


def load_solution(dsl):
    path = os.path.join(AKO, dsl, "solution", "scaled_dot_product_attention.py")
    if not os.path.exists(path):
        return None, f"no solution at {path}"
    spec = importlib.util.spec_from_file_location(f"sol_{dsl}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc(limit=3)
    cls = getattr(mod, "ModelNew", None) or getattr(mod, "Model", None)
    if cls is None:
        return None, "no ModelNew/Model in solution"
    try:
        return cls().cuda().eval(), None
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc(limit=3)


def time3(fn, q, k, v, warmup_s=2.0, trials=50):
    import time as _t
    t0, n = _t.perf_counter(), 0
    while True:
        fn(q, k, v)
        n += 1
        if n % 4 == 0:
            torch.cuda.synchronize()
            if _t.perf_counter() - t0 >= warmup_s:
                break
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(q, k, v); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--d", type=int, default=common2.S_D_BENCH)
    ap.add_argument("--out", default=os.path.join(common2.RESULTS_DIR,
                                                  "sdpa_incumbent_check.json"))
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    d = a.d

    q32, k32, v32 = common2.sdpa_inputs(d, dtype=torch.float32)
    q16, k16, v16 = q32.half(), k32.half(), v32.half()

    sdpa = lambda x, y, z: F.scaled_dot_product_attention(x, y, z)
    ref32_out = sdpa(q32, k32, v32)
    ms_ref32 = time3(sdpa, q32, k32, v32)
    ms_ref16 = time3(sdpa, q16, k16, v16)
    print(f"torch SDPA fp32 in : {ms_ref32:8.2f} ms   (the published denominator)")
    print(f"torch SDPA fp16 in : {ms_ref16:8.2f} ms   (precision-matched denominator)")

    del q16, k16, v16
    torch.cuda.empty_cache()

    # fp64 truth on a SUBSET. At D=1024 one fp64 copy of the output is 4.3 GB and
    # Q/K/V are 2.1 GB each, so a full-tensor fp64 truth does not fit alongside a
    # loaded kernel. Two (batch, head) slices are enough to characterize how far
    # the fp32 oracle itself sits from truth, which is all this number is for --
    # the gate is against the fp32 oracle either way.
    NB = 2
    try:
        tr = common2.sdpa_reference(q32[:NB], k32[:NB], v32[:NB], torch.float64)
        oracle_err = (tr - ref32_out[:NB].double()).abs().max().item()
        del tr
        torch.cuda.empty_cache()
        print(f"    torch fp32 oracle vs fp64 truth (first {NB} batches): "
              f"max abs err {oracle_err:.3e}")
    except torch.OutOfMemoryError:
        oracle_err = None
        torch.cuda.empty_cache()
        print("    (fp64 truth skipped: out of memory)")

    recs = [
        {"who": "torch_sdpa_fp32", "ms": ms_ref32, "note": "published denominator"},
        {"who": "torch_sdpa_fp16", "ms": ms_ref16, "note": "precision-matched denominator"},
    ]
    for dsl in DSLS:
        model, err = load_solution(dsl)
        if model is None:
            print(f"{dsl:<16s} SKIP: {str(err).splitlines()[-1][:90]}")
            recs.append({"who": dsl, "error": str(err)[-400:]})
            continue
        try:
            with torch.no_grad():
                out = model(q32, k32, v32)
                torch.cuda.synchronize()
                gs = common2.gate_stats(ref32_out, out.float())
                ms = time3(lambda x, y, z: model(x, y, z), q32, k32, v32)
        except Exception:  # noqa: BLE001
            print(f"{dsl:<16s} FAILED: {traceback.format_exc(limit=2).splitlines()[-1][:90]}")
            recs.append({"who": dsl, "error": traceback.format_exc(limit=3)[-600:]})
            torch.cuda.empty_cache()
            continue
        r = {"who": dsl, "ms": ms,
             "vs_fp32_ref": ms_ref32 / ms, "vs_fp16_ref": ms_ref16 / ms,
             "max_abs_err": gs["max_abs_err"], "gate_pass": gs["gate_pass"],
             "pct_elems_failing_gate": gs["pct_elems_failing_gate"],
             "max_err_vs_fp64": gs.get("kernel_max_abs_err_vs_fp64")}
        r["oracle_max_err_vs_fp64_subset"] = oracle_err
        recs.append(r)
        print(f"{dsl:<16s} {ms:8.2f} ms   vs fp32 ref {r['vs_fp32_ref']:5.2f}x   "
              f"vs fp16 ref {r['vs_fp16_ref']:5.2f}x   "
              f"gate={'PASS' if gs['gate_pass'] else 'FAIL'} "
              f"maxerr={gs['max_abs_err']:.2e}")
        del model, out
        torch.cuda.empty_cache()

    common2.write_json(a.out, {"head_dim": d, "records": recs,
                               "shape": {"B": common2.S_B, "H": common2.S_H,
                                         "S": common2.S_S}})
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
