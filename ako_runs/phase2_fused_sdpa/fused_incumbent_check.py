#!/usr/bin/env python3
"""Re-measure the shipped fused solutions against precision-matched denominators.

The published cell reports, against a 6.88 ms fp32 reference:

    tilelang 1.17 ms / 5.29x     triton 2.91 ms / 2.36x
    cuda_unlimited 5.56 / 1.24x  cuda_noptx 6.57 / 1.05x

Two things need checking before any of that can be read as a code-generation
result, and they are the same two Phase 1 found in the GEMM cell:

1. The denominator is fp32 and the winners are fp16/tf32, so the ratio mixes
   arithmetic with codegen. The fp16 denominator is measured here alongside.
2. `triton`'s own convergence log names the problem out loud -- its winning
   iteration is described as "tol forgiving via softmax". Softmax outputs are
   ~1/8192 = 1.22e-4 and the gate is 1e-4 + 1e-4*|ref|, so the tolerance is
   roughly the size of the values being checked. `gate_sensitivity()` measures
   how much a result can be wrong and still pass.

Everything is timed in ONE process under the Phase-1 protocol, with the L2 flush
as an explicit two-way control, because the flush is the one protocol difference
between this harness and AKO's bench.py.

usage: python fused_incumbent_check.py [--gpu 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
common2.setup_cuda_env()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

AKO = os.path.join(os.path.dirname(HERE), "matmul_gelu_softmax")
DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
IN_F, OUT_F, BATCH = common2.F_K, common2.F_N, common2.F_M


def load_solution(dsl):
    path = os.path.join(AKO, dsl, "solution", "matmul_gelu_softmax.py")
    if not os.path.exists(path):
        return None, f"no solution at {path}"
    spec = importlib.util.spec_from_file_location(f"fsol_{dsl}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc(limit=3)
    cls = getattr(mod, "ModelNew", None) or getattr(mod, "Model", None)
    if cls is None:
        return None, "no ModelNew/Model in solution"
    try:
        return cls(IN_F, OUT_F).cuda().eval(), None
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc(limit=3)


def timed(fn, x, warmup_s=2.0, trials=50, flush_l2=True):
    t0, n = time.perf_counter(), 0
    while True:
        fn(x)
        n += 1
        if n % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - t0 >= warmup_s:
                break
    torch.cuda.synchronize()
    flusher = torch.empty(int(128e6 // 4), dtype=torch.float32, device="cuda") \
        if flush_l2 else None
    ts = []
    for _ in range(trials):
        if flusher is not None:
            flusher.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(x); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    del flusher
    ts.sort()
    return ts[len(ts) // 2]


def gate_sensitivity(ref):
    """How wrong may a result be and still pass this cell's correctness gate?

    Reports, for progressively cruder fakes, the fraction of elements inside
    |ref-got| <= 1e-4 + 1e-4*|ref|. A gate that a constant tensor nearly passes
    is not measuring the kernel.
    """
    tol = common2.GATE_ATOL + common2.GATE_RTOL * ref.abs()
    out = {}
    N = ref.shape[1]
    fakes = {
        "uniform_1_over_N": torch.full_like(ref, 1.0 / N),
        "row_mean":         ref.mean(dim=1, keepdim=True).expand_as(ref),
        "zeros":            torch.zeros_like(ref),
        "shuffled_rows":    ref.flip(0),
    }
    for name, g in fakes.items():
        ok = ((g - ref).abs() <= tol)
        out[name] = {"pct_pass": 100.0 * ok.float().mean().item(),
                     "max_abs_err": (g - ref).abs().max().item(),
                     "gate_pass": bool(ok.all())}
        del g
    out["_ref_scale"] = {"mean": ref.mean().item(), "max": ref.max().item(),
                         "min": ref.min().item(),
                         "median_tol": tol.median().item()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(common2.RESULTS_DIR,
                                                  "fused_incumbent_check.json"))
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)

    x, W, b = common2.fused_inputs(seed=0)
    ref = common2.fused_reference(x, W, b, arm="GBGS", dtype=torch.float32)

    print("== gate sensitivity: what else passes this cell's 1e-4 gate ==")
    gs = gate_sensitivity(ref)
    sc = gs.pop("_ref_scale")
    print(f"   softmax output scale: mean {sc['mean']:.3e}  max {sc['max']:.3e}  "
          f"median tolerance {sc['median_tol']:.3e}")
    for name, r in gs.items():
        print(f"   {name:<20s} {r['pct_pass']:6.2f}% of elements inside tol   "
              f"gate={'PASS' if r['gate_pass'] else 'fail'}")
    gs["_ref_scale"] = sc

    # denominators: the reference module itself, fp32 and fp16
    lin = nn.Linear(IN_F, OUT_F).cuda().eval()
    with torch.no_grad():
        lin.weight.copy_(W)
        lin.bias.copy_(b)

    def ref_fp32(t):
        return F.softmax(F.gelu(lin(t)), dim=1)

    lin16 = nn.Linear(IN_F, OUT_F).cuda().eval().half()
    with torch.no_grad():
        lin16.weight.copy_(W.half())
        lin16.bias.copy_(b.half())

    def ref_fp16(t):
        return F.softmax(F.gelu(lin16(t.half())), dim=1).float()

    print("\n== denominators ==")
    recs = []
    for who, fn in (("torch_fp32", ref_fp32), ("torch_fp16", ref_fp16)):
        for flush in (True, False):
            ms = timed(fn, x, flush_l2=flush)
            recs.append({"who": who, "flush_l2": flush, "ms": ms})
            print(f"   {who:<12s} flush_l2={str(flush):<5s} {ms:8.3f} ms")
    base32 = min(r["ms"] for r in recs if r["who"] == "torch_fp32" and not r["flush_l2"])
    base16 = min(r["ms"] for r in recs if r["who"] == "torch_fp16" and not r["flush_l2"])

    print("\n== shipped solutions ==")
    for dsl in DSLS:
        model, err = load_solution(dsl)
        if model is None:
            print(f"   {dsl:<16s} SKIP: {str(err).splitlines()[-1][:80]}")
            recs.append({"who": dsl, "error": str(err)[-400:]})
            continue
        try:
            with torch.no_grad():
                for p, src in (("weight", W), ("bias", b)):
                    for mod in model.modules():
                        if isinstance(mod, nn.Linear):
                            getattr(mod, p).copy_(src.to(getattr(mod, p).dtype))
                got = model(x)
                torch.cuda.synchronize()
                st = common2.gate_stats(ref, got.float())
                r = {"who": dsl, "gate_pass": st["gate_pass"],
                     "max_abs_err": st["max_abs_err"],
                     "pct_elems_failing_gate": st["pct_elems_failing_gate"]}
                del got
                torch.cuda.empty_cache()
            for flush in (True, False):
                r[f"ms_flush{int(flush)}"] = timed(lambda t: model(t), x,
                                                   flush_l2=flush)
        except Exception:  # noqa: BLE001
            print(f"   {dsl:<16s} FAILED: {traceback.format_exc(limit=2).splitlines()[-1][:80]}")
            recs.append({"who": dsl, "error": traceback.format_exc(limit=3)[-600:]})
            torch.cuda.empty_cache()
            continue
        r["vs_fp32"] = base32 / r["ms_flush0"]
        r["vs_fp16"] = base16 / r["ms_flush0"]
        recs.append(r)
        print(f"   {dsl:<16s} noflush {r['ms_flush0']:7.3f} ms  flush {r['ms_flush1']:7.3f} ms"
              f"   vs fp32 {r['vs_fp32']:5.2f}x  vs fp16 {r['vs_fp16']:5.2f}x"
              f"   gate={'PASS' if r['gate_pass'] else 'FAIL'} err={r['max_abs_err']:.2e}")
        del model
        torch.cuda.empty_cache()

    common2.write_json(a.out, {"records": recs, "gate_sensitivity": gs,
                               "shape": {"M": BATCH, "K": IN_F, "N": OUT_F}})
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
