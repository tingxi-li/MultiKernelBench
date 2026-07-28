#!/usr/bin/env python3
"""Which backend does `F.scaled_dot_product_attention` actually run?

The benchmark's SDPA cell uses B=32, H=32, S=512, **D=1024**. FlashAttention and
the mem-efficient kernel both cap head_dim well below that, so the reference is
very likely falling back to the `math` backend -- which materializes the full
S = QK^T matrix (32*32*512*512*4 B = 8 GB of traffic) and is not a serious
baseline. Every speedup reported for this cell is divided by that number.

This settles it by measurement rather than inference, three ways:

1. `torch.backends.cuda.can_use_*` predicates on the real shapes.
2. Forcing each backend with `sdpa_kernel(...)` and recording which ones run at
   all versus raise.
3. Timing the default path against each forced path, so a fallback shows up as
   the default matching `math` and nothing else.

Run at D in {128, 256, 1024}: the smaller dimensions are where a real flash
backend is available, so they show what the reference *could* have been.

usage: python sdpa_reference_audit.py [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = os.path.join(os.path.dirname(HERE), "phase1_matmul")
sys.path.insert(0, PHASE1)
import common  # noqa: E402  -- reuse the frozen timing protocol
common.setup_cuda_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

B, H, S = 32, 32, 512
HEAD_DIMS = (128, 256, 1024)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    BACKENDS = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None),
    }
    BACKENDS = {k: v for k, v in BACKENDS.items() if v is not None}
except ImportError:  # older torch
    BACKENDS = {}
    SDPBackend = sdpa_kernel = None


def make(d, dtype, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.rand(B, H, S, d, generator=g).to(device=device, dtype=dtype)
    return mk(), mk(), mk()


def timed(fn, q, k, v, warmup_s=2.0, trials=30):
    """Same discipline as Phase 1: fixed WARMUP TIME (the card thermally soaks,
    so a fixed iteration count biases toward whichever variant is already fast),
    cuda-event timed, median of trials."""
    t0 = time.perf_counter()
    n = 0
    while True:
        fn(q, k, v)
        n += 1
        if n % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - t0 >= warmup_s:
                break
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn(q, k, v)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def audit_one(d, dtype):
    rec = {"head_dim": d, "dtype": str(dtype).replace("torch.", ""),
           "B": B, "H": H, "S": S}
    q, k, v = make(d, dtype)

    # 1) what torch says it *can* use
    try:
        from torch.backends.cuda import SDPAParams, can_use_flash_attention, \
            can_use_efficient_attention
        p = SDPAParams(q, k, v, None, 0.0, False, False)
        rec["can_use_flash"] = bool(can_use_flash_attention(p, False))
        rec["can_use_mem_efficient"] = bool(can_use_efficient_attention(p, False))
    except Exception as e:  # noqa: BLE001
        rec["can_use_probe_error"] = f"{type(e).__name__}: {e}"

    # 2) + 3) which backends run, and how fast
    rec["backends"] = {}
    for name, be in BACKENDS.items():
        try:
            with sdpa_kernel(be):
                out = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                ms = timed(lambda a, b, c: F.scaled_dot_product_attention(a, b, c),
                           q, k, v)
            rec["backends"][name] = {"ok": True, "ms": ms}
            del out
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            rec["backends"][name] = {"ok": False, "error": msg}
        torch.cuda.empty_cache()

    # the default path -- what the benchmark's reference actually measures
    rec["default_ms"] = timed(lambda a, b, c: F.scaled_dot_product_attention(a, b, c),
                              q, k, v)

    # attribute the default to whichever forced backend it matches
    ok = {n: b["ms"] for n, b in rec["backends"].items() if b.get("ok")}
    if ok:
        best = min(ok, key=lambda n: abs(ok[n] - rec["default_ms"]))
        rec["default_matches"] = best
        rec["default_vs_fastest_available"] = rec["default_ms"] / min(ok.values())
        rec["fastest_available"] = min(ok, key=lambda n: ok[n])
    del q, k, v
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sdpa_reference_audit.json"))
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)

    recs = []
    for dtype in (torch.float32, torch.float16):
        for d in HEAD_DIMS:
            r = audit_one(d, dtype)
            recs.append(r)
            bl = " ".join(f"{n}={'%.1f' % b['ms'] if b.get('ok') else 'X'}"
                          for n, b in r["backends"].items())
            print(f"D={d:<5d} {r['dtype']:<8s} default={r['default_ms']:8.2f} ms "
                  f"-> {r.get('default_matches','?'):<14s} | {bl}")
    common.write_json(a.out, {"records": recs, "shape": {"B": B, "H": H, "S": S},
                              "torch": torch.__version__})
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
