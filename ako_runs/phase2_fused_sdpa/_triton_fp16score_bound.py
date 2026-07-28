#!/usr/bin/env python3
"""Is the (fp16, fp16) failure at d=1024 a property of the KERNEL or of the CELL?

Computes an implementation-free lower bound on the error any correct kernel must
incur when the score matrix is rounded to fp16: take the fp32 reference, round
ONLY the scaled scores to fp16, and do everything else (softmax, PV) in fp64.
No tiling, no tensor cores, no accumulation error -- only the sdtype rounding.
"""
import os, sys, math, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common2  # noqa

for d in (128, 256, 1024):
    q, k, v = common2.sdpa_inputs(d)
    scale = 1.0 / math.sqrt(d)
    worst = 0.0; nfail = 0; ntot = 0; budget_at_worst = 0.0
    for i in range(0, q.shape[0], 2):
        qq, kk, vv = q[i:i+2].double(), k[i:i+2].double(), v[i:i+2].double()
        s = (qq @ kk.transpose(-2, -1)) * scale
        ref = torch.softmax(s, -1) @ vv                     # fp64 truth
        s16 = s.half().double()                             # ONLY sdtype rounding
        got = torch.softmax(s16, -1) @ vv
        e = (ref - got).abs()
        bud = 1e-4 + 1e-4 * ref.abs()
        nfail += int((e > bud).sum()); ntot += e.numel()
        m = float(e.max())
        if m > worst:
            worst = m
        del qq, kk, vv, s, ref, s16, got, e, bud
    print(f"d={d:5d}  fp16-score-only max_abs={worst:.3e}  "
          f"elems over budget: {nfail}/{ntot} ({100.0*nfail/ntot:.3g}%)", flush=True)
    del q, k, v
    torch.cuda.empty_cache()
