#!/usr/bin/env python3
"""run() vs kernels-only gap, same timing discipline as runner2.time_kernel3."""
import argparse, os, sys, statistics, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2
common2.setup_cuda_env()
import torch, variants2
from variants2 import sdpa_tilelang as X

ap = argparse.ArgumentParser()
ap.add_argument("--algo", required=True)
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--s", default="fp32")
ap.add_argument("--p", default="fp16")
ap.add_argument("--trials", type=int, default=15)
ap.add_argument("--warmup-s", dest="ws", type=float, default=0.5)
a = ap.parse_args()

# capture the exact compiled kernel objects build() uses
caught = []
_orig = X._compile
def cap(src, tag, out_idx):
    kf, s = _orig(src, tag, out_idx)
    caught.append((tag, kf))
    return kf, s
X._compile = cap

cfg = common2.make_sdpa_config("tilelang", a.algo,
                               extra={"d": a.d, "sdtype": a.s, "pdtype": a.p})
q, k, v = common2.sdpa_inputs(a.d, seed=0, dist="rand")
built = X.build(cfg)
print("NOTES:", built.notes)
print("CAPTURED KERNELS:", [t for t, _ in caught])
B, H, S = common2.S_B, common2.S_H, common2.S_S
BH, d = B * H, a.d
qv, kv, vv = q.view(BH, S, d), k.view(BH, S, d), v.view(BH, S, d)

ks = dict(caught)
if a.algo == "FLASH":
    kf = caught[0][1]
    def kernels_only():
        return kf(qv, kv, vv)
elif a.algo == "K2":
    k1, k2 = caught[0][1], caught[1][1]
    def kernels_only():
        p = k1(qv, kv)
        return k2(p, vv)
else:
    k1, k2, k3 = caught[0][1], caught[1][1], caught[2][1]
    def kernels_only():
        sc = k1(qv, kv)
        p = k2(sc)
        del sc
        return k3(p, vv)

def med(fn, *args):
    t0, n = time.perf_counter(), 0
    while True:
        fn(*args); n += 1
        if n % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - t0 >= a.ws: break
    torch.cuda.synchronize()
    fl = torch.empty(int(128e6 // 4), dtype=torch.float32, device="cuda")
    ts = []
    for _ in range(a.trials):
        fl.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); r = fn(*args); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1)); del r
    del fl
    torch.cuda.empty_cache()
    return statistics.median(ts), min(ts)

if os.environ.get("REV") == "1":
    mk, mink = med(kernels_only)
    mr, minr = med(built.run, q, k, v)
else:
    mr, minr = med(built.run, q, k, v)
    mk, mink = med(kernels_only)
gap = mr - mk
flops = common2.sdpa_flops(a.d)
print(f"RESULT algo={a.algo} d={a.d} s={a.s} p={a.p} "
      f"run_median_ms={mr:.3f} run_min={minr:.3f} kernels_median_ms={mk:.3f} "
      f"kernels_min={mink:.3f} gap_ms={gap:.3f} gap_pct={100*gap/mr:.1f} "
      f"TFLOPs_run={flops/(mr*1e-3)/1e12:.1f} TFLOPs_kern={flops/(mk*1e-3)/1e12:.1f}")
