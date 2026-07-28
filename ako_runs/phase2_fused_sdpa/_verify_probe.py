#!/usr/bin/env python3
"""Adversarial probe: does run() launch ONLY the algorithm kernels (no host cast),
are the kernel params fp32, and what does the generated CUDA look like."""
import argparse, os, re, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2
common2.setup_cuda_env()
import torch, common, variants2

ap = argparse.ArgumentParser()
ap.add_argument("--algo", required=True)
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--s", default="fp32")
ap.add_argument("--p", default="fp16")
ap.add_argument("--dsl", default="tilelang")
ap.add_argument("--dump", default="")
ap.add_argument("--gate", action="store_true")
a = ap.parse_args()

cfg = common2.make_sdpa_config(a.dsl, a.algo,
                               extra={"d": a.d, "sdtype": a.s, "pdtype": a.p})
q, k, v = common2.sdpa_inputs(a.d, seed=0, dist="rand")
print("INPUT DTYPES:", q.dtype, k.dtype, v.dtype, q.shape, "contig", q.is_contiguous())
built = variants2.build("sdpa", cfg)
print("NOTES:", built.notes)
print("N_KERNELS field:", built.n_kernels)
print("ARTIFACT KEYS:", sorted(built.artifacts))
print("TILE:", built.artifacts.get("tile"))
print("BACKEND_DETAIL:", built.artifacts.get("backend_detail", "")[:400])

# --- generated CUDA signatures -------------------------------------------
srcs = {kk: vv for kk, vv in built.artifacts.items() if kk.startswith("cuda_source")}
for name, src in srcs.items():
    sig = re.findall(r"extern\s+\"C\"\s+__global__[^\{]*\{", src)
    print(f"--- {name}: {len(src)} chars")
    for s in sig:
        print("   SIG:", " ".join(s.split())[:400])
    # count mma / wmma / cvt occurrences
    print("   n_gemm_mma:", len(re.findall(r"mma_sync|wmma", src)),
          " has_cvt_f32_f16:", len(re.findall(r"__float2half|cvt.rn.f16.f32|half\)", src)))
    if a.dump:
        p = os.path.join(a.dump, f"{a.dsl}_{a.algo}_d{a.d}_{a.s}_{a.p}_{name}.cu")
        open(p, "w").write(src)
        print("   wrote", p)

# --- correctness ----------------------------------------------------------
if a.gate:
    with torch.no_grad():
        ref = common2.sdpa_reference(q, k, v, torch.float32)
        got = built.run(q, k, v)
        torch.cuda.synchronize()
        st = common.gate_stats(ref, got.float())
    print("GATE:", json.dumps({kk: st[kk] for kk in st}, default=str))
    del ref, got
    torch.cuda.empty_cache()

# --- what kernels does run() actually launch? -----------------------------
for _ in range(3):
    o = built.run(q, k, v)
torch.cuda.synchronize()
del o
torch.cuda.empty_cache()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    o = built.run(q, k, v)
    torch.cuda.synchronize()
evs = [e for e in prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")]
kern = {}
for e in prof.key_averages():
    if e.device_time_total > 0 and e.count > 0 and "cuda" not in e.key.lower()[:5]:
        pass
print("--- CUDA kernels launched by ONE run() call:")
tot = 0.0
rows = []
for e in prof.key_averages():
    dt = getattr(e, "device_time_total", 0.0) or 0.0
    if dt > 0 and getattr(e, "device_type", None) is not None:
        rows.append((e.key, e.count, dt))
for kkey, cnt, dt in sorted(rows, key=lambda r: -r[2]):
    print(f"    {dt:10.1f} us  n={cnt:3d}  {kkey[:110]}")
print("--- raw kernel-event names:")
names = []
for e in prof.events():
    if str(getattr(e, "device_type", "")).endswith("CUDA") or getattr(e, "cuda_time_total", 0):
        pass
try:
    ka = prof.profiler.kineto_results.events()
except Exception:
    ka = []
print("N kernel rows above (dedup):", len(rows))
del o
