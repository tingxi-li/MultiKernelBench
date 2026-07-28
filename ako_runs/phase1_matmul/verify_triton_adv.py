"""Adversarial numeric verification of variants/triton_gemm.py. Read-only wrt the module."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, variants
common.setup_cuda_env()

def build_cfg(variant, **over):
    return common.make_config("triton", variant, over.pop("geom", "primary"), **over)

def check_small():
    # non-square, non-symmetric small case; kc must divide K and be mult of BK
    M, N, K = 256, 384, 2048
    for variant, kc in (("D", 512), ("C", 512), ("B", 0), ("A", 0)):
        cfg = build_cfg(variant, M=M, N=N, K=K, kc=kc, BM=128, BN=128, BK=32)
        cfg.M, cfg.N, cfg.K = M, N, K
        b = variants.build(cfg)
        g = torch.Generator(device="cpu").manual_seed(7)
        A32 = torch.rand(M, K, generator=g); B32 = torch.rand(K, N, generator=g)
        A32 = A32.cuda(); B32 = B32.cuda()
        A = A32.half() if b.input_dtype == torch.float16 else A32
        B = B32.half() if b.input_dtype == torch.float16 else B32
        C = b.run(A, B)
        ref = torch.matmul(A32, B32)
        d = (ref - C).abs()
        rel = d.max().item() / ref.abs().max().item()
        # transpose check: compare against ref.T shaped thing is impossible (non-square) -> shape guard
        # partial-sum check: compare against half-K matmul
        half = torch.matmul(A32[:, :K//2], B32[:K//2, :])
        print(f"[small {variant}] shape={tuple(C.shape)} dtype={C.dtype} max|C|={C.abs().max().item():.4f} "
              f"maxabs_err={d.max().item():.6g} rel={rel:.3e} "
              f"corr_vs_full={torch.nn.functional.cosine_similarity(C.reshape(1,-1).double(), ref.reshape(1,-1).double()).item():.10f} "
              f"maxerr_vs_halfK={(half-C).abs().max().item():.4g} nonzero={int((C!=0).sum())}/{C.numel()}")
        del b
        torch.cuda.empty_cache()

def check_full():
    A32, B32 = common.load_inputs("rand", 0, device="cuda")
    ref = torch.matmul(A32, B32)
    for variant in ("A", "B", "C", "D"):
        cfg = build_cfg(variant)
        b = variants.build(cfg)
        A = A32.half().contiguous() if b.input_dtype == torch.float16 else A32
        B = B32.half().contiguous() if b.input_dtype == torch.float16 else B32
        C = b.run(A, B)
        torch.cuda.synchronize()
        st = common.gate_stats(ref, C.float())
        # explicit anti-stale check: perturb A, rerun, output must change
        A2 = A.clone(); A2[0, 0] += (1.0 if A2.dtype == torch.float32 else 1.0)
        C2 = b.run(A2, B)
        changed = (C2 - C).abs().max().item()
        # partial sum check
        halfk = torch.matmul(A32[:, :4096], B32[:4096, :])
        print(f"[full {variant}] max_abs_err={st['max_abs_err']:.6g} pct_fail={st['pct_elems_failing_gate']:.4f} "
              f"gate_pass={st['gate_pass']} mean_abs_err={st['mean_abs_err']:.4g} "
              f"C[0,0]={C[0,0].item():.4f} ref[0,0]={ref[0,0].item():.4f} "
              f"delta_on_perturb={changed:.4g} maxerr_vs_halfK={(halfk-C).abs().max().item():.4g} "
              f"regs={b.artifacts['n_regs']} spills={b.artifacts['n_spills']} smem={b.artifacts['shared_bytes']} "
              f"mma={b.artifacts['ptx_mma_sync_count']} cpasync={b.artifacts['ptx_cp_async_count']} compile_s={b.compile_s:.3f}")
        del b, C, C2, A2
        torch.cuda.empty_cache()

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("small", "both"): check_small()
    if which in ("full", "both"): check_full()
