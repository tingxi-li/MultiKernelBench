#!/usr/bin/env python3
"""TileLang GEMM with an explicitly forced shared-memory bias/GELU stage."""
# TileLang resolves eager annotation types from live closure globals; do not add
# postponed annotations to this module.
import time


def build(cfg):
    import torch
    import tilelang
    import tilelang.language as T

    import common2

    M, N, K = cfg.M, cfg.N, cfg.K
    BM, BN, BK = cfg.BM, cfg.BN, cfg.BK
    threads, stages, kc = cfg.threads, cfg.stages, cfg.kc
    if cfg.arith != "fp16" or cfg.cast != "precast":
        raise ValueError("TileLang smem strategy requires fp16/precast")
    if cfg.extra.get("wcache") != "cached":
        raise ValueError("TileLang smem strategy requires cached weight")
    if any(total % tile for total, tile in ((M, BM), (N, BN), (K, BK))):
        raise ValueError("grid tile must divide the frozen fused shape")
    if not kc or K % kc or kc % BK:
        raise ValueError("KC must be a nonzero multiple of BK and divide K")
    n_chunks, inner = K // kc, kc // BK
    shared_epilogue_bytes = BM * (BN + 4) * 4

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((M, K), "float16"),
                 B: T.Tensor((K, N), "float16"),
                 Bias: T.Tensor((N,), "float32"),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float16")
                Bs = T.alloc_shared((BK, BN), "float16")
                Cs = T.alloc_shared((BM, BN + 4), "float32")
                accumulator = T.alloc_fragment((BM, BN), "float32")
                chunk = T.alloc_fragment((BM, BN), "float32")
                T.clear(accumulator)
                for c in T.serial(n_chunks):
                    T.clear(chunk)
                    for ko in T.Pipelined(inner, num_stages=stages):
                        T.copy(A[by * BM, c * kc + ko * BK], As)
                        T.copy(B[c * kc + ko * BK, bx * BN], Bs)
                        T.gemm(As, Bs, chunk)
                    for i, j in T.Parallel(BM, BN):
                        accumulator[i, j] += chunk[i, j]
                # The forced strategy: materialize the complete accumulator
                # tile in shared memory before either epilogue expression.
                for i, j in T.Parallel(BM, BN):
                    Cs[i, j] = accumulator[i, j]
                T.sync_threads()
                for i, j in T.Parallel(BM, BN):
                    value = Cs[i, j] + Bias[bx * BN + j]
                    Cs[i, j] = value * T.float32(0.5) * (
                        T.float32(1.0) + T.erf(value * T.float32(0.70710678118654752440))
                    )
                T.sync_threads()
                for i, j in T.Parallel(BM, BN):
                    C[by * BM + i, bx * BN + j] = Cs[i, j]
        return main

    # Reuse the already checked TileLang common row-softmax implementation.
    from ako_runs.phase2_fused_sdpa.variants2.fused_tilelang import _kernel_softmax

    started = time.perf_counter()
    gemm = kernel()
    softmax = _kernel_softmax(M, N, common2.SOFT_THREADS)
    weight = common2.weight_fn("cached")

    def run(x, W, bias):
        intermediate = gemm(x, weight(W), bias)
        return softmax(intermediate)

    x0 = torch.zeros((M, K), device="cuda", dtype=torch.float16)
    w0 = torch.zeros((K, N), device="cuda", dtype=torch.float16)
    b0 = torch.zeros((N,), device="cuda", dtype=torch.float32)
    intermediate0 = gemm(x0, w0, b0)
    softmax(intermediate0)
    torch.cuda.synchronize()
    del x0, w0, b0, intermediate0
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - started
    artifacts = {
        "backend_detail": "TileLang explicit fp32 accumulator tile in shared memory, then exact GELU; common TileLang row-softmax",
        "block": [threads, 1, 1],
        "epilogue": "smem_staged",
        "grid": [N // BN, M // BM, 1],
        "n_kernels": 2,
        "shared_epilogue_bytes": shared_epilogue_bytes,
        "wcache": "cached",
    }
    try:
        artifacts["cuda_source_gemm"] = gemm.get_kernel_source()
        artifacts["cuda_source_softmax"] = softmax.get_kernel_source()
    except Exception as exc:
        artifacts["source_capture_error"] = repr(exc)
    return common2.Built2(
        run=run,
        compile_s=compile_s,
        artifacts=artifacts,
        notes="crossed-v1 TileLang explicitly shared-staged bias/exact-GELU plus common softmax",
        n_kernels=2,
        x_dtype=torch.float16,
    )
