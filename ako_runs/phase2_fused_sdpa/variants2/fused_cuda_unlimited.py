"""cuda_unlimited lane of the Phase-2 fused ladder: inline `mma.sync` PTX.

Like the noptx lane, the GEMM is Phase 1's and is reused rather than rewritten:
this module takes Phase 1's `_MMA_SRC` template, truncates it at its epilogue
marker, and appends the fused epilogue plus the softmax kernel. Everything above
that cut -- `ldmatrix.sync.aligned.m8n8.x4[.trans]`, `mma.sync.aligned.m16n8k16`,
`cp.async.cg.shared.global`, the KCB flush, the STAGES pipeline -- is byte-
identical to Phase 1's.

TWO EPILOGUES, and the difference is a finding rather than an accident:

  epilogue=regs   Phase 1's own store path already computes explicit (gr, gc)
                  global indices, because this lane wrote the `ldmatrix`/`mma`
                  pairing itself and therefore knows where each accumulator
                  register lives. Bias -- indexed by column -- can be applied
                  straight in registers, with no staging.
  epilogue=smem   the same tile pushed through shared memory and then walked
                  with (row, col) indices, which is the only thing the WMMA lane
                  can do, since fragment layout is not part of the wmma API.

`smem` is the matched point (it is what cuda_noptx must do); `regs` is this
lane's native-tuning point. Reporting only one of them would either hide a real
capability of the lower abstraction level or silently unmatch the comparison.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

import common
import common2
from . import cuda_fused_common as cfc

_P1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "phase1_matmul")
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
from variants import cuda_unlimited_gemm as p1  # noqa: E402

_CUT = "    /* ---------------- epilogue ---"

# Register epilogue: Phase 1's store loop with bias/GELU folded in before the
# float2 writes. The index arithmetic is Phase 1's, unchanged.
EPI_REGS = r"""
    /* ---------------- Phase-2 fused epilogue, in registers --------------- */
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int x = (i*NT+j)*4;
#if KCB > 0
            float v0 = sacc[x+0] + acc[x+0], v1 = sacc[x+1] + acc[x+1];
            float v2 = sacc[x+2] + acc[x+2], v3 = sacc[x+3] + acc[x+3];
#else
            float v0 = acc[x+0], v1 = acc[x+1], v2 = acc[x+2], v3 = acc[x+3];
#endif
            int gr = m0 + wm + i*16 + (lane >> 2);
            int gc = n0 + wn + j*8  + ((lane & 3) << 1);
#if HAS_BIAS
            const float b0 = Bias[gc], b1 = Bias[gc+1];
            v0 += b0; v1 += b1; v2 += b0; v3 += b1;
#endif
#if HAS_GELU
            v0 = gelu_exact(v0); v1 = gelu_exact(v1);
            v2 = gelu_exact(v2); v3 = gelu_exact(v3);
#endif
            *(float2*)(Cg + (long long)gr*N + gc)       = make_float2(v0, v1);
            *(float2*)(Cg + (long long)(gr+8)*N + gc)   = make_float2(v2, v3);
        }
    }
}
"""

# Shared-memory epilogue: the matched one. Same staging the WMMA lane is forced
# into, so the two CUDA lanes differ only in the inner loop's instruction path.
EPI_SMEM = r"""
    /* ---------------- Phase-2 fused epilogue, staged through smem -------- */
    __syncthreads();                       /* A/B stage buffers are dead now */
    float* Cs = reinterpret_cast<float*>(sraw);
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int x = (i*NT+j)*4;
#if KCB > 0
            float v0 = sacc[x+0] + acc[x+0], v1 = sacc[x+1] + acc[x+1];
            float v2 = sacc[x+2] + acc[x+2], v3 = sacc[x+3] + acc[x+3];
#else
            float v0 = acc[x+0], v1 = acc[x+1], v2 = acc[x+2], v3 = acc[x+3];
#endif
            const int r = wm + i*16 + (lane >> 2);
            const int c = wn + j*8  + ((lane & 3) << 1);
            Cs[r * CSTRIDE + c]           = v0;
            Cs[r * CSTRIDE + c + 1]       = v1;
            Cs[(r + 8) * CSTRIDE + c]     = v2;
            Cs[(r + 8) * CSTRIDE + c + 1] = v3;
        }
    }
    __syncthreads();
    fused_epilogue(Cs, Bias, Cg, m0, n0, threadIdx.x, N);
}
"""

WRAPPER = r"""
torch::Tensor fused(torch::Tensor A, torch::Tensor B, torch::Tensor Bias) {
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(A.scalar_type() == torch::kHalf && B.scalar_type() == torch::kHalf,
                "operands must be fp16");
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({M, N}, opts);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(mma_gemm,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FSMEM_BYTES);
        attr_set = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(N/BN, M/BM), block(THREADS);
    mma_gemm<<<grid, block, FSMEM_BYTES, stream>>>(
        A.data_ptr(), B.data_ptr(), Bias.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
#if HAS_SOFTMAX
    auto Y = torch::empty({M, N}, opts);
    softmax_kernel<<<M, SOFT_TH, 0, stream>>>(C.data_ptr<float>(),
                                              Y.data_ptr<float>());
    return Y;
#else
    return C;
#endif
}
"""


def build(cfg) -> common2.Built2:
    common.setup_cuda_env()
    arm = common2.FUSED_ARMS[cfg.variant]
    wmode = cfg.extra.get("wcache", "cached")
    epi = cfg.extra.get("epilogue", "smem")
    if wmode == "native":
        raise NotImplementedError(
            "cuda_unlimited lane implements the two-way cached/uncached factor "
            "only; a native (N,K) B operand would need a different ldmatrix "
            "staging and would confound the factor")
    if epi not in ("smem", "regs"):
        raise ValueError(f"epilogue={epi!r} must be 'smem' or 'regs'")
    if cfg.arith != "fp16":
        raise ValueError("the fused ladder is fp16-only in this lane")

    kcb = (cfg.kc // cfg.BK) if cfg.kc else 0
    if cfg.kc and cfg.kc % cfg.BK:
        raise ValueError(f"kc={cfg.kc} must be a multiple of BK={cfg.BK}")

    apad, bpad = 8, 8
    def _smem(ap, bp):
        return (cfg.BM * (cfg.BK + ap) + cfg.BK * (cfg.BN + bp)) * 2 * cfg.stages
    smem = _smem(apad, bpad)
    if smem > p1._MAX_SMEM_OPTIN:
        apad, bpad = 0, 0
        smem = _smem(apad, bpad)
    if smem > p1._MAX_SMEM_OPTIN:
        raise RuntimeError(f"smem {smem} B exceeds the sm_89 cap for this geometry")

    src = p1._MMA_SRC.substitute(
        BM=cfg.BM, BN=cfg.BN, BK=cfg.BK, THREADS=cfg.threads,
        STAGES=cfg.stages, KCB=kcb, F32G=0, APAD=apad, BPAD=bpad,
        SMEM_BYTES=smem)
    cut = src.index(_CUT)
    body = src[:cut]
    # The kernel signature gains the bias pointer; nothing else about the
    # kernel's prologue or main loop is touched.
    body = body.replace(
        "        const void* __restrict__ Ag, const void* __restrict__ Bg,\n"
        "        float* __restrict__ Cg, int M, int N, int K) {",
        "        const void* __restrict__ Ag, const void* __restrict__ Bg,\n"
        "        const float* __restrict__ Bias,\n"
        "        float* __restrict__ Cg, int M, int N, int K) {", 1)
    if "const float* __restrict__ Bias" not in body:
        raise RuntimeError("failed to splice the bias pointer into mma_gemm")

    # The truncation point is INSIDE mma_gemm's body, so the device helpers
    # cannot simply be appended there -- they are spliced in above the kernel,
    # after the template's #define block has established BM/BN/THREADS.
    _MARK = "__global__ void __launch_bounds__(THREADS) mma_gemm("
    if _MARK not in body:
        raise RuntimeError("failed to locate mma_gemm for helper splice")
    body = body.replace(
        _MARK, cfc.arm_defines(arm, cfg.N) + cfc.EPILOGUE_DEVICE + _MARK, 1)
    full = (body + (EPI_SMEM if epi == "smem" else EPI_REGS)
            + (cfc.SOFTMAX_KERNEL if arm["softmax"] else "")
            + WRAPPER)
    # gelu_exact is used by both epilogues, so EPILOGUE_DEVICE is always in.

    name = ("p2unl_%s_%dx%dx%d_t%d_kc%d_s%d_%s_%s"
            % (cfg.variant, cfg.BM, cfg.BN, cfg.BK, cfg.threads,
               cfg.kc, cfg.stages, wmode, epi))

    t0 = time.perf_counter()
    mod = load_inline(
        name=name,
        cpp_sources="#include <torch/extension.h>\ntorch::Tensor fused("
                    "torch::Tensor A, torch::Tensor B, torch::Tensor Bias);",
        cuda_sources=full, functions=["fused"], with_cuda=True, verbose=False,
        extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas=-v",
                           "-gencode=arch=compute_89,code=sm_89"])

    wf = common2.weight_fn(wmode)
    xcast = cfg.cast == "in_region"
    xf = (lambda x: x.half()) if xcast else (lambda x: x)

    def run(x, W, b):
        return mod.fused(xf(x), wf(W), b)

    xw = torch.zeros((cfg.M, cfg.K), dtype=torch.float16, device="cuda")
    Bw = torch.zeros((cfg.K, cfg.N), dtype=torch.float16, device="cuda")
    bw = torch.zeros((cfg.N,), dtype=torch.float32, device="cuda")
    mod.fused(xw, Bw, bw)
    torch.cuda.synchronize()
    del xw, Bw, bw
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    art = {
        "cuda_source": full,
        "shared_bytes": max(smem, cfg.BM * (cfg.BN + 4) * 4) if epi == "smem" else smem,
        "grid": f"({cfg.N // cfg.BN},{cfg.M // cfg.BM})",
        "block": f"({cfg.threads},1,1)", "ext_name": name,
        "wcache": wmode, "epilogue": epi,
        "n_kernels": 2 if arm["softmax"] else 1,
        "backend_detail": (
            "inline PTX mma.sync.aligned.m16n8k16.f32.f16.f16.f32 + "
            "ldmatrix.sync.aligned.m8n8.x4[.trans] + "
            + ("cp.async.cg.shared.global x %d stages" % cfg.stages
               if cfg.stages > 1 else "synchronous smem stage")
            + f"; KCB={kcb}; epilogue={epi} "
            + ("(fragments staged through smem, matched to the WMMA lane)"
               if epi == "smem" else
               "(bias applied directly to accumulator registers -- possible "
               "only because this lane knows its own mma fragment layout)")
            + f"; bias={arm['bias']} gelu={arm['gelu']} (erff, exact form)"
            + ("; + row-softmax kernel (256 thr, 32 elem/thread cached in "
               "registers, __shfl_down_sync + smem tree)" if arm["softmax"] else "")),
    }
    notes = ("cuda_unlimited fused arm %s (%s) wcache=%s epilogue=%s"
             % (cfg.variant, arm["label"], wmode, epi))
    return common2.Built2(run=run, compile_s=compile_s, artifacts=art,
                          notes=notes, n_kernels=art["n_kernels"],
                          x_dtype=torch.float32 if xcast else torch.float16)
