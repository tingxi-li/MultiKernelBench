"""Versioned CUDA candidates for the fused reachability audit.

The v1 CUDA lanes materialize a complete ``BM x BN`` fp32 accumulator tile in
shared memory solely to attach the bias/GELU epilogue.  That makes the wide
tiles structurally unlaunchable on sm_89.  These prospective v2 candidates
leave the Phase-1 GEMM realization intact, store its fp32 accumulator tile to
the global intermediate it already owns, and apply bias + exact GELU + row
softmax in the existing second kernel.  No receipt-bound v1 source is edited.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

# Match the receipt-bound Phase-1 CUDA toolchain before torch's extension
# module snapshots CUDA_HOME at import time.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
_CUDA_BIN = "/usr/local/cuda-13.1/bin"
if _CUDA_BIN not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = _CUDA_BIN + ":" + os.environ.get("PATH", "")

import torch
from torch.utils.cpp_extension import load_inline


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs" / "phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs" / "phase2_fused_sdpa"
for path in (str(PHASE2), str(PHASE1)):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402
import common2  # noqa: E402
from variants import cuda_noptx_gemm as p1_noptx  # noqa: E402
from variants import cuda_unlimited_gemm as p1_unlimited  # noqa: E402


LANES = ("cuda_noptx", "cuda_unlimited")
MAX_SMEM_OPTIN = 101_376


POSTPROCESS_SOFTMAX = r"""
#define POST_TH 256
#define POST_EPT (FUSED_N / POST_TH)
#define POST_NWARPS (POST_TH / 32)

__device__ __forceinline__ float post_gelu_exact(float v) {
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}

__global__ __launch_bounds__(POST_TH)
void postprocess_softmax_kernel(const float* __restrict__ X,
                                const float* __restrict__ Bias,
                                float* __restrict__ Y) {
    __shared__ float sm_m[POST_NWARPS];
    __shared__ float sm_s[POST_NWARPS];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const float* __restrict__ xr = X + (size_t)row * FUSED_N;
    float* __restrict__ yr = Y + (size_t)row * FUSED_N;

    float lexp[POST_EPT];
    float m = -3.402823466e+38f;
#pragma unroll
    for (int k = 0; k < POST_EPT; ++k) {
        const int col = tid * POST_EPT + k;
        const float v = post_gelu_exact(xr[col] + Bias[col]);
        lexp[k] = v;
        m = fmaxf(m, v);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) sm_m[wid] = m;
    __syncthreads();
#pragma unroll
    for (int s = POST_NWARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_m[tid] = fmaxf(sm_m[tid], sm_m[tid + s]);
        __syncthreads();
    }
    const float row_max = sm_m[0];

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < POST_EPT; ++k) {
        const float e = __expf(lexp[k] - row_max);
        lexp[k] = e;
        sum += e;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) sm_s[wid] = sum;
    __syncthreads();
#pragma unroll
    for (int s = POST_NWARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_s[tid] += sm_s[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / sm_s[0];
#pragma unroll
    for (int k = 0; k < POST_EPT; ++k)
        yr[tid * POST_EPT + k] = lexp[k] * inv;
}
"""


NOPT_WRAPPER = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor fused_streamed(torch::Tensor A, torch::Tensor B,
                             torch::Tensor Bias) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Bias.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(A.get_device() == B.get_device() &&
                A.get_device() == Bias.get_device(), "same CUDA device required");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Bias.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kHalf &&
                B.scalar_type() == torch::kHalf,
                "A and B must be fp16");
    TORCH_CHECK(Bias.scalar_type() == torch::kFloat32, "bias must be fp32");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == M_ && A.size(1) == K_, "A shape");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == K_ && B.size(1) == N_, "B shape");
    TORCH_CHECK(Bias.dim() == 1 && Bias.numel() == N_, "bias shape");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({M_, N_}, opts);
    static bool attr_done = false;
    if (!attr_done) {
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute(", SMEM_BYTES,
                    ") failed: ", cudaGetErrorString(e));
        attr_done = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    gemm_kernel<<<dim3(N_ / BN, M_ / BM), dim3(THREADS), SMEM_BYTES, stream>>>(
        reinterpret_cast<const GTYPE*>(A.data_ptr()),
        reinterpret_cast<const GTYPE*>(B.data_ptr()), C.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto Y = torch::empty({M_, N_}, opts);
    postprocess_softmax_kernel<<<M_, POST_TH, 0, stream>>>(
        C.data_ptr<float>(), Bias.data_ptr<float>(), Y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
"""


UNLIMITED_DEFINES = r"""
#include <c10/cuda/CUDAException.h>
#define FUSED_M 1024
#define FUSED_K 8192
#define FUSED_N 8192
"""


UNLIMITED_WRAPPER = r"""

torch::Tensor fused_streamed(torch::Tensor A, torch::Tensor B,
                             torch::Tensor Bias) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Bias.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(A.get_device() == B.get_device() &&
                A.get_device() == Bias.get_device(), "same CUDA device required");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Bias.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kHalf &&
                B.scalar_type() == torch::kHalf,
                "A and B must be fp16");
    TORCH_CHECK(Bias.scalar_type() == torch::kFloat32, "bias must be fp32");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == FUSED_M && A.size(1) == FUSED_K,
                "A shape");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == FUSED_K && B.size(1) == FUSED_N,
                "B shape");
    TORCH_CHECK(Bias.dim() == 1 && Bias.numel() == FUSED_N, "bias shape");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({FUSED_M, FUSED_N}, opts);
    static bool attr_done = false;
    if (!attr_done) {
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)mma_gemm,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute(", SMEM_BYTES,
                    ") failed: ", cudaGetErrorString(e));
        attr_done = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    mma_gemm<<<dim3(FUSED_N / BN, FUSED_M / BM), dim3(THREADS),
               SMEM_BYTES, stream>>>(A.data_ptr(), B.data_ptr(),
                                    C.data_ptr<float>(), FUSED_M, FUSED_N,
                                    FUSED_K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto Y = torch::empty({FUSED_M, FUSED_N}, opts);
    postprocess_softmax_kernel<<<FUSED_M, POST_TH, 0, stream>>>(
        C.data_ptr<float>(), Bias.data_ptr<float>(), Y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
"""


CPP_DECL = (
    "#include <torch/extension.h>\n"
    "torch::Tensor fused_streamed(torch::Tensor A, torch::Tensor B, "
    "torch::Tensor Bias);"
)


class BuiltCandidate:
    def __init__(self, run, compile_s: float, artifacts: dict[str, Any]):
        self.run = run
        self.compile_s = compile_s
        self.artifacts = artifacts
        self.x_dtype = torch.float16
        self.n_kernels = 2


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load(name: str, source: str):
    return load_inline(
        name=name,
        cpp_sources=CPP_DECL,
        cuda_sources=source,
        functions=["fused_streamed"],
        with_cuda=True,
        verbose=False,
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-Xptxas=-v",
            "-gencode=arch=compute_89,code=sm_89",
        ],
    )


def _build_noptx(cfg) -> BuiltCandidate:
    saved = (common.M, common.N, common.K)
    common.M, common.N, common.K = cfg.M, cfg.N, cfg.K
    try:
        generated = p1_noptx._make_source(cfg)
    finally:
        common.M, common.N, common.K = saved
    prefix = "\n#define FUSED_N 8192\n"
    source = generated["kernel_src"] + prefix + POSTPROCESS_SOFTMAX + NOPT_WRAPPER
    for forbidden in ("asm(", "asm (", "asm volatile", "__asm"):
        if forbidden in source:
            raise RuntimeError(f"inline asm {forbidden!r} found in cuda_noptx v2")
    name = (
        f"cfr2_n_{cfg.BM}_{cfg.BN}_{cfg.BK}_t{cfg.threads}_"
        f"s{generated['info']['stages_smem']}_kc{cfg.kc}"
    )
    started = time.perf_counter()
    module = _load(name, source)
    compile_s = time.perf_counter() - started
    weight = common2.weight_fn("cached")

    def run(x, W, bias):
        return module.fused_streamed(x, weight(W), bias)

    return BuiltCandidate(
        run,
        compile_s,
        {
            "lane": "cuda_noptx",
            "epilogue_policy": "global_intermediate_then_bias_gelu_softmax",
            "shared_bytes": generated["smem"],
            "shared_padding_halfs": {
                "A": generated["info"]["pad_halfs"],
                "B": generated["info"]["pad_halfs"],
            },
            "configured_stages": cfg.stages,
            "realized_stages": generated["info"]["stages_smem"],
            "deviations": generated["deviations"],
            "cuda_source_sha256": _sha256_text(source),
            "extension_name": name,
            "n_kernels": 2,
        },
    )


def _unlimited_smem(cfg) -> tuple[int, int, int]:
    def amount(apad: int, bpad: int) -> int:
        return (
            cfg.BM * (cfg.BK + apad) + cfg.BK * (cfg.BN + bpad)
        ) * 2 * cfg.stages

    apad = bpad = 8
    smem = amount(apad, bpad)
    if smem > MAX_SMEM_OPTIN:
        apad = bpad = 0
        smem = amount(apad, bpad)
    if smem > MAX_SMEM_OPTIN:
        raise RuntimeError(
            f"A/B staging needs {smem} B, above sm_89 cap {MAX_SMEM_OPTIN}"
        )
    return smem, apad, bpad


def _build_unlimited(cfg) -> BuiltCandidate:
    if cfg.kc % cfg.BK:
        raise ValueError("kc must be a multiple of BK")
    smem, apad, bpad = _unlimited_smem(cfg)
    templated = p1_unlimited._MMA_SRC.substitute(
        BM=cfg.BM,
        BN=cfg.BN,
        BK=cfg.BK,
        THREADS=cfg.threads,
        STAGES=cfg.stages,
        KCB=cfg.kc // cfg.BK,
        F32G=0,
        APAD=apad,
        BPAD=bpad,
        SMEM_BYTES=smem,
    )
    marker = "\ntorch::Tensor gemm(torch::Tensor A, torch::Tensor B)"
    if marker not in templated:
        raise RuntimeError("could not separate Phase-1 unlimited GEMM wrapper")
    kernel_source = templated[: templated.index(marker)]
    source = (
        kernel_source + UNLIMITED_DEFINES + POSTPROCESS_SOFTMAX
        + UNLIMITED_WRAPPER
    )
    name = (
        f"cfr2_u_{cfg.BM}_{cfg.BN}_{cfg.BK}_t{cfg.threads}_"
        f"s{cfg.stages}_kc{cfg.kc}_p{apad}"
    )
    started = time.perf_counter()
    module = _load(name, source)
    compile_s = time.perf_counter() - started
    weight = common2.weight_fn("cached")

    def run(x, W, bias):
        return module.fused_streamed(x, weight(W), bias)

    return BuiltCandidate(
        run,
        compile_s,
        {
            "lane": "cuda_unlimited",
            "epilogue_policy": "global_intermediate_then_bias_gelu_softmax",
            "shared_bytes": smem,
            "shared_padding_halfs": {"A": apad, "B": bpad},
            "configured_stages": cfg.stages,
            "realized_stages": cfg.stages,
            "deviations": ["shared-memory padding dropped"] if apad == 0 else [],
            "cuda_source_sha256": _sha256_text(source),
            "extension_name": name,
            "n_kernels": 2,
        },
    )


def build(lane: str, cfg) -> BuiltCandidate:
    """Build one streamed-epilogue candidate for the frozen fused shape."""
    common.setup_cuda_env()
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}")
    if (cfg.M, cfg.K, cfg.N) != (1024, 8192, 8192):
        raise ValueError("candidate is frozen to M=1024,K=8192,N=8192")
    if cfg.arith != "fp16" or cfg.cast != "precast" or cfg.variant != "GBGS":
        raise ValueError("candidate requires GBGS/fp16/precast")
    if cfg.extra.get("wcache") != "cached":
        raise ValueError("candidate requires cached fp16 weight")
    if cfg.extra.get("epilogue") != "streamed_global":
        raise ValueError("candidate requires epilogue=streamed_global")
    return _build_noptx(cfg) if lane == "cuda_noptx" else _build_unlimited(cfg)


def make_config(job: dict[str, Any]):
    overrides: dict[str, Any] = {}
    for component in job["set"].split(","):
        key, raw = (value.strip() for value in component.split("=", 1))
        if key in ("cast", "arith", "algo"):
            overrides[key] = raw
        elif key.startswith("x_"):
            overrides.setdefault("extra", {})[key[2:]] = raw
        else:
            overrides[key] = int(raw)
    return common2.make_fused_config(job["lane"], "GBGS", **overrides)
