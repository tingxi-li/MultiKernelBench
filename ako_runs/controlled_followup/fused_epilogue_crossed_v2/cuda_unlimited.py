#!/usr/bin/env python3
"""Corrected descendant of the Phase-2 inline-PTX fused builder."""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"
CHECKED_LAUNCH_DIR = HERE.parent / "legacy_cuda_harness_fix"
for path in (str(PHASE2), str(PHASE1), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
import common  # noqa: E402
import common2  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402
from variants2 import cuda_fused_common as cfc  # noqa: E402
from variants2 import fused_cuda_unlimited as historical  # noqa: E402
try:  # noqa: E402
    from .core import ProtocolError, ptxas_kernel_resources
except ImportError:  # direct script execution
    from core import ProtocolError, ptxas_kernel_resources


_CUT = "    /* ---------------- epilogue ---"
_KERNEL_MARK = "__global__ void __launch_bounds__(THREADS) mma_gemm("

REGS_DEVICE = r"""
__device__ __forceinline__ float gelu_exact(float v) {
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}
"""

SMEM_DEVICE = r"""
#define CPAD 4
#define CSTRIDE (BN + CPAD)
__device__ __forceinline__ float gelu_exact(float v) {
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}
__device__ __forceinline__ void fused_epilogue(const float* __restrict__ Cs,
                                               const float* __restrict__ Bias,
                                               float* __restrict__ Cg,
                                               int bm, int bn, int tid, int N) {
    constexpr int NELEM = BM * BN;
    for (int g = tid; g < NELEM; g += THREADS) {
        const int r = g / BN, c = g % BN;
        float v = Cs[r * CSTRIDE + c];
#if HAS_BIAS
        v += Bias[bn + c];
#endif
#if HAS_GELU
        v = gelu_exact(v);
#endif
        Cg[(long long)(bm + r) * N + bn + c] = v;
    }
}
"""

WRAPPER = r"""
#include "checked_cuda_launch.h"
#include <ATen/cuda/CUDAContext.h>

torch::Tensor fused(torch::Tensor A, torch::Tensor B, torch::Tensor Bias) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Bias.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.get_device() == B.get_device() && A.get_device() == Bias.get_device(),
                "same CUDA device required");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Bias.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kHalf && B.scalar_type() == torch::kHalf,
                "operands must be fp16");
    TORCH_CHECK(Bias.scalar_type() == torch::kFloat32, "bias must be fp32");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == PROBLEM_M && A.size(1) == PROBLEM_K,
                "A shape mismatch");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == PROBLEM_K && B.size(1) == PROBLEM_N,
                "B shape mismatch");
    TORCH_CHECK(Bias.dim() == 1 && Bias.numel() == PROBLEM_N, "bias shape mismatch");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({PROBLEM_M, PROBLEM_N}, opts);
    static bool attr_set = false;
    if (!attr_set) {
        checked_dynamic_smem((const void*)mma_gemm, DYNAMIC_SMEM_BYTES, "mma_gemm");
        attr_set = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(PROBLEM_N / BN, PROBLEM_M / BM), block(THREADS);
    mma_gemm<<<grid, block, DYNAMIC_SMEM_BYTES, stream>>>(
        A.data_ptr(), B.data_ptr(), Bias.data_ptr<float>(),
        C.data_ptr<float>(), PROBLEM_M, PROBLEM_N, PROBLEM_K);
    checked_kernel_launch("mma_gemm");
#if HAS_SOFTMAX
    auto Y = torch::empty({PROBLEM_M, PROBLEM_N}, opts);
    softmax_kernel<<<PROBLEM_M, SOFT_TH, 0, stream>>>(C.data_ptr<float>(),
                                                      Y.data_ptr<float>());
    checked_kernel_launch("softmax_kernel");
    return Y;
#else
    return C;
#endif
}
"""


def _geometry(cfg):
    if cfg.arith != "fp16" or cfg.cast != "precast":
        raise ValueError("corrected cuda_unlimited requires fp16/precast")
    if cfg.extra.get("wcache", "cached") != "cached":
        raise ValueError("corrected cuda_unlimited requires cached weight")
    epilogue = cfg.extra.get("epilogue")
    if epilogue not in ("regs", "smem"):
        raise ValueError("epilogue must be 'regs' or 'smem'")
    if any(total % tile for total, tile in ((cfg.M, cfg.BM), (cfg.N, cfg.BN), (cfg.K, cfg.BK))):
        raise ValueError("grid tile must divide the fused shape")
    if cfg.kc and (cfg.kc % cfg.BK or cfg.K % cfg.kc):
        raise ValueError("kc must be a BK multiple that divides K")

    p1 = historical.p1
    apad, bpad = 8, 8

    def operand_bytes(ap, bp):
        return (cfg.BM * (cfg.BK + ap) + cfg.BK * (cfg.BN + bp)) * 2 * cfg.stages

    operand = operand_bytes(apad, bpad)
    if operand > p1._MAX_SMEM_OPTIN:
        apad = bpad = 0
        operand = operand_bytes(apad, bpad)
    epilogue_tile = cfg.BM * (cfg.BN + 4) * 4 if epilogue == "smem" else 0
    dynamic = max(operand, epilogue_tile)
    if dynamic > p1._MAX_SMEM_OPTIN:
        raise RuntimeError(
            f"dynamic shared memory {dynamic} B exceeds the sm_89 cap "
            f"{p1._MAX_SMEM_OPTIN} B"
        )
    return epilogue, apad, bpad, operand, epilogue_tile, dynamic


def make_source(cfg) -> dict:
    epilogue, apad, bpad, operand, epilogue_tile, dynamic = _geometry(cfg)
    arm = common2.FUSED_ARMS[cfg.variant]
    kcb = cfg.kc // cfg.BK if cfg.kc else 0
    src = historical.p1._MMA_SRC.substitute(
        BM=cfg.BM,
        BN=cfg.BN,
        BK=cfg.BK,
        THREADS=cfg.threads,
        STAGES=cfg.stages,
        KCB=kcb,
        F32G=0,
        APAD=apad,
        BPAD=bpad,
        SMEM_BYTES=operand,
    )
    body = src[: src.index(_CUT)].replace(
        "        const void* __restrict__ Ag, const void* __restrict__ Bg,\n"
        "        float* __restrict__ Cg, int M, int N, int K) {",
        "        const void* __restrict__ Ag, const void* __restrict__ Bg,\n"
        "        const float* __restrict__ Bias,\n"
        "        float* __restrict__ Cg, int M, int N, int K) {",
        1,
    )
    if "const float* __restrict__ Bias" not in body or _KERNEL_MARK not in body:
        raise RuntimeError("failed to splice corrected mma_gemm")
    defines = (
        cfc.arm_defines(arm, cfg.N)
        + f"#define PROBLEM_M {cfg.M}\n#define PROBLEM_N {cfg.N}\n"
        + f"#define PROBLEM_K {cfg.K}\n#define DYNAMIC_SMEM_BYTES {dynamic}\n"
    )
    body = body.replace(
        _KERNEL_MARK,
        defines + (SMEM_DEVICE if epilogue == "smem" else REGS_DEVICE) + _KERNEL_MARK,
        1,
    )
    full = (
        body
        + (historical.EPI_SMEM if epilogue == "smem" else historical.EPI_REGS)
        + (cfc.SOFTMAX_KERNEL if arm["softmax"] else "")
        + WRAPPER
    )
    return {
        "source": full,
        "operand_shared_bytes": operand,
        "epilogue_tile_shared_bytes": epilogue_tile,
        "total_dynamic_shared_bytes": dynamic,
    }


def build(cfg) -> common2.Built2:
    common.setup_cuda_env()
    generated = make_source(cfg)
    source = generated["source"]
    epilogue = cfg.extra["epilogue"]
    arm = common2.FUSED_ARMS[cfg.variant]
    kcb = cfg.kc // cfg.BK if cfg.kc else 0
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    name = (
        f"crossed_v2_unl_{cfg.variant}_{cfg.BM}x{cfg.BN}x{cfg.BK}_"
        f"t{cfg.threads}_kc{cfg.kc}_s{cfg.stages}_{epilogue}_{digest[:8]}"
    )
    started = time.perf_counter()
    os.makedirs(common.ARTIFACTS_DIR, exist_ok=True)
    build_log_path = os.path.join(common.ARTIFACTS_DIR, f"{name}.ptxas.log")
    with historical.p1._FDCapture(build_log_path) as capture:
        module = load_inline(
            name=name,
            cpp_sources=(
                "#include <torch/extension.h>\n"
                "torch::Tensor fused(torch::Tensor A, torch::Tensor B, torch::Tensor Bias);"
            ),
            cuda_sources=source,
            functions=["fused"],
            with_cuda=True,
            verbose=True,
            extra_include_paths=[str(CHECKED_LAUNCH_DIR)],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "-Xptxas=-v",
                "-gencode=arch=compute_89,code=sm_89",
            ],
        )
    try:
        kernel_resources = ptxas_kernel_resources(capture.text, "mma_gemm")
        resource_error = None
    except ProtocolError as exc:
        kernel_resources, resource_error = None, str(exc)
    weight = common2.weight_fn("cached")

    def run(x, W, bias):
        return module.fused(x, weight(W), bias)

    x0 = torch.zeros((cfg.M, cfg.K), dtype=torch.float16, device="cuda")
    w0 = torch.zeros((cfg.K, cfg.N), dtype=torch.float16, device="cuda")
    b0 = torch.zeros((cfg.N,), dtype=torch.float32, device="cuda")
    module.fused(x0, w0, b0)
    torch.cuda.synchronize()
    del x0, w0, b0
    torch.cuda.empty_cache()

    artifacts = {
        "backend_detail": (
            "corrected inline-PTX MMA descendant; checked attribute and launches; "
            f"source-level epilogue={epilogue}; bias={arm['bias']} gelu={arm['gelu']}; KCB={kcb}; "
            "physical spills are reported by the per-kernel ptxas census when compilation is fresh"
        ),
        "block": [cfg.threads, 1, 1],
        "cuda_source": source,
        "cuda_source_sha256": digest,
        "build_log_path": build_log_path,
        "ptxas_log": capture.text,
        "kernel_resources": kernel_resources,
        "kernel_resources_error": resource_error,
        "epilogue": epilogue,
        "epilogue_tile_bytes": generated["epilogue_tile_shared_bytes"],
        "epilogue_tile_shared_bytes": generated["epilogue_tile_shared_bytes"],
        "grid": [cfg.N // cfg.BN, cfg.M // cfg.BM, 1],
        "has_bias": bool(arm["bias"]),
        "has_gelu": bool(arm["gelu"]),
        "n_kernels": 2 if arm["softmax"] else 1,
        "operand_smem_bytes": generated["operand_shared_bytes"],
        "operand_shared_bytes": generated["operand_shared_bytes"],
        "dynamic_smem_bytes": generated["total_dynamic_shared_bytes"],
        "shared_bytes": generated["total_dynamic_shared_bytes"],
        "total_dynamic_shared_bytes": generated["total_dynamic_shared_bytes"],
        "wcache": "cached",
    }
    return common2.Built2(
        run=run,
        compile_s=time.perf_counter() - started,
        artifacts=artifacts,
        notes=f"crossed-v2 corrected cuda_unlimited {cfg.variant} epilogue={epilogue}",
        n_kernels=artifacts["n_kernels"],
        x_dtype=torch.float16,
    )
