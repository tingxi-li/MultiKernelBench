#!/usr/bin/env python3
"""Measured no-inline-PTX WMMA source-level register-epilogue candidate."""
from __future__ import annotations

import hashlib
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
from variants import cuda_noptx_gemm as p1  # noqa: E402
from variants2 import cuda_fused_common as cfc  # noqa: E402
from variants2 import fused_cuda_noptx as historical  # noqa: E402
try:  # noqa: E402
    from .core import ptxas_kernel_resources
except ImportError:  # direct script execution
    from core import ptxas_kernel_resources


_TAIL_MARK = "    /* ---- the only departure from Phase 1: stage, then epilogue"

REGS_DEVICE = r"""
__device__ __forceinline__ float gelu_exact(float v) {
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}
"""

REGISTER_EPILOGUE = r"""
    /* Apply the epilogue directly to the accumulator fragment before its
       global store. Physical register/spill allocation is recorded from ptxas. */
    __syncthreads();
    float* bias_tiles = reinterpret_cast<float*>(smem_raw);
    float* warp_bias = bias_tiles + warp * 16 * 16;
#pragma unroll
    for (int j = 0; j < NNF; ++j) {
#if HAS_BIAS
        for (int g = (tid & 31); g < 16 * 16; g += 32)
            warp_bias[g] = Bias[bn + wn * WNT + j * 16 + (g & 15)];
        __syncwarp();
        FragC bias_fragment;
        wmma::load_matrix_sync(bias_fragment, warp_bias, 16, wmma::mem_row_major);
#endif
#pragma unroll
        for (int i = 0; i < NMF; ++i) {
#pragma unroll
            for (int e = 0; e < FRAG_ELEMS; ++e) {
                float value = acc[i][j].x[e];
#if HAS_BIAS
                value += bias_fragment.x[e];
#endif
#if HAS_GELU
                value = gelu_exact(value);
#endif
                acc[i][j].x[e] = value;
            }
            wmma::store_matrix_sync(
                &Cg[(bm + wm * WMT + i * 16) * N_ + bn + wn * WNT + j * 16],
                acc[i][j], N_, wmma::mem_row_major);
        }
        __syncwarp();
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
    TORCH_CHECK(A.dim() == 2 && A.size(0) == M_ && A.size(1) == K_, "A shape");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == K_ && B.size(1) == N_, "B shape");
    TORCH_CHECK(Bias.dim() == 1 && Bias.numel() == N_, "bias shape");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({M_, N_}, opts);
    static bool attr_done = false;
    if (!attr_done) {
        checked_dynamic_smem((const void*)fused_kernel, DYNAMIC_SMEM_BYTES,
                             "fused_kernel");
        attr_done = true;
    }
    dim3 grid(N_ / BN, M_ / BM), block(THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_kernel<<<grid, block, DYNAMIC_SMEM_BYTES, stream>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const half*>(B.data_ptr()),
        Bias.data_ptr<float>(), C.data_ptr<float>());
    checked_kernel_launch("fused_kernel");
#if HAS_SOFTMAX
    auto Y = torch::empty({M_, N_}, opts);
    softmax_kernel<<<M_, SOFT_TH, 0, stream>>>(C.data_ptr<float>(),
                                               Y.data_ptr<float>());
    checked_kernel_launch("softmax_kernel");
    return Y;
#else
    return C;
#endif
}
"""


def _config_source(cfg):
    if cfg.arith != "fp16" or cfg.cast != "precast":
        raise ValueError("cuda_noptx register probe requires fp16/precast")
    if cfg.extra.get("wcache", "cached") != "cached":
        raise ValueError("cuda_noptx register probe requires cached weight")
    if cfg.extra.get("epilogue") != "regs":
        raise ValueError("cuda_noptx register probe requires epilogue=regs")
    saved = (common.M, common.N, common.K)
    common.M, common.N, common.K = cfg.M, cfg.N, cfg.K
    try:
        return p1._make_source(cfg)
    finally:
        common.M, common.N, common.K = saved


def make_source(cfg) -> dict:
    generated = _config_source(cfg)
    arm = common2.FUSED_ARMS[cfg.variant]
    prefix, marker, _ = historical.FUSED_KERNEL.partition(_TAIL_MARK)
    if not marker:
        raise RuntimeError("historical WMMA body lost its epilogue splice marker")
    bias_tile = cfg.threads // 32 * 16 * 16 * 4 if arm["bias"] else 0
    dynamic = max(generated["smem"], bias_tile)
    if dynamic > p1.SMEM_LIMIT:
        raise RuntimeError(f"dynamic shared memory {dynamic} B exceeds {p1.SMEM_LIMIT} B")
    source = (
        generated["kernel_src"]
        + cfc.arm_defines(arm, cfg.N)
        + f"#define DYNAMIC_SMEM_BYTES {dynamic}\n"
        + REGS_DEVICE
        + prefix
        + REGISTER_EPILOGUE
        + (cfc.SOFTMAX_KERNEL if arm["softmax"] else "")
        + WRAPPER
    )
    for forbidden in ("asm(", "asm (", "asm volatile", "__asm"):
        if forbidden in source:
            raise RuntimeError(f"inline asm {forbidden!r} found in cuda_noptx source")
    return {
        "source": source,
        "kernel_source": source.removesuffix(WRAPPER),
        "operand_shared_bytes": generated["smem"],
        "bias_tile_shared_bytes": bias_tile,
        "epilogue_tile_shared_bytes": 0,
        "total_dynamic_shared_bytes": dynamic,
        "geometry": generated,
    }


def build(cfg) -> common2.Built2:
    common.setup_cuda_env()
    generated = make_source(cfg)
    source = generated["source"]
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    name = (
        f"crossed_v2_noptx_regs_{cfg.variant}_{cfg.BM}x{cfg.BN}x{cfg.BK}_"
        f"t{cfg.threads}_kc{cfg.kc}_s{cfg.stages}_{digest[:8]}"
    )
    started = time.perf_counter()
    module = load_inline(
        name=name,
        cpp_sources=cfc.CPP_DECL,
        cuda_sources=source,
        functions=["fused"],
        with_cuda=True,
        verbose=False,
        extra_include_paths=[str(CHECKED_LAUNCH_DIR)],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "-Xptxas=-v",
            "-gencode=arch=compute_89,code=sm_89",
        ],
    )
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

    arm = common2.FUSED_ARMS[cfg.variant]
    geometry = generated["geometry"]
    artifacts = {
        "backend_detail": (
            "no-inline-PTX WMMA; bias uses a same-layout accumulator fragment; "
            "bias/GELU are expressed on the accumulator fragment before global store"
        ),
        "bias_tile_shared_bytes": generated["bias_tile_shared_bytes"],
        "block": geometry["info"]["block"],
        "cuda_source": source,
        "cuda_source_sha256": digest,
        "epilogue": "regs",
        "epilogue_tile_bytes": 0,
        "epilogue_tile_shared_bytes": 0,
        "geom_info": geometry["info"],
        "grid": geometry["info"]["grid"],
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
    compiler = p1._side_compile(generated["kernel_source"], name)
    compiler["kernel_resources"] = ptxas_kernel_resources(
        compiler.get("ptxas_log", ""), "fused_kernel"
    )
    for ambiguous in ("n_regs", "n_spills", "stack_frame_bytes", "static_smem_bytes"):
        compiler.pop(ambiguous, None)
    artifacts.update(compiler)
    notes = (
        f"crossed-v2 measured cuda_noptx source-level register epilogue {cfg.variant}; "
        "physical spills, if any, are recorded in kernel_resources"
    )
    if geometry["deviations"]:
        notes += " | DEVIATION: " + "; ".join(geometry["deviations"])
    return common2.Built2(
        run=run,
        compile_s=time.perf_counter() - started,
        artifacts=artifacts,
        notes=notes,
        n_kernels=artifacts["n_kernels"],
        x_dtype=torch.float16,
    )
