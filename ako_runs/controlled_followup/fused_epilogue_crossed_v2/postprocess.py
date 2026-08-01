#!/usr/bin/env python3
"""Checked common CUDA postprocess with compile-time bias/GELU switches."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CHECKED_LAUNCH_DIR = Path(__file__).resolve().parents[1] / "legacy_cuda_harness_fix"

KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "checked_cuda_launch.h"
#include <cmath>

#define PP_M 1024
#define PP_N 8192
#define PP_THREADS 256
#define PP_EPT (PP_N / PP_THREADS)
#define PP_WARPS (PP_THREADS / 32)

__device__ __forceinline__ float exact_gelu(float value) {
    return value * 0.5f * (1.0f + erff(value * 0.70710678118654752440f));
}

__global__ __launch_bounds__(PP_THREADS)
void common_postprocess_kernel(const float* __restrict__ input,
                               const float* __restrict__ bias,
                               float* __restrict__ output) {
    __shared__ float warp_max[PP_WARPS];
    __shared__ float warp_sum[PP_WARPS];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* row_input = input + (size_t)row * PP_N;
    float* row_output = output + (size_t)row * PP_N;
    float local[PP_EPT];
    float maximum = -3.402823466e+38f;
#pragma unroll
    for (int index = 0; index < PP_EPT; ++index) {
        const int column = tid * PP_EPT + index;
        float value = row_input[column];
#if HAS_BIAS
        value += bias[column];
#endif
#if HAS_GELU
        value = exact_gelu(value);
#endif
        local[index] = value;
        maximum = fmaxf(maximum, value);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    if (lane == 0) warp_max[warp] = maximum;
    __syncthreads();
#pragma unroll
    for (int span = PP_WARPS >> 1; span > 0; span >>= 1) {
        if (tid < span) warp_max[tid] = fmaxf(warp_max[tid], warp_max[tid + span]);
        __syncthreads();
    }
    const float row_maximum = warp_max[0];
    float sum = 0.0f;
#pragma unroll
    for (int index = 0; index < PP_EPT; ++index) {
        const float exponential = __expf(local[index] - row_maximum);
        local[index] = exponential;
        sum += exponential;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
#pragma unroll
    for (int span = PP_WARPS >> 1; span > 0; span >>= 1) {
        if (tid < span) warp_sum[tid] += warp_sum[tid + span];
        __syncthreads();
    }
    const float inverse = 1.0f / warp_sum[0];
#pragma unroll
    for (int index = 0; index < PP_EPT; ++index)
        row_output[tid * PP_EPT + index] = local[index] * inverse;
}

torch::Tensor common_postprocess(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(input.get_device() == bias.get_device(), "same CUDA device required");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 &&
                bias.scalar_type() == torch::kFloat32, "fp32 input and bias required");
    TORCH_CHECK(input.dim() == 2 && input.size(0) == PP_M && input.size(1) == PP_N,
                "input must be (1024,8192)");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == PP_N, "bias must be (8192,)");
    auto output = torch::empty_like(input);
    auto stream = at::cuda::getCurrentCUDAStream();
    common_postprocess_kernel<<<PP_M, PP_THREADS, 0, stream>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>());
    checked_kernel_launch("common_postprocess_kernel");
    return output;
}
"""

CPP = "#include <torch/extension.h>\ntorch::Tensor common_postprocess(torch::Tensor input, torch::Tensor bias);"


def make_source(*, has_bias: bool, has_gelu: bool) -> str:
    if type(has_bias) is not bool or type(has_gelu) is not bool:
        raise TypeError("has_bias and has_gelu must be explicit bool values")
    return f"#define HAS_BIAS {int(has_bias)}\n#define HAS_GELU {int(has_gelu)}\n" + KERNEL_SOURCE


def source_metadata(*, has_bias: bool, has_gelu: bool) -> dict[str, Any]:
    source = make_source(has_bias=has_bias, has_gelu=has_gelu)
    return {
        "cuda_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "has_bias": has_bias,
        "has_gelu": has_gelu,
        "softmax_source_sha256": hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest(),
    }


@dataclass
class BuiltPostprocess:
    run: Callable[[Any, Any], Any]
    compile_s: float
    artifacts: dict[str, Any]


def build(*, has_bias: bool, has_gelu: bool) -> BuiltPostprocess:
    source = make_source(has_bias=has_bias, has_gelu=has_gelu)
    metadata = source_metadata(has_bias=has_bias, has_gelu=has_gelu)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
    cuda_bin = "/usr/local/cuda-13.1/bin"
    if cuda_bin not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
    from torch.utils.cpp_extension import load_inline

    digest = metadata["cuda_source_sha256"]
    started = time.perf_counter()
    module = load_inline(
        name=f"fused_crossed_v2_pp_{digest[:12]}",
        cpp_sources=CPP,
        cuda_sources=source,
        functions=["common_postprocess"],
        with_cuda=True,
        verbose=False,
        extra_include_paths=[str(CHECKED_LAUNCH_DIR)],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "-Xptxas=-v",
            "-gencode=arch=compute_89,code=sm_89",
        ],
    )
    return BuiltPostprocess(
        run=module.common_postprocess,
        compile_s=time.perf_counter() - started,
        artifacts={
            "backend_detail": "common CUDA compile-time bias/GELU plus row-softmax",
            "block": [256, 1, 1],
            "cuda_source_sha256": digest,
            "grid": [1024, 1, 1],
            "has_bias": metadata["has_bias"],
            "has_gelu": metadata["has_gelu"],
            "n_kernels": 1,
            "operand_smem_bytes": 0,
            "epilogue_tile_bytes": 0,
            "shared_bytes": 0,
            "softmax_source_sha256": metadata["softmax_source_sha256"],
            "total_dynamic_shared_bytes": 0,
        },
    )
