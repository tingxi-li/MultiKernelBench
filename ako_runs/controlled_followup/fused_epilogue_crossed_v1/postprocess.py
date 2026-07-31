#!/usr/bin/env python3
"""One checked common CUDA postprocess for the global-intermediate strategy."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Callable


SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
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
void combined_postprocess_kernel(const float* __restrict__ input,
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
        const float value = exact_gelu(row_input[column] + bias[column]);
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

torch::Tensor combined_postprocess(torch::Tensor input, torch::Tensor bias) {
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
    combined_postprocess_kernel<<<PP_M, PP_THREADS, 0, stream>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

CPP = "#include <torch/extension.h>\ntorch::Tensor combined_postprocess(torch::Tensor input, torch::Tensor bias);"


@dataclass
class BuiltPostprocess:
    run: Callable[[Any, Any], Any]
    compile_s: float
    artifacts: dict[str, Any]


def build() -> BuiltPostprocess:
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
    cuda_bin = "/usr/local/cuda-13.1/bin"
    if cuda_bin not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
    from torch.utils.cpp_extension import load_inline

    digest = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    started = time.perf_counter()
    module = load_inline(
        name=f"fused_crossed_pp_{digest[:12]}",
        cpp_sources=CPP,
        cuda_sources=SOURCE,
        functions=["combined_postprocess"],
        with_cuda=True,
        verbose=False,
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "-Xptxas=-v",
            "-gencode=arch=compute_89,code=sm_89",
        ],
    )
    compile_s = time.perf_counter() - started
    return BuiltPostprocess(
        run=module.combined_postprocess,
        compile_s=compile_s,
        artifacts={
            "backend_detail": "common CUDA combined bias + exact-erf GELU + row-softmax kernel",
            "block": [256, 1, 1],
            "cuda_source_sha256": digest,
            "grid": [1024, 1, 1],
            "n_kernels": 1,
        },
    )

