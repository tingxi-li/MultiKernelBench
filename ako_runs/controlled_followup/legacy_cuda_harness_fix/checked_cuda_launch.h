#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

// Use these helpers in any post-review CUDA harness.  The historical Phase-2
// wrappers remain receipt-bound; this header provides the fail-closed contract
// for corrected descendants without rewriting those historical sources.
inline void checked_dynamic_smem(const void* kernel, std::size_t bytes,
                                 const char* kernel_name) {
    cudaError_t error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(bytes));
    TORCH_CHECK(error == cudaSuccess, "cudaFuncSetAttribute(", kernel_name,
                ", ", bytes, ") failed: ", cudaGetErrorString(error));
}

inline void checked_kernel_launch(const char* kernel_name) {
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, kernel_name, " launch failed: ",
                cudaGetErrorString(error));
}
