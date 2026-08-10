#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(float* __restrict__ Out, const float* __restrict__ X);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ Out, const float* __restrict__ X) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* smem_m = ((void*)((char*)buf_dyn_shmem + 0));
  void* smem_s = ((void*)((char*)buf_dyn_shmem + 32));
  float lmax[1];
  float lsum[1];
  float lexp[32];
  lmax[0] = -0x1.fffffdff07036p+127f/*-3.402823e+38*/;
  for (int k = 0; k < 32; ++k) {
    float v = X[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 32)) + k)];
    if (lmax[0] < v) {
      lmax[0] = v;
    }
  }
  lmax[0] = max(lmax[0], __shfl_down_sync((uint)4294967295, lmax[0], 16, 32));
  lmax[0] = max(lmax[0], __shfl_down_sync((uint)4294967295, lmax[0], 8, 32));
  lmax[0] = max(lmax[0], __shfl_down_sync((uint)4294967295, lmax[0], 4, 32));
  lmax[0] = max(lmax[0], __shfl_down_sync((uint)4294967295, lmax[0], 2, 32));
  lmax[0] = max(lmax[0], __shfl_down_sync((uint)4294967295, lmax[0], 1, 32));
  if ((((int)threadIdx.x) & 31) == 0) {
    ((float*)smem_m)[(((int)threadIdx.x) >> 5)] = lmax[0];
  }
  __syncthreads();
  for (int _lvl = 0; _lvl < 3; ++_lvl) {
    if (((int)threadIdx.x) < (8 >> (_lvl + 1))) {
      ((float*)smem_m)[((int)threadIdx.x)] = max(((float*)smem_m)[((int)threadIdx.x)], ((float*)smem_m)[(((int)threadIdx.x) + (8 >> (_lvl + 1)))]);
    }
    __syncthreads();
  }
  float row_max = ((float*)smem_m)[0];
  lsum[0] = 0x0p+0f/*0.000000e+00*/;
  for (int k_1 = 0; k_1 < 32; ++k_1) {
    float e = expf((X[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 32)) + k_1)] - row_max));
    lexp[k_1] = e;
    lsum[0] = (lsum[0] + e);
  }
  lsum[0] = (lsum[0] + __shfl_down_sync((uint)4294967295, lsum[0], 16, 32));
  lsum[0] = (lsum[0] + __shfl_down_sync((uint)4294967295, lsum[0], 8, 32));
  lsum[0] = (lsum[0] + __shfl_down_sync((uint)4294967295, lsum[0], 4, 32));
  lsum[0] = (lsum[0] + __shfl_down_sync((uint)4294967295, lsum[0], 2, 32));
  lsum[0] = (lsum[0] + __shfl_down_sync((uint)4294967295, lsum[0], 1, 32));
  if ((((int)threadIdx.x) & 31) == 0) {
    ((float*)smem_s)[(((int)threadIdx.x) >> 5)] = lsum[0];
  }
  __syncthreads();
  for (int _lvl_1 = 0; _lvl_1 < 3; ++_lvl_1) {
    if (((int)threadIdx.x) < (8 >> (_lvl_1 + 1))) {
      ((float*)smem_s)[((int)threadIdx.x)] = (((float*)smem_s)[((int)threadIdx.x)] + ((float*)smem_s)[(((int)threadIdx.x) + (8 >> (_lvl_1 + 1)))]);
    }
    __syncthreads();
  }
  float inv_sum = (0x1p+0f/*1.000000e+00*/ / ((float*)smem_s)[0]);
  for (int k_2 = 0; k_2 < 32; ++k_2) {
    Out[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 32)) + k_2)] = (lexp[k_2] * inv_sum);
  }
}

