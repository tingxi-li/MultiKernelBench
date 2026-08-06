#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
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
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  float xs[32];
  float mx[1];
  float sm[1];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(float4*)(xs + (i * 4)) = *(float4*)(X + (((((int)blockIdx.x) * 8192) + (i * 1024)) + (((int)threadIdx.x) * 4)));
  }
  mx[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 32; ++rv) {
    mx[0] = max(mx[0], xs[(((rv & 7) * 4) + (rv >> 3))]);
  }
  mx[0] = tl::AllReduce<tl::MaxOp, 256, 1, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    xs[i_1] = expf((xs[i_1] - mx[0]));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
    sm[0] = (sm[0] + xs[(((rv_1 & 7) * 4) + (rv_1 >> 3))]);
  }
  __syncthreads();
  sm[0] = tl::AllReduce<tl::SumOp, 256, 1, 0>::run(sm[0], (&(((float*)workspace)[0])));
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    float4 __1;
      float4 v_ = *(float4*)(xs + (i_2 * 4));
      float4 v__1 = make_float4(sm[0], sm[0], sm[0], sm[0]);
      __1.x = (v_.x/v__1.x);
      __1.y = (v_.y/v__1.y);
      __1.z = (v_.z/v__1.z);
      __1.w = (v_.w/v__1.w);
    *(float4*)(Out + (((((int)blockIdx.x) * 8192) + (i_2 * 1024)) + (((int)threadIdx.x) * 4))) = __1;
  }
}

