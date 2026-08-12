#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, const float* __restrict__ Bias, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, const float* __restrict__ Bias, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Bs = ((void*)((char*)buf_dyn_shmem + 0));
  void* As = ((void*)((char*)buf_dyn_shmem + 32768));
  float Cacc[32];
  float Cchunk[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(Cacc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int c = 0; c < 4; ++c) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(Cchunk + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      tl::cp_async_gs<16>((&(((half_t*)As)[(((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.y) * 524288) + (i_2 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (c * 2048)) + ((((int)threadIdx.x) & 7) * 8))])));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(((half_t*)Bs)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((c * 16777216) + (i_3 * 131072)) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
    }
    tl::cp_async_commit();
    for (int ko = 0; ko < 31; ++ko) {
      __syncthreads();
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        tl::cp_async_gs<16>((&(((half_t*)As)[((((((((ko + 1) & 1) * 4096) + (i_4 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.y) * 524288) + (i_4 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (c * 2048)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        tl::cp_async_gs<16>((&(((half_t*)Bs)[(((((((((ko + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((c * 16777216) + (ko * 524288)) + (i_5 * 131072)) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 524288)])));
      }
      tl::cp_async_commit();
      tl::cp_async_wait<1>();
      __syncthreads();
      {
        half_t A_local[16];
        half_t B_local[16];
        for (int ki = 0; ki < 4; ++ki) {
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            tl::ptx_ldmatrix_x4((&(((half_t*)As)[((((((ko & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_6 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_6 * 8)])));
          }
          for (int i_7 = 0; i_7 < 2; ++i_7) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((((ko & 1) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_7) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_7 * 8)])));
          }
          for (int i_8 = 0; i_8 < 2; ++i_8) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_8 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (((i_8 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      half_t A_local_1[16];
      half_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)As)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_9 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4096)])), (&(A_local_1[(i_9 * 8)])));
        }
        for (int i_10 = 0; i_10 < 2; ++i_10) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((((((int)threadIdx.x) >> 7) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_10) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(B_local_1[(i_10 * 8)])));
        }
        for (int i_11 = 0; i_11 < 2; ++i_11) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_11 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (((i_11 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      Cacc[i_12] = (Cacc[i_12] + Cchunk[i_12]);
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 16; ++i_13) {
    for (int vec_s = 0; vec_s < 2; ++vec_s) {
      float v = (Cacc[((i_13 * 2) + vec_s)] + Bias[(((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_13 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s)]);
      Cacc[((i_13 * 2) + vec_s)] = ((v * 0x1p-1f/*5.000000e-01*/) * (0x1p+0f/*1.000000e+00*/ + erff((v * 0x1.6a09e667f3bcdp-1f/*7.071068e-01*/))));
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 16; ++i_14) {
    *(float2*)(C + (((((((((((int)blockIdx.y) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_14 >> 3) * 131072)) + ((i_14 & 1) * 65536)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_14 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(Cacc + (i_14 * 2));
  }
}

