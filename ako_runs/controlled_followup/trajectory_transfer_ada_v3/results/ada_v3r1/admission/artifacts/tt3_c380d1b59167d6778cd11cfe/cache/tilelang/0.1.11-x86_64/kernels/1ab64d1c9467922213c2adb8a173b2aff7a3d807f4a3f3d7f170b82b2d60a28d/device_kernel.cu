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
  void* As = ((void*)((char*)buf_dyn_shmem + 0));
  void* Bs = ((void*)((char*)buf_dyn_shmem + 65536));
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
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      tl::cp_async_gs<16>((&(((half_t*)As)[(((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.y) * 1048576) + (i_2 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (c * 2048)) + ((((int)threadIdx.x) & 7) * 8))])));
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      tl::cp_async_gs<16>((&(((half_t*)Bs)[(((((i_3 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((c * 16777216) + (i_3 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)As)[((((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A[((((((((int)blockIdx.y) * 1048576) + (i_4 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (c * 2048)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)Bs)[((((((i_5 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B[((((((c * 16777216) + (i_5 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)As)[((((((i_6 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(A[((((((((int)blockIdx.y) * 1048576) + (i_6 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (c * 2048)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      tl::cp_async_gs<16>((&(((half_t*)Bs)[((((((i_7 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[((((((c * 16777216) + (i_7 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1048576)])));
    }
    tl::cp_async_commit();
    for (int ko = 0; ko < 29; ++ko) {
      __syncthreads();
      #pragma unroll
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::cp_async_gs<16>((&(((half_t*)As)[((((((((ko + 3) & 3) * 8192) + (i_8 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.y) * 1048576) + (i_8 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (c * 2048)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 192)])));
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        tl::cp_async_gs<16>((&(((half_t*)Bs)[((((((((ko + 3) & 3) * 4096) + (i_9 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((c * 16777216) + (ko * 524288)) + (i_9 * 262144)) + ((((int)threadIdx.x) >> 3) * 8192)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1572864)])));
      }
      tl::cp_async_commit();
      tl::cp_async_wait<3>();
      __syncthreads();
      {
        half_t A_local[32];
        half_t B_local[8];
        for (int ki = 0; ki < 4; ++ki) {
          for (int i_10 = 0; i_10 < 4; ++i_10) {
            tl::ptx_ldmatrix_x4((&(((half_t*)As)[((((((ko & 3) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_10 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_10 * 8)])));
          }
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[(((((ko & 3) * 4096) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[0])));
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (i_11 * 8)), reinterpret_cast<const unsigned*>(A_local + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_11 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local + 4));
          }
        }
      }
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      half_t A_local_1[32];
      half_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int i_12 = 0; i_12 < 4; ++i_12) {
          tl::ptx_ldmatrix_x4((&(((half_t*)As)[(((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_12 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_1[(i_12 * 8)])));
        }
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4096)])), (&(B_local_1[0])));
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (i_13 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_13 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 4));
        }
      }
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local_2[32];
      half_t B_local_2[8];
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        for (int i_14 = 0; i_14 < 4; ++i_14) {
          tl::ptx_ldmatrix_x4((&(((half_t*)As)[(((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_14 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_2[(i_14 * 8)])));
        }
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((ki_2 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(B_local_2[0])));
        for (int i_15 = 0; i_15 < 4; ++i_15) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (i_15 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_15 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 4));
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      half_t A_local_3[32];
      half_t B_local_3[8];
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        for (int i_16 = 0; i_16 < 4; ++i_16) {
          tl::ptx_ldmatrix_x4((&(((half_t*)As)[(((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_16 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])), (&(A_local_3[(i_16 * 8)])));
        }
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((ki_3 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])), (&(B_local_3[0])));
        for (int i_17 = 0; i_17 < 4; ++i_17) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (i_17 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_17 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 4));
        }
      }
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 32; ++i_18) {
      Cacc[i_18] = (Cacc[i_18] + Cchunk[i_18]);
    }
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 16; ++i_19) {
    for (int vec_s = 0; vec_s < 2; ++vec_s) {
      float v = (Cacc[((i_19 * 2) + vec_s)] + Bias[(((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s)]);
      Cacc[((i_19 * 2) + vec_s)] = ((v * 0x1p-1f/*5.000000e-01*/) * (0x1p+0f/*1.000000e+00*/ + erff((v * 0x1.6a09e667f3bcdp-1f/*7.071068e-01*/))));
    }
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 16; ++i_20) {
    *(float2*)(C + (((((((((((int)blockIdx.y) * 1048576) + (((((int)threadIdx.x) & 63) >> 5) * 524288)) + ((i_20 >> 2) * 131072)) + ((i_20 & 1) * 65536)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_20 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(Cacc + (i_20 * 2));
  }
}

