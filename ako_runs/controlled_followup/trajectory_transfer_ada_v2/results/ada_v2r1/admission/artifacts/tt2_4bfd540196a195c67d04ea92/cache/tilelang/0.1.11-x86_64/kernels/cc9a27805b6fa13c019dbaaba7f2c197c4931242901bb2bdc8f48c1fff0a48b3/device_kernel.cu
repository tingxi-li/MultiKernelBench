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
  void* As = ((void*)((char*)buf_dyn_shmem + 65536));
  float Cacc[128];
  float Cchunk[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(Cacc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int c = 0; c < 4; ++c) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(Cchunk + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      tl::cp_async_gs<16>((&(((half_t*)As)[((((i_2 * 2048) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.y) * 1048576) + (i_2 * 524288)) + ((((int)threadIdx.x) >> 2) * 8192)) + (c * 2048)) + ((((int)threadIdx.x) & 3) * 8))])));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(((half_t*)Bs)[((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_3 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((c * 16777216) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 5) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)As)[(((((i_4 * 2048) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[((((((((int)blockIdx.y) * 1048576) + (i_4 * 524288)) + ((((int)threadIdx.x) >> 2) * 8192)) + (c * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)Bs)[(((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_5 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[((((((c * 16777216) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 5) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 262144)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)As)[(((((i_6 * 2048) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A[((((((((int)blockIdx.y) * 1048576) + (i_6 * 524288)) + ((((int)threadIdx.x) >> 2) * 8192)) + (c * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs<16>((&(((half_t*)Bs)[(((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_7 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B[((((((c * 16777216) + (i_7 * 65536)) + ((((int)threadIdx.x) >> 5) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 524288)])));
    }
    tl::cp_async_commit();
    for (int ko = 0; ko < 61; ++ko) {
      __syncthreads();
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        tl::cp_async_gs<16>((&(((half_t*)As)[(((((((ko + 3) & 3) * 4096) + (i_8 * 2048)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.y) * 1048576) + (i_8 * 524288)) + ((((int)threadIdx.x) >> 2) * 8192)) + (c * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 96)])));
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        tl::cp_async_gs<16>((&(((half_t*)Bs)[(((((((((ko + 3) & 3) * 8192) + (((((int)threadIdx.x) & 31) >> 3) * 2048)) + (i_9 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((c * 16777216) + (ko * 262144)) + (i_9 * 65536)) + ((((int)threadIdx.x) >> 5) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 786432)])));
      }
      tl::cp_async_commit();
      tl::cp_async_wait<3>();
      __syncthreads();
      {
        half_t A_local[32];
        half_t B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_10 = 0; i_10 < 4; ++i_10) {
            tl::ptx_ldmatrix_x4((&(((half_t*)As)[(((((((ko & 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_10 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_10 * 8)])));
          }
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((((ko & 3) * 8192) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_11 * 8)])));
          }
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_12 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (((i_12 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      half_t A_local_1[32];
      half_t B_local_1[32];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          tl::ptx_ldmatrix_x4((&(((half_t*)As)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_1[(i_13 * 8)])));
        }
        for (int i_14 = 0; i_14 < 4; ++i_14) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(B_local_1[(i_14 * 8)])));
        }
        for (int i_15 = 0; i_15 < 4; ++i_15) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_15 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (((i_15 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local_2[32];
      half_t B_local_2[32];
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        for (int i_16 = 0; i_16 < 4; ++i_16) {
          tl::ptx_ldmatrix_x4((&(((half_t*)As)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_16 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 8192)])), (&(A_local_2[(i_16 * 8)])));
        }
        for (int i_17 = 0; i_17 < 4; ++i_17) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(B_local_2[(i_17 * 8)])));
        }
        for (int i_18 = 0; i_18 < 4; ++i_18) {
          for (int j_2 = 0; j_2 < 4; ++j_2) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_18 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (((i_18 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
          }
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      half_t A_local_3[32];
      half_t B_local_3[32];
      for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
        for (int i_19 = 0; i_19 < 4; ++i_19) {
          tl::ptx_ldmatrix_x4((&(((half_t*)As)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_19 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12288)])), (&(A_local_3[(i_19 * 8)])));
        }
        for (int i_20 = 0; i_20 < 4; ++i_20) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Bs)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_20 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])), (&(B_local_3[(i_20 * 8)])));
        }
        for (int i_21 = 0; i_21 < 4; ++i_21) {
          for (int j_3 = 0; j_3 < 4; ++j_3) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + ((i_21 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(Cchunk + (((i_21 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
          }
        }
      }
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 128; ++i_22) {
      Cacc[i_22] = (Cacc[i_22] + Cchunk[i_22]);
    }
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 64; ++i_23) {
    for (int vec_s = 0; vec_s < 2; ++vec_s) {
      float v = (Cacc[((i_23 * 2) + vec_s)] + Bias[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_23 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s)]);
      Cacc[((i_23 * 2) + vec_s)] = ((v * 0x1p-1f/*5.000000e-01*/) * (0x1p+0f/*1.000000e+00*/ + erff((v * 0x1.6a09e667f3bcdp-1f/*7.071068e-01*/))));
    }
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 64; ++i_24) {
    *(float2*)(C + (((((((((((int)blockIdx.y) * 1048576) + (((((int)threadIdx.x) & 63) >> 5) * 524288)) + ((i_24 >> 4) * 131072)) + ((i_24 & 1) * 65536)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_24 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(Cacc + (i_24 * 2));
  }
}

