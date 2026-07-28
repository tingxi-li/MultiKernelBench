"""Shared CUDA source for the two hand-written Phase-2 fused lanes.

Both CUDA lanes reuse their Phase-1 kernel body *unmodified* -- the same
`_KERNEL_BODY` string, the same `_make_source` schedule logic (warp grid,
shared-memory padding, stage count, KC flush) -- and this module supplies only
what the fused ladder adds on top: an epilogue kernel and a softmax kernel.

The epilogue is written once, in CUDA C, and both lanes include it. That is
deliberate: the study varies the *inner loop*'s instruction path between the two
lanes (WMMA C++ intrinsics vs inline `mma.sync` PTX), so the epilogue must be
identical or the ladder's increments would confound the two.

Why the fragment goes through shared memory: a WMMA/mma accumulator's mapping
from `frag.x[e]` to (row, col) is not part of either API's contract, so bias --
which is indexed by column -- cannot be applied in registers without hard-coding
an undocumented lane layout. Staging the tile through shared memory and then
walking it with explicit (row, col) indices is what a real fused kernel does, and
it costs one smem round trip that is paid identically by every arm of the ladder
including `G`. TileLang's `T.copy(Cacc, C[...])` and Triton's block `tl.store`
both stage through shared memory too, so this is the matched behaviour, not a
handicap.

`FUSED_EPILOGUE` expects these macros to already be defined by the lane's
Phase-1 const block: BM, BN, THREADS, M_, N_, NMF, NNF, WMT, WNT, WARPS_N,
FRAG_ELEMS. It defines CPAD and FSMEM_BYTES itself.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Applied to the (BM, BN) fp32 tile once it is in shared memory. `HAS_BIAS` and
# `HAS_GELU` are compile-time, so arm G's epilogue really is a plain copy.
EPILOGUE_DEVICE = r"""
/* ---- Phase-2 fused epilogue -------------------------------------------- */
#define CPAD 4                       /* pad the smem tile to break bank conflicts */
#define CSTRIDE (BN + CPAD)
#define FSMEM_TILE_BYTES (BM * CSTRIDE * 4)
#if FSMEM_TILE_BYTES > SMEM_BYTES
#define FSMEM_BYTES FSMEM_TILE_BYTES
#else
#define FSMEM_BYTES SMEM_BYTES
#endif

__device__ __forceinline__ float gelu_exact(float v) {
    /* F.gelu(approximate='none'): v * 0.5 * (1 + erf(v / sqrt(2))).
       erff, not tanhf -- the tanh form is a different function and using it
       would be a precision shortcut hidden inside a "fusion" result. */
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}

/* Walk the staged tile with explicit indices and write it out coalesced. */
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

# ---------------------------------------------------------------------------
# One row per block, THREADS threads, EPT elements per thread held in registers
# so `expf` is evaluated exactly once. Algorithmically identical to the TileLang
# F4 kernel and to the Triton one-tile-per-row kernel: warp-shuffle reduction
# into a shared-memory tree, exponentials cached, two passes over global memory.
SOFTMAX_KERNEL = r"""
/* ---- Phase-2 row softmax ------------------------------------------------ */
#define SOFT_TH 256
#define SOFT_EPT (SOFT_N / SOFT_TH)
#define SOFT_NWARPS (SOFT_TH / 32)

__global__ __launch_bounds__(SOFT_TH)
void softmax_kernel(const float* __restrict__ X, float* __restrict__ Y) {
    __shared__ float sm_m[SOFT_NWARPS];
    __shared__ float sm_s[SOFT_NWARPS];
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    const float* __restrict__ xr = X + (size_t)row * SOFT_N;
    float* __restrict__ yr = Y + (size_t)row * SOFT_N;

    float lexp[SOFT_EPT];
    float m = -3.402823466e+38f;
#pragma unroll
    for (int k = 0; k < SOFT_EPT; ++k) {
        lexp[k] = xr[tid * SOFT_EPT + k];   /* reuse the buffer to hold x first */
        m = fmaxf(m, lexp[k]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) sm_m[wid] = m;
    __syncthreads();
#pragma unroll
    for (int s = SOFT_NWARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_m[tid] = fmaxf(sm_m[tid], sm_m[tid + s]);
        __syncthreads();
    }
    const float row_max = sm_m[0];

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < SOFT_EPT; ++k) {
        const float e = __expf(lexp[k] - row_max);
        lexp[k] = e;
        sum += e;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) sm_s[wid] = sum;
    __syncthreads();
#pragma unroll
    for (int s = SOFT_NWARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_s[tid] = sm_s[tid] + sm_s[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / sm_s[0];

#pragma unroll
    for (int k = 0; k < SOFT_EPT; ++k) yr[tid * SOFT_EPT + k] = lexp[k] * inv;
}
"""

# ---------------------------------------------------------------------------
WRAPPER = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor fused(torch::Tensor A, torch::Tensor B, torch::Tensor Bias) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Bias.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contiguous required");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be fp16");
    TORCH_CHECK(Bias.scalar_type() == torch::kFloat32, "bias must be fp32");
    TORCH_CHECK(A.size(0) == M_ && A.size(1) == K_, "A shape");
    TORCH_CHECK(B.size(0) == K_ && B.size(1) == N_, "B shape");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({M_, N_}, opts);

    static bool attr_done = false;
    if (!attr_done) {
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)fused_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FSMEM_BYTES);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute(", FSMEM_BYTES,
                    ") failed: ", cudaGetErrorString(e));
        attr_done = true;
    }

    dim3 grid(N_ / BN, M_ / BM);
    dim3 block(THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_kernel<<<grid, block, FSMEM_BYTES, stream>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const half*>(B.data_ptr()),
        Bias.data_ptr<float>(), C.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#if HAS_SOFTMAX
    auto Y = torch::empty({M_, N_}, opts);
    softmax_kernel<<<M_, SOFT_TH, 0, stream>>>(C.data_ptr<float>(),
                                               Y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
#else
    return C;
#endif
}
"""

CPP_DECL = ("#include <torch/extension.h>\n"
            "torch::Tensor fused(torch::Tensor A, torch::Tensor B, "
            "torch::Tensor Bias);")


def arm_defines(arm: dict, n: int) -> str:
    """`SOFT_N` is passed explicitly rather than reusing a lane macro: the noptx
    lane compiles N into `N_`, the unlimited lane takes N as a runtime kernel
    argument, and the softmax kernel needs it at compile time to size its
    per-thread register buffer."""
    return ("\n#define HAS_BIAS %d\n#define HAS_GELU %d\n#define HAS_SOFTMAX %d\n"
            "#define SOFT_N %d\n"
            % (int(arm["bias"]), int(arm["gelu"]), int(arm["softmax"]), n))
