import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# matmul_gelu_softmax: x(1024,8192) -> Linear(8192,8192) -> GELU -> softmax(dim=1)
#
# Iter 3: Warp-reduce softmax to avoid shared-memory tree overhead.
# Use T.warp_reduce_sum/max (intra-warp shuffle) for the softmax reductions.
# 32-thread warp handles 8192/32=256 elements per thread -> need inter-warp reduction.
# Use a hybrid: warp shuffle + minimal shared memory for inter-warp combine.
# Also try tanh-GELU approximation (fewer instructions than erf):
#   GELU(x) ≈ 0.5*x*(1 + tanh(0.797884560802*(x + 0.044715*x^3)))

_BM    = 128
_BN    = 128
_BK    = 64
_KC    = 2048
_STAGES = 2
_GEMM_TH = 256

_SOFT_TH  = 256   # 8 warps per block
_SOFT_EPT = 8192 // 256  # = 32 elements/thread


def _build_splitk_gemm_gelu(M, N, K, BM, BN, BK, KC, stages, th):
    NC = K // KC
    sqrt2inv = 0.7071067811865476

    @tilelang.jit(out_idx=[-1])
    def _make():
        @T.prim_func
        def main(A:    T.Tensor((M, K), "float16"),
                 WT:   T.Tensor((K, N), "float16"),
                 Bias: T.Tensor((N,),   "float32"),
                 Out:  T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=th) as (bx, by):
                As     = T.alloc_shared((BM, BK), "float16")
                Bs     = T.alloc_shared((BK, BN), "float16")
                Cchunk = T.alloc_fragment((BM, BN), "float32")
                Cacc   = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)
                for kc in T.serial(NC):
                    T.clear(Cchunk)
                    for ko in T.Pipelined(KC // BK, num_stages=stages):
                        T.copy(A[by * BM, kc * KC + ko * BK], As)
                        T.copy(WT[kc * KC + ko * BK, bx * BN], Bs)
                        T.gemm(As, Bs, Cchunk)
                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] += Cchunk[i, j]
                # erf-GELU epilogue (exact, passes 1e-4 gate)
                for i, j in T.Parallel(BM, BN):
                    val = Cacc[i, j] + Bias[bx * BN + j]
                    Out[by * BM + i, bx * BN + j] = val * T.float32(0.5) * (
                        T.float32(1.0) + T.erf(val * T.float32(sqrt2inv)))
        return main
    return _make()


def _build_softmax_warp(M, N, th):
    """
    Row-wise softmax using warp shuffles for intra-warp reduce
    + a tiny shared-mem step for inter-warp reduce.
    th=256 -> 8 warps, each warp handles N/th*32 = 32*32=1024 elements.
    """
    ept     = N // th    # = 32
    nwarps  = th // 32   # = 8
    nlevels = nwarps.bit_length() - 1  # log2(8)=3

    @tilelang.jit(out_idx=[-1])
    def _make():
        @T.prim_func
        def main(X:   T.Tensor((M, N), "float32"),
                 Out: T.Tensor((M, N), "float32")):
            with T.Kernel(M, threads=th) as bx:
                tid     = T.get_thread_binding(0)
                wid     = tid >> 5      # warp id
                lane    = tid & 31      # lane id

                smem_m  = T.alloc_shared((nwarps,), "float32")
                smem_s  = T.alloc_shared((nwarps,), "float32")
                lmax    = T.alloc_local((1,), "float32")
                lsum    = T.alloc_local((1,), "float32")

                # ---- phase 1: thread-local max --------------------------------
                lmax[0] = T.float32(-3.402823466e+38)
                for k in T.serial(ept):
                    v = X[bx, tid * ept + k]
                    if v > lmax[0]:
                        lmax[0] = v

                # intra-warp reduce max via shfl_down(value, delta)
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 16))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 8))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 4))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 2))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 1))

                # lane 0 writes warp result to shared
                if lane == 0:
                    smem_m[wid] = lmax[0]
                T.sync_threads()

                # inter-warp tree reduce (in thread 0)
                if tid < nwarps:
                    lmax[0] = smem_m[tid]
                for _lvl in range(nlevels):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_m[tid] = T.max(smem_m[tid], smem_m[tid + stride])
                    T.sync_threads()
                row_max = smem_m[0]

                # ---- phase 2: thread-local exp sum ----------------------------
                lsum[0] = T.float32(0.0)
                for k in T.serial(ept):
                    lsum[0] = lsum[0] + T.exp(X[bx, tid * ept + k] - row_max)

                # intra-warp reduce sum
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 16)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 8)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 4)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 2)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 1)

                if lane == 0:
                    smem_s[wid] = lsum[0]
                T.sync_threads()

                if tid < nwarps:
                    lsum[0] = smem_s[tid]
                for _lvl in range(nlevels):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_s[tid] = smem_s[tid] + smem_s[tid + stride]
                    T.sync_threads()
                inv_sum = T.float32(1.0) / smem_s[0]

                # ---- phase 3: write output ------------------------------------
                for k in T.serial(ept):
                    Out[bx, tid * ept + k] = T.exp(X[bx, tid * ept + k] - row_max) * inv_sum
        return main
    return _make()


_GG = (_build_splitk_gemm_gelu,)
_SS = (_build_softmax_warp,)
_CACHE: dict = {}

_SOFT_TH = 256


def _get(M, N, K):
    key = (M, N, K)
    if key not in _CACHE:
        kg = _GG[0](M, N, K, _BM, _BN, _BK, _KC, _STAGES, _GEMM_TH)
        ks = _SS[0](M, N, _SOFT_TH)
        _CACHE[key] = (kg, ks)
    return _CACHE[key]


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._kg = None
        self._ks = None

    def forward(self, x):
        M, K = x.shape[0], x.shape[1]
        N    = self.linear.weight.shape[0]
        if self._kg is None:
            self._kg, self._ks = _get(M, N, K)
        xh = x.half()
        wt = self.linear.weight.t().contiguous().half()
        scratch = self._kg(xh, wt, self.linear.bias)
        return self._ks(scratch)
