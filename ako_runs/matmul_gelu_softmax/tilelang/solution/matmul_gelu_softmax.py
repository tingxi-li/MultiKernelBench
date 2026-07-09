import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# matmul_gelu_softmax: x(1024,8192) -> Linear(8192,8192) -> GELU -> softmax(dim=1)
#
# Iter 5: Store weight as W (N,K) fp16 (no transpose copy at runtime).
# The nn.Linear weight is already (N, K) = (8192, 8192) in row-major.
# Use T.gemm(As, Bs, Acc, transpose_B=True) to compute x @ W^T in-place.
# This avoids the contiguous() copy needed for .t().
# Also tune KC=1024 for more SM wave overlap.

_BM    = 128
_BN    = 128
_BK    = 64
_KC    = 1024   # smaller chunks -> more CTAs -> better SM utilization
_STAGES = 2
_GEMM_TH = 256

_SOFT_TH  = 256
_SOFT_EPT = 8192 // 256

sqrt2inv = 0.7071067811865476


def _build_splitk_gemm_gelu_transB(M, N, K, BM, BN, BK, KC, stages, th):
    """GEMM with W stored as (N, K) using transpose_B=True."""
    NC = K // KC

    @tilelang.jit(out_idx=[-1])
    def _make():
        @T.prim_func
        def main(A:    T.Tensor((M, K), "float16"),
                 W:    T.Tensor((N, K), "float16"),   # (N, K) stored row-major
                 Bias: T.Tensor((N,),   "float32"),
                 Out:  T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=th) as (bx, by):
                As     = T.alloc_shared((BM, BK), "float16")
                Bs     = T.alloc_shared((BN, BK), "float16")   # (BN, BK) for transposed B
                Cchunk = T.alloc_fragment((BM, BN), "float32")
                Cacc   = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)
                for kc in T.serial(NC):
                    T.clear(Cchunk)
                    for ko in T.Pipelined(KC // BK, num_stages=stages):
                        T.copy(A[by * BM, kc * KC + ko * BK], As)
                        T.copy(W[bx * BN, kc * KC + ko * BK], Bs)
                        T.gemm(As, Bs, Cchunk, transpose_B=True)
                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] += Cchunk[i, j]
                for i, j in T.Parallel(BM, BN):
                    val = Cacc[i, j] + Bias[bx * BN + j]
                    Out[by * BM + i, bx * BN + j] = val * T.float32(0.5) * (
                        T.float32(1.0) + T.erf(val * T.float32(sqrt2inv)))
        return main
    return _make()


def _build_softmax(M, N, th):
    ept     = N // th
    nwarps  = th // 32
    nlevels = nwarps.bit_length() - 1

    @tilelang.jit(out_idx=[-1])
    def _make():
        @T.prim_func
        def main(X:   T.Tensor((M, N), "float32"),
                 Out: T.Tensor((M, N), "float32")):
            with T.Kernel(M, threads=th) as bx:
                tid    = T.get_thread_binding(0)
                wid    = tid >> 5
                lane   = tid & 31
                smem_m = T.alloc_shared((nwarps,), "float32")
                smem_s = T.alloc_shared((nwarps,), "float32")
                lmax   = T.alloc_local((1,), "float32")
                lsum   = T.alloc_local((1,), "float32")

                lmax[0] = T.float32(-3.402823466e+38)
                for k in T.serial(ept):
                    v = X[bx, tid * ept + k]
                    if v > lmax[0]:
                        lmax[0] = v
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 16))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 8))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 4))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 2))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 1))
                if lane == 0:
                    smem_m[wid] = lmax[0]
                T.sync_threads()
                for _lvl in range(nlevels):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_m[tid] = T.max(smem_m[tid], smem_m[tid + stride])
                    T.sync_threads()
                row_max = smem_m[0]

                lsum[0] = T.float32(0.0)
                for k in T.serial(ept):
                    lsum[0] = lsum[0] + T.exp(X[bx, tid * ept + k] - row_max)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 16)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 8)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 4)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 2)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 1)
                if lane == 0:
                    smem_s[wid] = lsum[0]
                T.sync_threads()
                for _lvl in range(nlevels):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_s[tid] = smem_s[tid] + smem_s[tid + stride]
                    T.sync_threads()
                inv_sum = T.float32(1.0) / smem_s[0]

                for k in T.serial(ept):
                    Out[bx, tid * ept + k] = T.exp(X[bx, tid * ept + k] - row_max) * inv_sum
        return main
    return _make()


_GG = (_build_splitk_gemm_gelu_transB,)
_SS = (_build_softmax,)
_CACHE: dict = {}


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
        self._kg  = None
        self._ks  = None
        self._wh  = None   # cached fp16 weight W (N, K) -- no transpose needed

    def forward(self, x):
        M, K = x.shape[0], x.shape[1]
        N    = self.linear.weight.shape[0]
        if self._kg is None:
            self._kg, self._ks = _get(M, N, K)
        # Cache W.half() in (N, K) layout — no transpose needed
        if self._wh is None:
            self._wh = self.linear.weight.half()   # (N, K)
        xh      = x.half()
        scratch = self._kg(xh, self._wh, self.linear.bias)
        return self._ks(scratch)
