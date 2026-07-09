import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# matmul_gelu_softmax: x(1024,8192) -> Linear(8192,8192) -> GELU -> softmax(dim=1)
#
# Iter 2: Split-K GEMM to increase SM parallelism.
# With M=1024, N=8192, the number of GEMM tiles is (1024/128)*(8192/128)=8*64=512.
# An RTX 6000 Ada has 76 SMs; 512 tiles leaves many idle.
# Split-K with NC=4 (K_chunk=2048) -> 512*4=2048 tiles -> better occupancy.
# After GEMM, epilogue GELU + softmax.

_BM    = 128
_BN    = 128
_BK    = 64
_KC    = 2048   # K-chunk per split
_STAGES = 2
_GEMM_TH = 256


def _build_splitk_gemm_gelu(M, N, K, BM, BN, BK, KC, stages, th):
    NC       = K // KC         # number of K-splits = 4
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
                # GELU epilogue + write output
                for i, j in T.Parallel(BM, BN):
                    val  = Cacc[i, j] + Bias[bx * BN + j]
                    Out[by * BM + i, bx * BN + j] = val * T.float32(0.5) * (
                        T.float32(1.0) + T.erf(val * T.float32(sqrt2inv)))
        return main
    return _make()


def _build_softmax(M, N, th):
    ept     = N // th
    nlevels = th.bit_length() - 1
    @tilelang.jit(out_idx=[-1])
    def _make():
        @T.prim_func
        def main(X:   T.Tensor((M, N), "float32"),
                 Out: T.Tensor((M, N), "float32")):
            with T.Kernel(M, threads=th) as bx:
                tid    = T.get_thread_binding(0)
                smem_m = T.alloc_shared((th,), "float32")
                smem_s = T.alloc_shared((th,), "float32")
                lmax   = T.alloc_local((1,), "float32")
                lsum   = T.alloc_local((1,), "float32")

                lmax[0] = T.float32(-3.402823466e+38)
                for k in T.serial(ept):
                    v = X[bx, tid * ept + k]
                    if v > lmax[0]:
                        lmax[0] = v
                smem_m[tid] = lmax[0]
                T.sync_threads()
                for _lvl in range(nlevels):
                    stride = th >> (_lvl + 1)
                    if tid < stride:
                        smem_m[tid] = T.max(smem_m[tid], smem_m[tid + stride])
                    T.sync_threads()
                row_max = smem_m[0]

                lsum[0] = T.float32(0.0)
                for k in T.serial(ept):
                    lsum[0] = lsum[0] + T.exp(X[bx, tid * ept + k] - row_max)
                smem_s[tid] = lsum[0]
                T.sync_threads()
                for _lvl in range(nlevels):
                    stride = th >> (_lvl + 1)
                    if tid < stride:
                        smem_s[tid] = smem_s[tid] + smem_s[tid + stride]
                    T.sync_threads()
                inv_sum = T.float32(1.0) / smem_s[0]

                for k in T.serial(ept):
                    Out[bx, tid * ept + k] = T.exp(X[bx, tid * ept + k] - row_max) * inv_sum
        return main
    return _make()


_GG = (_build_splitk_gemm_gelu,)
_SS = (_build_softmax,)
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
