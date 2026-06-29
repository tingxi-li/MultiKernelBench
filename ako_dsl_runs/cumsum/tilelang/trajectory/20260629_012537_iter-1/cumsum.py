import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[1])
def _build(Rr, N, C=4096, TH=256, dtype="float32"):
    @T.prim_func
    def main(X: T.Tensor((Rr, N), dtype), Y: T.Tensor((Rr, N), dtype)):
        with T.Kernel(Rr, threads=TH) as br:
            buf = T.alloc_shared((C,), dtype)
            carry = T.alloc_shared((1,), dtype)
            carry[0] = T.Cast(dtype, 0)
            T.sync_threads()
            for c0 in T.serial(0, N, C):
                T.copy(X[br, c0:c0 + C], buf)
                T.sync_threads()
                T.cumsum(buf, dim=0)          # inclusive prefix sum of the chunk
                T.sync_threads()
                base = carry[0]
                for i in T.Parallel(C):
                    Y[br, c0 + i] = buf[i] + base
                T.sync_threads()
                carry[0] = base + buf[C - 1]   # all threads write identical chunk total
                T.sync_threads()
    return main


_B = (_build,)
_CACHE = {}


class Model(nn.Module):
    """Row cumsum (dim=1) via a TileLang chunked scan-with-carry."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        x = x.contiguous()
        Rr = x.shape[0]; N = x.shape[1]
        key = (Rr, N)
        k = _CACHE.get(key)
        if k is None:
            k = _B[0](Rr, N)
            _CACHE[key] = k
        return k(x)
