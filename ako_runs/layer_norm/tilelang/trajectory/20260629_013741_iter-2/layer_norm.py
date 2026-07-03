import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[3])
def _build(M, N, eps, TH=256, dtype="float32"):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Wt: T.Tensor((N,), dtype),
             Bs: T.Tensor((N,), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(M, threads=TH) as m:
            red = T.alloc_shared((2,), "float64")
            red[0] = T.Cast("float64", 0)
            red[1] = T.Cast("float64", 0)
            T.sync_threads()
            tid = T.get_thread_binding(0)
            ls = T.alloc_local((1,), "float64")
            lq = T.alloc_local((1,), "float64")
            ls[0] = T.Cast("float64", 0)
            lq[0] = T.Cast("float64", 0)
            for j in T.serial(tid, N, TH):
                v = X[m, j]
                ls[0] += v
                lq[0] += v * v
            T.atomic_add(red[0], ls[0])
            T.atomic_add(red[1], lq[0])
            T.sync_threads()
            mean = red[0] / N
            rstd = T.rsqrt(red[1] / N - mean * mean + eps)
            for j in T.serial(tid, N, TH):
                Y[m, j] = (X[m, j] - mean) * rstd * Wt[j] + Bs[j]
    return main


_B = (_build,)
_CACHE = {}


class Model(nn.Module):
    """LayerNorm via a TileLang per-row reduce (block/row, atomic to shared) + affine.
    self.ln is a parameter container only (weight/bias/eps); never called."""
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
    def forward(self, x):
        x = x.contiguous()
        w = self.ln.weight.contiguous()
        b = self.ln.bias.contiguous()
        M = x.shape[0]
        N = w.numel()
        x2 = x.reshape(M, N)
        key = (M, N)
        k = _CACHE.get(key)
        if k is None:
            k = _B[0](M, N, float(self.ln.eps))
            _CACHE[key] = k
        return k(x2, w, b).reshape(x.shape)
