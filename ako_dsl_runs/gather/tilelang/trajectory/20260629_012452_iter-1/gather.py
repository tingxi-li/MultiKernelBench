import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2])
def _build(M, Cin, Cout, TH=256):
    @T.prim_func
    def main(X: T.Tensor((M, Cin), "float32"), IDX: T.Tensor((M, Cout), "int32"),
             OUT: T.Tensor((M, Cout), "float32")):
        with T.Kernel(M, threads=TH) as r:
            for c in T.Parallel(Cout):
                OUT[r, c] = X[r, IDX[r, c]]
    return main


_B = (_build,)
_CACHE = {}


class Model(nn.Module):
    """gather(x, dim=1, index=idx) via a TileLang indexed-load kernel."""
    def forward(self, x, idx):
        x = x.contiguous()
        idx = idx.contiguous().to(torch.int32)
        M = x.shape[0]; Cin = x.shape[1]; Cout = idx.shape[1]
        key = (M, Cin, Cout)
        k = _CACHE.get(key)
        if k is None:
            k = _B[0](M, Cin, Cout)
            _CACHE[key] = k
        return k(x, idx)
