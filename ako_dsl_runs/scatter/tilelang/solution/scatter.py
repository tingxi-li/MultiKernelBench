import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


@tilelang.jit
def _build_pass1(Rr, K, W, TH=256):
    @T.prim_func
    def main(IDX: T.Tensor((Rr, K), "int32"), WIN: T.Tensor((Rr, W), "int32")):
        with T.Kernel(Rr, threads=TH) as r:
            for k in T.Parallel(K):
                T.atomic_max(WIN[r, IDX[r, k]], k)
    return main


@tilelang.jit(out_idx=[3])
def _build_pass2(Rr, K, W, TH=256):
    @T.prim_func
    def main(X: T.Tensor((Rr, W), "float32"), WIN: T.Tensor((Rr, W), "int32"),
             UPD: T.Tensor((Rr, K), "float32"), OUT: T.Tensor((Rr, W), "float32")):
        with T.Kernel(Rr, threads=TH) as r:
            for c in T.Parallel(W):
                wk = WIN[r, c]
                OUT[r, c] = T.if_then_else(wk >= 0, UPD[r, T.max(wk, 0)], X[r, c])
    return main


_B1 = (_build_pass1,)
_B2 = (_build_pass2,)
_CACHE = {}


class Model(nn.Module):
    """Deterministic last-wins scatter (dim=1) via TileLang atomicMax two-pass."""
    def forward(self, x, idx, updates):
        x = x.contiguous()
        idx = idx.contiguous().to(torch.int32)
        updates = updates.contiguous()
        Rr = x.shape[0]; W = x.shape[1]; K = idx.shape[1]
        win = torch.full(x.shape, -1, device=x.device, dtype=torch.int32)
        key = (Rr, K, W)
        kk = _CACHE.get(key)
        if kk is None:
            kk = (_B1[0](Rr, K, W), _B2[0](Rr, K, W))
            _CACHE[key] = kk
        kk[0](idx, win)
        return kk[1](x, win, updates)
