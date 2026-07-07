import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# P2 LEVER TEST — tilelang gather, shared-memory x-row staging.
#
# The committed tilelang gather does OUT[r,c] = X[r, IDX[r,c]] directly: every
# gather is a RANDOM global load into the 8192-wide x-row, uncoalesced. The two
# winners (triton/unlimited, 1.50x vs tilelang 1.31x) stage the x-row once (a
# coalesced read) and gather from fast on-chip memory instead. Here: cooperatively
# load X[r,:] (8192 f32 = 32 KB, fits shared) into shared with a coalesced pass,
# sync, then gather OUT[r,c] = Xs[IDX[r,c]] from shared. The random access now
# hits shared (single-cycle-ish, bank-conflict at worst) not HBM.
# forward() stays glue-only.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[2])
def _build(M, Cin, Cout, TH=256):
    @T.prim_func
    def main(X: T.Tensor((M, Cin), "float32"), IDX: T.Tensor((M, Cout), "int64"),
             OUT: T.Tensor((M, Cout), "float32")):
        with T.Kernel(M, threads=TH) as r:
            Xs = T.alloc_shared((Cin,), "float32")
            # stage the whole x-row into shared (coalesced, vectorized copy)
            for j in T.Parallel(Cin):
                Xs[j] = X[r, j]
            T.sync_threads()
            # gather from shared: random index now hits on-chip memory, not HBM
            for c in T.Parallel(Cout):
                OUT[r, c] = Xs[IDX[r, c]]
    return main


_B = (_build,)
_CACHE = {}


class Model(nn.Module):
    """gather(x, dim=1, index=idx) via TileLang with shared-memory x-row staging:
    stage X[r,:] into shared once (coalesced), then gather from shared."""
    def forward(self, x, idx):
        x = x.contiguous()
        idx = idx.contiguous()
        M = x.shape[0]; Cin = x.shape[1]; Cout = idx.shape[1]
        key = (M, Cin, Cout)
        k = _CACHE.get(key)
        if k is None:
            k = _B[0](M, Cin, Cout)
            _CACHE[key] = k
        return k(x, idx)
