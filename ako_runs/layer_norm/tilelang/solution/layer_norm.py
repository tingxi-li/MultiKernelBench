import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[3])
def _build(M, N, eps, G=780, TH=64, dtype="float32"):
    # PERSISTENT single-read LayerNorm.
    # The row is huge (N ~ 4.2M) so it cannot be cached in shared memory: a naive
    # per-row block must read X twice (stats pass + apply pass) -> 3*N DRAM traffic
    # (2 reads of X + 1 write of Y), which is the 3.98ms / 1.60x memory roofline.
    # Here ONE cooperative grid processes ONE row at a time: all G blocks stream
    # row m for the mean/var reduction (fills L2 with just 16MB, which fits the
    # 96MB AD102 L2), grid-sync, then all blocks RE-READ row m -- now an L2 hit --
    # and write Y. So X is read from DRAM only ONCE: traffic drops 3*N -> 2*N.
    # A grid-sync after apply keeps X[m] resident (stops the next row's stats
    # reads from evicting it mid-apply). Per-row accumulator slots Acc[M,2] are
    # zeroed once up front so no per-row reset barrier is needed; grid-sync
    # lockstep guarantees no block runs a row ahead, so slot m is race-free.
    NT = G * TH

    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Wt: T.Tensor((N,), dtype),
             Bs: T.Tensor((N,), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(G, threads=TH) as bx:
            tid = T.get_thread_binding(0)
            # cross-block reduction scratch (one sum/sumsq pair per row)
            Acc = T.alloc_global((M, 2), "float32")
            red = T.alloc_shared((2,), "float32")
            ls = T.alloc_local((1,), "float32")
            lq = T.alloc_local((1,), "float32")
            gtid = bx * TH + tid
            # zero all slots once (cudaMalloc'd scratch is uninitialised)
            for s in T.serial(gtid, M * 2, NT):
                Acc[s // 2, s % 2] = T.Cast("float32", 0)
            T.sync_grid()
            for m in T.serial(0, M):
                # ---- stats: grid-strided partial sum / sumsq (fp32 accum) ----
                ls[0] = T.Cast("float32", 0)
                lq[0] = T.Cast("float32", 0)
                for j in T.serial(gtid, N, NT):
                    v = X[m, j]
                    ls[0] += v
                    lq[0] += v * v
                # block-local reduce into shared, then one global atomic/block
                red[0] = T.Cast("float32", 0)
                red[1] = T.Cast("float32", 0)
                T.sync_threads()
                T.atomic_add(red[0], ls[0])
                T.atomic_add(red[1], lq[0])
                T.sync_threads()
                if tid == 0:
                    T.atomic_add(Acc[m, 0], red[0])
                    T.atomic_add(Acc[m, 1], red[1])
                T.sync_grid()
                # ---- apply: X[m] is now L2-resident (single DRAM read) ----
                mean = Acc[m, 0] / N
                rstd = T.rsqrt(Acc[m, 1] / N - mean * mean + eps)
                for j in T.serial(gtid, N, NT):
                    Y[m, j] = (X[m, j] - mean) * rstd * Wt[j] + Bs[j]
                T.sync_grid()
    return main


_B = (_build,)
_CACHE = {}


class Model(nn.Module):
    """LayerNorm via a persistent cooperative-grid kernel that reads X once.
    self.ln is a parameter container only (weight/bias/eps); never called."""
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
    def forward(self, x):
        x = x.contiguous()
        w = self.ln.weight.contiguous().flatten()
        b = self.ln.bias.contiguous().flatten()
        M = x.shape[0]
        N = w.numel()
        x2 = x.reshape(M, N)
        key = (M, N)
        k = _CACHE.get(key)
        if k is None:
            k = _B[0](M, N, float(self.ln.eps))
            _CACHE[key] = k
        return k(x2, w, b).reshape(x.shape)
