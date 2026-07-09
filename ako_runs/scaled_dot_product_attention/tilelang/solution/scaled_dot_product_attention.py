import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# Flash attention for large head_dim (1024) using D-tiling.
# Strategy: outer loop over output D tiles (regular for loop),
# inner flash-attention KV loop (T.Pipelined).
# QK is accumulated by n_d_tiles inner gemm calls (splitting dim).
# block_M=64, block_N=64, D_TILE=128, n_d_tiles=8.
# Shared memory: 4 * 64*128*2B = 64KB per SM (fits Ada Lovelace).
# Fragment acc_o: 64*128*4B = 32KB registers.
# Bench uses --precision float16 so Q/K/V arrive as float16.

def _build_kernel(batch, heads, seq_len, dim,
                  block_M=64, block_N=64, D_TILE=128,
                  num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    dtype = T.float16
    accum_dtype = T.float32
    n_d_tiles = dim // D_TILE
    n_kv_blocks = seq_len // block_N

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, D_TILE], dtype)
            K_shared = T.alloc_shared([block_N, D_TILE], dtype)
            V_shared = T.alloc_shared([block_N, D_TILE], dtype)
            O_shared = T.alloc_shared([block_M, D_TILE], dtype)

            acc_s       = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast  = T.alloc_fragment([block_M, block_N], dtype)
            acc_o       = T.alloc_fragment([block_M, D_TILE],  accum_dtype)

            scores_max      = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale    = T.alloc_fragment([block_M], accum_dtype)
            scores_sum      = T.alloc_fragment([block_M], accum_dtype)
            logsum          = T.alloc_fragment([block_M], accum_dtype)

            for d_out in range(n_d_tiles):
                T.fill(acc_o,      0)
                T.fill(logsum,     0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for k in T.Pipelined(n_kv_blocks, num_stages=num_stages):
                    # Initialize acc_s then accumulate QK over D chunks
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = 0
                    for d_in in range(n_d_tiles):
                        T.copy(Q[bz, by,
                                 bx * block_M : (bx + 1) * block_M,
                                 d_in * D_TILE : (d_in + 1) * D_TILE], Q_shared)
                        T.copy(K[bz, by,
                                 k * block_N : (k + 1) * block_N,
                                 d_in * D_TILE : (d_in + 1) * D_TILE], K_shared)
                        T.gemm(Q_shared, K_shared, acc_s,
                               transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)

                    # Online softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_M, D_TILE):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[bz, by,
                             k * block_N : (k + 1) * block_N,
                             d_out * D_TILE : (d_out + 1) * D_TILE], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o,
                           policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, D_TILE):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared,
                       Output[bz, by,
                              bx * block_M : (bx + 1) * block_M,
                              d_out * D_TILE : (d_out + 1) * D_TILE])

    return main


_kernel_cache = {}


def _get_kernel(batch, heads, seq_len, dim):
    key = (batch, heads, seq_len, dim)
    if key not in _kernel_cache:
        fn = _build_kernel(batch, heads, seq_len, dim,
                           block_M=64, block_N=64, D_TILE=128,
                           num_stages=1, threads=128)
        _kernel_cache[key] = tilelang.compile(
            fn,
            out_idx=[3],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            target="cuda",
        )
    return _kernel_cache[key]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, S, D = Q.shape
        kernel = _get_kernel(B, H, S, D)
        return kernel(Q, K, V)
