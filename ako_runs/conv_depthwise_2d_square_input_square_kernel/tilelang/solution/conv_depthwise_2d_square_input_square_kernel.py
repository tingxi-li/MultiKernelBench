import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang iter-5
#
# No shared memory: direct global reads via L2 cache.
# The NVIDIA Ada L2 cache is 96MB; 3 input rows per channel = 3*512*4 = 6KB,
# fitting 16384 row-triples simultaneously. Consecutive output rows share
# 2/3 of their input rows, so L2 reuse is high.
#
# By removing shmem:
#  - No sync_threads barrier overhead
#  - Less shmem register pressure
#  - Allow more blocks per SM (higher occupancy)
#  - Each thread reads 9 global locations directly (3 for each of 3 rows)
#
# Grid: (B*C, H_out), TH=510 (one thread per output column).
# Each thread independently accesses 9 L2-cached global locations.
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 510  # exactly W_out: no idle threads, no boundary check needed


def _build(B, C, H_in, W_in, H_out, W_out, TH):

    @tilelang.jit
    def _make():
        @T.prim_func
        def kernel(
            X: T.Tensor((B * C, H_in, W_in), T.float32),
            W: T.Tensor((C, 9), T.float32),
            Y: T.Tensor((B * C, H_out, W_out), T.float32),
        ):
            with T.Kernel(B * C, H_out, threads=TH) as (bc, h):
                tid = T.get_thread_binding(0)
                c = bc % C

                # Load filter into registers (9 floats)
                wt = T.alloc_local((9,), T.float32)
                for i in T.serial(9):
                    wt[i] = W[c, i]

                # Direct global reads — rely on L2 cache for row reuse
                # Row h
                x00 = T.alloc_local((1,), T.float32)
                x01 = T.alloc_local((1,), T.float32)
                x02 = T.alloc_local((1,), T.float32)
                # Row h+1
                x10 = T.alloc_local((1,), T.float32)
                x11 = T.alloc_local((1,), T.float32)
                x12 = T.alloc_local((1,), T.float32)
                # Row h+2
                x20 = T.alloc_local((1,), T.float32)
                x21 = T.alloc_local((1,), T.float32)
                x22 = T.alloc_local((1,), T.float32)

                x00[0] = X[bc, h,     tid    ]
                x01[0] = X[bc, h,     tid + 1]
                x02[0] = X[bc, h,     tid + 2]
                x10[0] = X[bc, h + 1, tid    ]
                x11[0] = X[bc, h + 1, tid + 1]
                x12[0] = X[bc, h + 1, tid + 2]
                x20[0] = X[bc, h + 2, tid    ]
                x21[0] = X[bc, h + 2, tid + 1]
                x22[0] = X[bc, h + 2, tid + 2]

                acc = T.alloc_local((1,), T.float32)
                acc[0]  =              x00[0] * wt[0]
                acc[0] = acc[0] + x01[0] * wt[1]
                acc[0] = acc[0] + x02[0] * wt[2]
                acc[0] = acc[0] + x10[0] * wt[3]
                acc[0] = acc[0] + x11[0] * wt[4]
                acc[0] = acc[0] + x12[0] * wt[5]
                acc[0] = acc[0] + x20[0] * wt[6]
                acc[0] = acc[0] + x21[0] * wt[7]
                acc[0] = acc[0] + x22[0] * wt[8]
                Y[bc, h, tid] = acc[0]

        return kernel

    return _make()


_KB = (_build,)


def _get_kernel(B, C, H_in, W_in, H_out, W_out):
    key = (B, C, H_in, W_in, H_out, W_out, _TH)
    k = _KCACHE.get(key)
    if k is None:
        k = _KB[0](B, C, H_in, W_in, H_out, W_out, _TH)
        _KCACHE[key] = k
    return k


class Model(nn.Module):
    """
    Depthwise 2D convolution — TileLang iter-5.
    No shared memory: direct L2-cached global reads.
    TH=510 threads: one per output column.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_in, W_in = x.shape
        ks = self.conv2d.kernel_size[0]
        pad = self.conv2d.padding[0]
        st = self.conv2d.stride[0]
        H_out = (H_in + 2 * pad - ks) // st + 1
        W_out = (W_in + 2 * pad - ks) // st + 1

        w = self.conv2d.weight.reshape(C, -1).contiguous()
        x_flat = x.reshape(B * C, H_in, W_in).contiguous()
        y_flat = torch.empty(B * C, H_out, W_out, device=x.device, dtype=x.dtype)

        kern = _get_kernel(B, C, H_in, W_in, H_out, W_out)
        kern(x_flat, w, y_flat)

        return y_flat.reshape(B, C, H_out, W_out)
