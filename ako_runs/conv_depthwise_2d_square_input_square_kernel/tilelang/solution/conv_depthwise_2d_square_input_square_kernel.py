import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang iter-4
#
# Reset to proven best: 1-row-per-block, TH=512.
# This time, try with padding support so the kernel handles general cases,
# and also try using T.vectorized for potential auto-vectorization.
#
# Actually: try increasing block size to TH=1024 to get 2 warps per block.
# Wait — W_in=512 fits exactly in TH=512 threads. TH=1024 means half the
# threads are idle during load. Not better.
#
# True insight: the prior best (2.65ms) uses 1 block per output row (B*C=1024
# blocks × 510 rows = 522K block launches). RTX6000Ada has 76 SMs.
# 522K/76 = 6869 waves. At ~2-3ms per wave, this confirms we're occupancy-limited.
#
# Key idea: process B*C*H_out = 522240 "work units" but pack 2 channels per
# block (share the spatial loads). Since inputs for different channels are
# different memory locations, this doesn't reduce bandwidth but increases
# compute density per block.
#
# But actually: the bottleneck is reading 3 input rows per output row.
# Try BF16 reduce: load fp32 inputs, store as fp32 output but compute with
# 9 MACs using 2-wide FMA to halve compute. Not relevant for fp32.
#
# Simplest idea: go back to the exact iter-6 design that achieves 2.65ms
# but try TH=256 for higher occupancy (2x blocks per SM = more warps
# to hide memory latency). The shmem is 3*512*4=6KB per block.
# With TH=256: shmem still 6KB but each block is 256 threads = 8 warps.
# Current TH=512 = 16 warps. Higher occupancy with smaller blocks?
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 512


def _build(B, C, H_in, W_in, H_out, W_out, TH):
    LOAD_ITERS = (W_in + TH - 1) // TH  # = ceil(512/256) = 2

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

                sh0 = T.alloc_shared((W_in,), T.float32)
                sh1 = T.alloc_shared((W_in,), T.float32)
                sh2 = T.alloc_shared((W_in,), T.float32)

                for li in T.serial(LOAD_ITERS):
                    idx = tid + li * TH
                    if idx < W_in:
                        sh0[idx] = X[bc, h,     idx]
                        sh1[idx] = X[bc, h + 1, idx]
                        sh2[idx] = X[bc, h + 2, idx]

                T.sync_threads()

                wt = T.alloc_local((9,), T.float32)
                for i in T.serial(9):
                    wt[i] = W[c, i]

                if tid < W_out:
                    v00 = T.alloc_local((1,), T.float32)
                    v01 = T.alloc_local((1,), T.float32)
                    v02 = T.alloc_local((1,), T.float32)
                    v10 = T.alloc_local((1,), T.float32)
                    v11 = T.alloc_local((1,), T.float32)
                    v12 = T.alloc_local((1,), T.float32)
                    v20 = T.alloc_local((1,), T.float32)
                    v21 = T.alloc_local((1,), T.float32)
                    v22 = T.alloc_local((1,), T.float32)

                    v00[0] = sh0[tid    ]
                    v01[0] = sh0[tid + 1]
                    v02[0] = sh0[tid + 2]
                    v10[0] = sh1[tid    ]
                    v11[0] = sh1[tid + 1]
                    v12[0] = sh1[tid + 2]
                    v20[0] = sh2[tid    ]
                    v21[0] = sh2[tid + 1]
                    v22[0] = sh2[tid + 2]

                    acc = T.alloc_local((1,), T.float32)
                    acc[0]  =             v00[0] * wt[0]
                    acc[0] = acc[0] + v01[0] * wt[1]
                    acc[0] = acc[0] + v02[0] * wt[2]
                    acc[0] = acc[0] + v10[0] * wt[3]
                    acc[0] = acc[0] + v11[0] * wt[4]
                    acc[0] = acc[0] + v12[0] * wt[5]
                    acc[0] = acc[0] + v20[0] * wt[6]
                    acc[0] = acc[0] + v21[0] * wt[7]
                    acc[0] = acc[0] + v22[0] * wt[8]
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
    Depthwise 2D convolution — TileLang iter-4.
    1-row-per-block, TH=256, 3 shmem rows. Higher occupancy test.
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
