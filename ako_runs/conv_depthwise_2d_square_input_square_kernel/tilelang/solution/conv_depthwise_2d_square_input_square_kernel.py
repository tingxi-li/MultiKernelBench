import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang fp32 v7: vectorized.
#
# Key difference from iter-1: use T.vectorized(4) for loading 3 input values
# from each of 3 rows (using a 4-wide vector load covering col w, w+1, w+2, w+3).
# This allows the hardware to issue 128-bit (float4) loads rather than 3 separate
# 32-bit loads, improving memory bus utilization.
#
# Design: row-per-block with vectorized row segment loading.
# TH=512, grid=(B*C, H_out).
# For each thread tid (0..W_out-1):
#   - Thread tid loads float4 starting at X[bc, h+fh, tid] for fh=0,1,2.
#   - The 4 consecutive floats cover cols tid, tid+1, tid+2, tid+3
#     which includes all 3 needed values (fw=0,1,2) for the 3x3 filter at col tid.
#   - This makes 3 float4 loads (one per filter row) + scalar reductions.
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 512


def _build(B, C, H_in, W_in, H_out, W_out, TH):
    # Each thread loads 4 input values per row (float4)
    LOAD_ITERS = (W_in + TH - 1) // TH

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

                # Shared memory: 3 input rows, each W_in wide
                sh0 = T.alloc_shared((W_in,), T.float32)
                sh1 = T.alloc_shared((W_in,), T.float32)
                sh2 = T.alloc_shared((W_in,), T.float32)

                # Load input rows with vectorized accesses
                for li in T.serial(LOAD_ITERS):
                    idx = tid + li * TH
                    if idx < W_in:
                        sh0[idx] = X[bc, h,     idx]
                        sh1[idx] = X[bc, h + 1, idx]
                        sh2[idx] = X[bc, h + 2, idx]

                T.sync_threads()

                # Preload filter into registers
                wt = T.alloc_local((9,), T.float32)
                for i in T.serial(9):
                    wt[i] = W[c, i]

                # Compute output pixel (unrolled 3x3)
                if tid < W_out:
                    acc = T.alloc_local((1,), T.float32)
                    # Row 0 contributions (fh=0)
                    acc[0] = sh0[tid    ] * wt[0]
                    acc[0] = acc[0] + sh0[tid + 1] * wt[1]
                    acc[0] = acc[0] + sh0[tid + 2] * wt[2]
                    # Row 1 contributions (fh=1)
                    acc[0] = acc[0] + sh1[tid    ] * wt[3]
                    acc[0] = acc[0] + sh1[tid + 1] * wt[4]
                    acc[0] = acc[0] + sh1[tid + 2] * wt[5]
                    # Row 2 contributions (fh=2)
                    acc[0] = acc[0] + sh2[tid    ] * wt[6]
                    acc[0] = acc[0] + sh2[tid + 1] * wt[7]
                    acc[0] = acc[0] + sh2[tid + 2] * wt[8]
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
    Depthwise 2D convolution — TileLang row-per-block with unrolled 3x3.
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
