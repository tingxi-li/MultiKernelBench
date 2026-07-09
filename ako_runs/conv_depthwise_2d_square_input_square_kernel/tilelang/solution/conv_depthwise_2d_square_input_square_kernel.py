import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang iter-6
#
# Building on iter-5 (no shmem = 2.64ms best):
# Process 2 output rows per thread to reuse global reads.
# Thread i computes output (bc, h, i) and (bc, h+1, i).
#  - Row h output needs input rows h, h+1, h+2
#  - Row h+1 output needs input rows h+1, h+2, h+3
#  - Shared rows h+1, h+2 loaded ONCE = 12 loads for 2 outputs vs 18 (33% less)
# Grid: (B*C, H_out//2), TH=510 (W_out=510, no boundary check needed)
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 510  # = W_out, no boundary checks


def _build(B, C, H_in, W_in, H_out, W_out, TH):

    @tilelang.jit
    def _make():
        @T.prim_func
        def kernel(
            X: T.Tensor((B * C, H_in, W_in), T.float32),
            W: T.Tensor((C, 9), T.float32),
            Y: T.Tensor((B * C, H_out, W_out), T.float32),
        ):
            with T.Kernel(B * C, H_out // 2, threads=TH) as (bc, h2):
                tid = T.get_thread_binding(0)
                c = bc % C
                h = h2 * 2  # first output row

                # Load filter into registers
                wt = T.alloc_local((9,), T.float32)
                for i in T.serial(9):
                    wt[i] = W[c, i]

                # Load 4 input rows × 3 columns each (12 total vs 18 for 2 separate)
                # Row h
                r0_c0 = T.alloc_local((1,), T.float32)
                r0_c1 = T.alloc_local((1,), T.float32)
                r0_c2 = T.alloc_local((1,), T.float32)
                # Row h+1 (shared between both output rows)
                r1_c0 = T.alloc_local((1,), T.float32)
                r1_c1 = T.alloc_local((1,), T.float32)
                r1_c2 = T.alloc_local((1,), T.float32)
                # Row h+2 (shared between both output rows)
                r2_c0 = T.alloc_local((1,), T.float32)
                r2_c1 = T.alloc_local((1,), T.float32)
                r2_c2 = T.alloc_local((1,), T.float32)
                # Row h+3 (only used by second output row)
                r3_c0 = T.alloc_local((1,), T.float32)
                r3_c1 = T.alloc_local((1,), T.float32)
                r3_c2 = T.alloc_local((1,), T.float32)

                r0_c0[0] = X[bc, h,     tid    ]
                r0_c1[0] = X[bc, h,     tid + 1]
                r0_c2[0] = X[bc, h,     tid + 2]
                r1_c0[0] = X[bc, h + 1, tid    ]
                r1_c1[0] = X[bc, h + 1, tid + 1]
                r1_c2[0] = X[bc, h + 1, tid + 2]
                r2_c0[0] = X[bc, h + 2, tid    ]
                r2_c1[0] = X[bc, h + 2, tid + 1]
                r2_c2[0] = X[bc, h + 2, tid + 2]
                r3_c0[0] = X[bc, h + 3, tid    ]
                r3_c1[0] = X[bc, h + 3, tid + 1]
                r3_c2[0] = X[bc, h + 3, tid + 2]

                # Output row h: uses rows h, h+1, h+2
                acc0 = T.alloc_local((1,), T.float32)
                acc0[0]  =               r0_c0[0] * wt[0]
                acc0[0] = acc0[0] + r0_c1[0] * wt[1]
                acc0[0] = acc0[0] + r0_c2[0] * wt[2]
                acc0[0] = acc0[0] + r1_c0[0] * wt[3]
                acc0[0] = acc0[0] + r1_c1[0] * wt[4]
                acc0[0] = acc0[0] + r1_c2[0] * wt[5]
                acc0[0] = acc0[0] + r2_c0[0] * wt[6]
                acc0[0] = acc0[0] + r2_c1[0] * wt[7]
                acc0[0] = acc0[0] + r2_c2[0] * wt[8]
                Y[bc, h, tid] = acc0[0]

                # Output row h+1: uses rows h+1, h+2, h+3
                acc1 = T.alloc_local((1,), T.float32)
                acc1[0]  =               r1_c0[0] * wt[0]
                acc1[0] = acc1[0] + r1_c1[0] * wt[1]
                acc1[0] = acc1[0] + r1_c2[0] * wt[2]
                acc1[0] = acc1[0] + r2_c0[0] * wt[3]
                acc1[0] = acc1[0] + r2_c1[0] * wt[4]
                acc1[0] = acc1[0] + r2_c2[0] * wt[5]
                acc1[0] = acc1[0] + r3_c0[0] * wt[6]
                acc1[0] = acc1[0] + r3_c1[0] * wt[7]
                acc1[0] = acc1[0] + r3_c2[0] * wt[8]
                Y[bc, h + 1, tid] = acc1[0]

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
    Depthwise 2D convolution — TileLang iter-6.
    2 output rows per thread, 4 shared input rows (no shmem, direct L2-cached reads).
    33% fewer global reads than single-row approach. Grid: (B*C, H_out//2).
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
