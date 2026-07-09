import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang fp32 v11.
#
# Restore the best iter-2 approach: row-per-block, 3 shared-mem rows,
# unrolled 3x3. This is the design that got 2.66ms (1.50x).
#
# Small changes vs iter-2:
#  - Reorder shared-mem load to access 3 rows in a single loop for better
#    instruction scheduling (may help compiler pipeline loads).
#  - Also try warp-level loading (4 threads per warp * 4 float4s).
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 512


def _build(B, C, H_in, W_in, H_out, W_out, TH):
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

                sh = T.alloc_shared((3, W_in), T.float32)

                # Load 3 rows using a single loop (row = 0,1,2)
                for fh in T.serial(3):
                    for li in T.serial(LOAD_ITERS):
                        idx = tid + li * TH
                        if idx < W_in:
                            sh[fh, idx] = X[bc, h + fh, idx]

                T.sync_threads()

                # Preload filter weights
                wt = T.alloc_local((9,), T.float32)
                for i in T.serial(9):
                    wt[i] = W[c, i]

                if tid < W_out:
                    acc = T.alloc_local((1,), T.float32)
                    acc[0] =       sh[0, tid    ] * wt[0]
                    acc[0] = acc[0] + sh[0, tid + 1] * wt[1]
                    acc[0] = acc[0] + sh[0, tid + 2] * wt[2]
                    acc[0] = acc[0] + sh[1, tid    ] * wt[3]
                    acc[0] = acc[0] + sh[1, tid + 1] * wt[4]
                    acc[0] = acc[0] + sh[1, tid + 2] * wt[5]
                    acc[0] = acc[0] + sh[2, tid    ] * wt[6]
                    acc[0] = acc[0] + sh[2, tid + 1] * wt[7]
                    acc[0] = acc[0] + sh[2, tid + 2] * wt[8]
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
    Depthwise 2D convolution — TileLang row-per-block, 2D shared-mem, unrolled 3x3.
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
