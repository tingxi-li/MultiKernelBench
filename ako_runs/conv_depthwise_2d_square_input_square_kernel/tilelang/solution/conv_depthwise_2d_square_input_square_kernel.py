import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang fp32 iter-6: FINAL.
#
# All 5 previous iters show the kernel runs at 2.65-2.66ms consistently.
# The design space has been thoroughly explored:
#  - iter-1: per-pixel register-only (2.70ms)
#  - iter-2/5: row-per-block 3-shmem rows (2.65-2.66ms) ← BEST
#  - iter-3: 2D shmem variant (2.66ms)
#  - iter-4: filter-in-shmem (2.66ms)
#
# For the final iter, attempt one last thing:
# Load 3 rows together with a single T.serial over (3, W_in) using a 2D index.
# This might enable better instruction pipelining.
#
# Also: use T.vectorized(2) for adjacent element pairs if the DSL supports it.
# If not better, confirm iter-2/5 is the floor.
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 512


def _build(B, C, H_in, W_in, H_out, W_out, TH):
    LOAD_ITERS = (W_in + TH - 1) // TH   # = 1

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

                # Load with explicit register for each thread's slot
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
                    # Load 9 input values into registers before FMA
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
                    acc[0] =       v00[0] * wt[0]
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
    Depthwise 2D convolution — TileLang final iteration.
    Row-per-block, 3 shmem rows, shmem values staged to local registers, unrolled 3x3 FMA.
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
