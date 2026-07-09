import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang fp32 iter-5.
#
# Restore iter-2 EXACT design (best: 1.50x, 2.66ms).
# The design space has converged. The kernel is consistently at 2.66ms.
#
# One last hypothesis: use TH=W_out exactly (=510) instead of 512.
# With 510 threads no tail-guard needed: all threads compute an output pixel.
# This might help the compiler (no `if tid < W_out` branch).
# Also: W_in=512 > TH=510, so we need 2 LOAD_ITERS for the 2 extra elements.
# Actually: threads 0..509 load sh[tid]=X[..., tid] and sh[510], sh[511]
# are loaded by threads 0, 1 in a second iteration. This adds complexity.
#
# Actually TH=512 with W_in=512 is already the cleanest design (exactly 1:1).
# The guard `if tid < W_out (=510)` only gates 2 idle threads out of 512.
#
# For iter-5 commit the iter-2 exact design for correctness, and run final.
# This gives a clean record and confirms the best result.
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 512


def _build(B, C, H_in, W_in, H_out, W_out, TH):
    LOAD_ITERS = (W_in + TH - 1) // TH   # = 1 for W_in=512, TH=512

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

                # Load 3 input rows (W_in=512, TH=512, LOAD_ITERS=1)
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
                    acc = T.alloc_local((1,), T.float32)
                    acc[0] =       sh0[tid    ] * wt[0]
                    acc[0] = acc[0] + sh0[tid + 1] * wt[1]
                    acc[0] = acc[0] + sh0[tid + 2] * wt[2]
                    acc[0] = acc[0] + sh1[tid    ] * wt[3]
                    acc[0] = acc[0] + sh1[tid + 1] * wt[4]
                    acc[0] = acc[0] + sh1[tid + 2] * wt[5]
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
    Depthwise 2D convolution — TileLang row-per-block, 3 shmem rows, unrolled 3x3.
    Best configuration (iter-2 confirmed best).
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
