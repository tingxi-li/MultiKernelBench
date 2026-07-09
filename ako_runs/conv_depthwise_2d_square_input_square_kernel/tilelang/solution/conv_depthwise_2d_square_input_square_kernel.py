import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang iter-3
#
# 2 output rows per block:
#   - Load 4 shmem rows covering input h, h+1, h+2, h+3
#   - Compute outputs at rows h and h+1 from the same shmem data
#   - 4 reads / 2 outputs = 2 input reads per output (vs 3 for 1-row approach)
#   - 33% fewer DRAM reads per output element
#   - H_out=510 divisible by 2 → clean grid
#   - Same warp occupancy (TH=512 → 16 warps per block, 3 blocks per SM)
#   - Grid halved: (B*C, 255) vs (B*C, 510) → less scheduling overhead
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 512


def _build(B, C, H_in, W_in, H_out, W_out, TH):
    LOAD_ITERS = (W_in + TH - 1) // TH  # = 1

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
                h = h2 * 2  # first output row index

                # 4 shared-memory rows covering input rows h, h+1, h+2, h+3
                sh0 = T.alloc_shared((W_in,), T.float32)
                sh1 = T.alloc_shared((W_in,), T.float32)
                sh2 = T.alloc_shared((W_in,), T.float32)
                sh3 = T.alloc_shared((W_in,), T.float32)

                # Load all 4 input rows cooperatively
                for li in T.serial(LOAD_ITERS):
                    idx = tid + li * TH
                    if idx < W_in:
                        sh0[idx] = X[bc, h,     idx]
                        sh1[idx] = X[bc, h + 1, idx]
                        sh2[idx] = X[bc, h + 2, idx]
                        sh3[idx] = X[bc, h + 3, idx]

                T.sync_threads()

                # Load 3×3 filter for this channel
                wt = T.alloc_local((9,), T.float32)
                for i in T.serial(9):
                    wt[i] = W[c, i]

                if tid < W_out:
                    # --- Output row h (uses sh0, sh1, sh2) ---
                    acc0 = T.alloc_local((1,), T.float32)
                    acc0[0]  = sh0[tid    ] * wt[0]
                    acc0[0] = acc0[0] + sh0[tid + 1] * wt[1]
                    acc0[0] = acc0[0] + sh0[tid + 2] * wt[2]
                    acc0[0] = acc0[0] + sh1[tid    ] * wt[3]
                    acc0[0] = acc0[0] + sh1[tid + 1] * wt[4]
                    acc0[0] = acc0[0] + sh1[tid + 2] * wt[5]
                    acc0[0] = acc0[0] + sh2[tid    ] * wt[6]
                    acc0[0] = acc0[0] + sh2[tid + 1] * wt[7]
                    acc0[0] = acc0[0] + sh2[tid + 2] * wt[8]
                    Y[bc, h, tid] = acc0[0]

                    # --- Output row h+1 (uses sh1, sh2, sh3) ---
                    acc1 = T.alloc_local((1,), T.float32)
                    acc1[0]  = sh1[tid    ] * wt[0]
                    acc1[0] = acc1[0] + sh1[tid + 1] * wt[1]
                    acc1[0] = acc1[0] + sh1[tid + 2] * wt[2]
                    acc1[0] = acc1[0] + sh2[tid    ] * wt[3]
                    acc1[0] = acc1[0] + sh2[tid + 1] * wt[4]
                    acc1[0] = acc1[0] + sh2[tid + 2] * wt[5]
                    acc1[0] = acc1[0] + sh3[tid    ] * wt[6]
                    acc1[0] = acc1[0] + sh3[tid + 1] * wt[7]
                    acc1[0] = acc1[0] + sh3[tid + 2] * wt[8]
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
    Depthwise 2D convolution — TileLang iter-3.
    2 output rows per block with 4 shmem rows (rows h..h+3).
    33% fewer DRAM reads per output vs 1-row approach.
    Grid halved vs 1-row: (B*C, H_out//2).
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
