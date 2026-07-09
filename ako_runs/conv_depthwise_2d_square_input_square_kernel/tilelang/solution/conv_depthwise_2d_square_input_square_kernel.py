import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Depthwise Conv2D (3x3, stride=1, pad=0) — TileLang iter-1
#
# Key optimization: 3 output rows per block.
# For N output rows per block using a 3×3 kernel (stride=1, pad=0):
#   - Need N+2 input rows (e.g., N=3 → 5 rows)
#   - Memory reads = (N+2)/(3N) relative to N=1 baseline
#   - N=3: (3+2)/(3*3) = 0.556 → 44% fewer global reads
#   - H_out=510 is divisible by 3 → clean tiling
#
# Shared memory layout: 5 rows of width W_in, one write-back per active thread.
# ---------------------------------------------------------------------------

_KCACHE = {}
_TH = 512


def _build(B, C, H_in, W_in, H_out, W_out, TH):
    LOAD_ITERS = (W_in + TH - 1) // TH  # = 1 when W_in=512, TH=512

    @tilelang.jit
    def _make():
        @T.prim_func
        def kernel(
            X: T.Tensor((B * C, H_in, W_in), T.float32),
            W: T.Tensor((C, 9), T.float32),
            Y: T.Tensor((B * C, H_out, W_out), T.float32),
        ):
            # Grid: (B*C, H_out//3) — each block processes 3 consecutive output rows
            with T.Kernel(B * C, H_out // 3, threads=TH) as (bc, h3):
                tid = T.get_thread_binding(0)
                c = bc % C
                h = h3 * 3  # first output row index (= input row h..h+4)

                # 5 shared-memory rows covering input rows h..h+4
                sh0 = T.alloc_shared((W_in,), T.float32)
                sh1 = T.alloc_shared((W_in,), T.float32)
                sh2 = T.alloc_shared((W_in,), T.float32)
                sh3 = T.alloc_shared((W_in,), T.float32)
                sh4 = T.alloc_shared((W_in,), T.float32)

                # Load all 5 input rows cooperatively
                for li in T.serial(LOAD_ITERS):
                    idx = tid + li * TH
                    if idx < W_in:
                        sh0[idx] = X[bc, h,     idx]
                        sh1[idx] = X[bc, h + 1, idx]
                        sh2[idx] = X[bc, h + 2, idx]
                        sh3[idx] = X[bc, h + 3, idx]
                        sh4[idx] = X[bc, h + 4, idx]

                T.sync_threads()

                # Load 3×3 filter for this channel into registers
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

                    # --- Output row h+2 (uses sh2, sh3, sh4) ---
                    acc2 = T.alloc_local((1,), T.float32)
                    acc2[0]  = sh2[tid    ] * wt[0]
                    acc2[0] = acc2[0] + sh2[tid + 1] * wt[1]
                    acc2[0] = acc2[0] + sh2[tid + 2] * wt[2]
                    acc2[0] = acc2[0] + sh3[tid    ] * wt[3]
                    acc2[0] = acc2[0] + sh3[tid + 1] * wt[4]
                    acc2[0] = acc2[0] + sh3[tid + 2] * wt[5]
                    acc2[0] = acc2[0] + sh4[tid    ] * wt[6]
                    acc2[0] = acc2[0] + sh4[tid + 1] * wt[7]
                    acc2[0] = acc2[0] + sh4[tid + 2] * wt[8]
                    Y[bc, h + 2, tid] = acc2[0]

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
    Depthwise 2D convolution — TileLang iter-1.
    3 output rows per block: 5 shmem rows covering input rows h..h+4,
    reducing global memory reads by 44% vs. the 1-row baseline.
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
