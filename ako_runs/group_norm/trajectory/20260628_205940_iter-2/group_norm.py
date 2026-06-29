import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _group_fused_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    group_numel, chan_numel, G, GPC, eps,
    BLOCK: tl.constexpr,
):
    # Single fused launch: one program per (batch, group). Pass 1 reduces the
    # group's contiguous block to mean/rstd (kept in registers); pass 2 re-reads
    # and writes the normalized + per-channel-affine output. Saves a kernel
    # launch and the mean/rstd global round-trip vs the two-kernel version.
    pid = tl.program_id(0)
    base = pid * group_numel
    g_local = pid % G

    acc_sum = tl.zeros([BLOCK], dtype=tl.float32)
    acc_sumsq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_numel, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + base + idx).to(tl.float32)
        acc_sum += v
        acc_sumsq += v * v

    n = group_numel.to(tl.float32)
    mean = tl.sum(acc_sum, axis=0) / n
    var = tl.sum(acc_sumsq, axis=0) / n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, group_numel, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        c = g_local * GPC + off // chan_numel  # channel constant within a chunk
        scale = rstd * tl.load(w_ptr + c)
        shift = tl.load(b_ptr + c) - mean * scale
        v = tl.load(x_ptr + base + idx).to(tl.float32)
        tl.store(y_ptr + base + idx, v * scale + shift)


class Model(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.num_groups = num_groups
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        N, C = x.shape[0], x.shape[1]
        spatial = x.numel() // (N * C)
        GPC = C // self.num_groups          # channels per group
        chan_numel = spatial                # elements per channel
        group_numel = GPC * spatial         # elements per group
        n_groups_total = N * self.num_groups

        y = torch.empty_like(x)
        _group_fused_kernel[(n_groups_total,)](
            x, y, self.gn.weight, self.gn.bias,
            group_numel, chan_numel, self.num_groups, GPC, self.gn.eps,
            BLOCK=4096,
            num_warps=8,
        )
        return y
