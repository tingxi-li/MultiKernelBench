import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(x_ptr, psum_ptr, psumsq_ptr,
                  N, S,
                  CHUNK: tl.constexpr, BLOCK: tl.constexpr, NS: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    base = row * N + chunk * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    accsq = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.range(0, CHUNK, BLOCK, num_stages=NS):
        offs = base + i + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        acc += x
        accsq += x * x
    s = tl.sum(acc, axis=0)
    ssq = tl.sum(accsq, axis=0)
    out_idx = row * S + chunk
    tl.store(psum_ptr + out_idx, s)
    tl.store(psumsq_ptr + out_idx, ssq)


@triton.jit
def _apply_kernel(x_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, out_ptr,
                  N,
                  CHUNK: tl.constexpr, BLOCK: tl.constexpr, NS: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    base = row * N + chunk * CHUNK
    woff = chunk * CHUNK
    m = tl.load(mean_ptr + row)
    r = tl.load(rstd_ptr + row)
    for i in tl.range(0, CHUNK, BLOCK, num_stages=NS):
        idx = i + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + idx)
        w = tl.load(w_ptr + woff + idx)
        b = tl.load(b_ptr + woff + idx)
        y = (x - m) * r * w + b
        tl.store(out_ptr + base + idx, y)


class Model(nn.Module):
    """LayerNorm over the last dims, split-row two-pass Triton kernel."""

    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        # tuning knobs
        self.S = 16          # chunks per row (programs = M * S)
        self.BLOCK = 4096    # inner tile size
        self.NS = 3          # software-pipeline stages for the load loop
        self.NW = 4          # num_warps

    def forward(self, x):
        ln = self.ln
        eps = ln.eps
        w = ln.weight
        b = ln.bias
        M = x.shape[0]
        N = w.numel()

        x2 = x.reshape(M, N)
        wf = w.reshape(N)
        bf = b.reshape(N)
        out = torch.empty_like(x2)

        S = self.S
        BLOCK = self.BLOCK
        NS = self.NS
        CHUNK = N // S

        psum = torch.empty((M, S), device=x.device, dtype=torch.float32)
        psumsq = torch.empty((M, S), device=x.device, dtype=torch.float32)

        grid = (M, S)
        _stats_kernel[grid](x2, psum, psumsq, N, S, CHUNK, BLOCK, NS,
                            num_warps=self.NW)

        ps = psum.double().sum(1)
        pss = psumsq.double().sum(1)
        mean = ps / N
        var = pss / N - mean * mean
        rstd = torch.rsqrt(var + eps)
        mean_f = mean.to(torch.float32)
        rstd_f = rstd.to(torch.float32)

        _apply_kernel[grid](x2, wf, bf, mean_f, rstd_f, out, N, CHUNK, BLOCK, NS,
                            num_warps=self.NW)

        return out.reshape(x.shape)
