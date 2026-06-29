#!/usr/bin/env python3
"""Generate autotuned Triton elementwise solutions into ako_runs/<op>/solution/<op>.py."""
import os

ROOT = "/home/lxt230026/MultiKernelBench"

HEADER = '''import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({{'BLOCK_SIZE': 2048}}, num_warps=4),
        triton.Config({{'BLOCK_SIZE': 4096}}, num_warps=8),
        triton.Config({{'BLOCK_SIZE': 8192}}, num_warps=8),
        triton.Config({{'BLOCK_SIZE': 16384}}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _act_kernel(x_ptr, y_ptr, n_elements,{alpha_sig} BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    {math}
    tl.store(y_ptr + offs, y, mask=mask)


class Model(nn.Module):
    """{doc}"""
    def __init__(self{init_sig}):
        super().__init__(){init_body}

    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _act_kernel[grid](x, y, n{alpha_call})
        return y
'''

OPS = {
    "relu":        dict(math="y = tl.maximum(x, 0.0)", doc="ReLU via Triton (bandwidth-bound)."),
    "sigmoid":     dict(math="y = tl.sigmoid(x)", doc="Sigmoid via Triton (bandwidth-bound)."),
    "hardsigmoid": dict(math="y = tl.minimum(tl.maximum(x * 0.16666666666666666 + 0.5, 0.0), 1.0)",
                        doc="HardSigmoid = clamp(x/6 + 1/2, 0, 1) via Triton."),
    "gelu":        dict(math="y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))",
                        doc="Exact GELU (erf) via Triton."),
    "swish":       dict(math="y = x * tl.sigmoid(x)",
                        doc="Swish = x*sigmoid(x), FUSED single pass (eager = 2 passes)."),
    "elu":         dict(math="y = tl.where(x > 0, x, ALPHA * (tl.exp(x) - 1.0))",
                        doc="ELU via Triton (alpha from init).",
                        alpha=True),
}

for op, spec in OPS.items():
    is_elu = spec.get("alpha", False)
    code = HEADER.format(
        alpha_sig=" ALPHA," if is_elu else "",
        math=spec["math"],
        doc=spec["doc"],
        init_sig=", alpha=1.0" if is_elu else "",
        init_body="\n        self.alpha = alpha" if is_elu else "",
        alpha_call=", self.alpha" if is_elu else "",
    )
    path = f"{ROOT}/ako_runs/{op}/solution/{op}.py"
    with open(path, "w") as f:
        f.write(code)
    print("wrote", path)
