import torch
import torch.nn as nn

# IDENTITY BASELINE for the cross-DSL convergence redo (branch cross-dsl-6op-ncu-redo).
# Mirrors reference/normalization/layer_norm.py exactly -> speedup ~1.0x at iter 1.
# The AKO convergence run replaces this with the DSL kernel; the committed winner is
# preserved in git history and gated by tools/committed_baseline.csv.

class Model(nn.Module):
    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        return self.ln(x)


batch_size = 64
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]
