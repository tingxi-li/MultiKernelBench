import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[3])
def _build(B, H, O, TH=128):
    @T.prim_func
    def main(LAST: T.Tensor((B, H), "float32"), W: T.Tensor((O, H), "float32"),
             BIAS: T.Tensor((O,), "float32"), Y: T.Tensor((B, O), "float32")):
        with T.Kernel(1, threads=TH) as _:
            for idx in T.Parallel(B * O):
                b = idx // O
                o = idx % O
                acc = T.alloc_local((1,), "float32")
                acc[0] = BIAS[o]
                for h in T.serial(H):
                    acc[0] += LAST[b, h] * W[o, h]
                Y[b, o] = acc[0]
    return main


_B = (_build,)
_CACHE = {}


class Model(nn.Module):
    """6-layer LSTM (cuDNN floor, h0/c0=zeros) + projection GEMM ported to TileLang."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :].contiguous()
        w = self.fc.weight.contiguous()
        b = self.fc.bias.contiguous()
        B = last.shape[0]; H = last.shape[1]; O = w.shape[0]
        key = (B, H, O)
        k = _CACHE.get(key)
        if k is None:
            k = _B[0](B, H, O)
            _CACHE[key] = k
        return k(last, w, b)
