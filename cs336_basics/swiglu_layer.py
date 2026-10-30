import math

import torch
from torch import nn

from cs336_basics.linear_layer import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int | None = None,
            device=None,
            dtype=None,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = math.ceil((8 * d_model / 3) / 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(
            d_model, d_ff, device=device, dtype=dtype
        )

        self.w2 = Linear(
            d_ff, d_model, device=device, dtype=dtype
        )

        self.w3 = Linear(
            d_model, d_ff, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        content = self.w3(x)

        hidden = gate * content

        return self.w2(hidden)