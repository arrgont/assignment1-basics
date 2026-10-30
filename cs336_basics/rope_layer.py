import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device=None,
    ):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even"

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_dims = torch.arange(
            0, d_k, 2,
            device=device,
            dtype=torch.float32,
        )

        inv_freq = 1.0 / (theta ** (pair_dims / d_k))

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer(
            "cos_cached", angles.cos(), persistent=False
        )

        self.register_buffer(
            "sin_cached", angles.sin(), persistent=False
        )

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor,
    ) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.stack(
            (out_even, out_odd), dim=-1
        ).flatten(-2)

        return  out.to(x.dtype)