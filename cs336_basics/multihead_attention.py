import torch
from torch import nn

from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.linear_layer import Linear
from cs336_basics.rope_layer import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            rope: RotaryPositionalEmbedding | None=None,
            device=None,
            dtype=None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

        self.output_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        head_shape = (
            *x.shape[:-1],
            self.num_heads,
            self.d_k,
        )

        Q = self.q_proj(x).reshape(head_shape).transpose(-3, -2)
        K = self.k_proj(x).reshape(head_shape).transpose(-3, -2)
        V = self.v_proj(x).reshape(head_shape).transpose(-3, -2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len, device=x.device
                )

            rope_positions = token_positions.unsqueeze(-2)

            Q = self.rope(Q, rope_positions)
            K = self.rope(K, rope_positions)

        positions = torch.arange(seq_len, device=x.device)
        mask = positions[None, :] <= positions[:, None]

        head_outputs = scaled_dot_product_attention(
            Q, K, V, mask=mask
        )

        merged = head_outputs.transpose(-3, -2).reshape(x.shape)

        return self.output_proj(merged)


