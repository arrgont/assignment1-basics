import torch
from torch import nn

from cs336_basics.multihead_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm_layer import RMSNorm
from cs336_basics.rope_layer import RotaryPositionalEmbedding
from cs336_basics.swiglu_layer import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RotaryPositionalEmbedding | None = None,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        attn_output = self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )

        y = x + attn_output

        ffn_output = self.ffn(self.ln2(y))
        return y + ffn_output