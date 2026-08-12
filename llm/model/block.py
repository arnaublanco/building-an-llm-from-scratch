import torch
import torch.nn as nn
from llm.model.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # x: (batch, seq_len, d_model) -> same shape out
       return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize, then sub-layer, then residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x