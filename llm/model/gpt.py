import torch
import torch.nn as nn

from llm.model.block import TransformerBlock
from llm.model.positional import LearnedPositionalEncoding

class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
        n_layers: int, max_len: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch, seq_len) token IDs
        x = self.token_emb(idx) # (batch, seq_len, d_model)
        x = self.pos_emb(x) # add learned positions
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (batch, seq_len, vocab_size)
        return logits