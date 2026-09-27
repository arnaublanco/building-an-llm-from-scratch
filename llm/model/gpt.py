import torch
import torch.functional as F
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
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch, seq_len) token IDs
        x = self.token_emb(idx) # (batch, seq_len, d_model)
        x = self.pos_emb(x) # add learned positions
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len :]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k , logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
        return idx