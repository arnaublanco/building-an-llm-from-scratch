import torch

from llm.model.attention import MultiHeadAttention
x = torch.randn(2, 5, 64)
mha = MultiHeadAttention(d_model=64, num_heads=4)
out = mha(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")