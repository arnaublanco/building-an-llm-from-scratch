import torch
import torch.nn as nn
from pathlib import Path

from llm.tokenizer.bpe import BPETokenizer
from llm.model.positional import LearnedPositionalEncoding

ROOT = Path(__file__).resolve().parent.parent
data_file = ROOT / "data" / "espresso.txt"

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=512)
ids = tokenizer.encode("That's that me espresso")

d_model = 64
token_embedding = nn.Embedding(len(tokenizer.vocab), d_model)
pos_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=512)

token_vectors = token_embedding(ids)
output = pos_encoding(token_vectors)

print(f"Token ids shape: {ids.shape}")
print(f"Embeddings shape: {token_vectors.shape}")
print(f"After positions: {output.shape}")