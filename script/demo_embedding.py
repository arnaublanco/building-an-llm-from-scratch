import torch
import torch.nn as nn
from pathlib import Path

from llm.tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
data_file = ROOT / "data" / "espresso.txt"

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=512)
ids = tokenizer.encode("That's that me espresso")

embedding = nn.Embedding(len(tokenizer.vocab), 64)
vectors = embedding(torch.tensor(ids))

print(vectors)