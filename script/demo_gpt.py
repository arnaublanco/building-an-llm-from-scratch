import torch
from pathlib import Path
from llm.tokenizer.bpe import BPETokenizer
from llm.model.gpt import GPT

ROOT = Path(__file__).resolve().parent.parent
data_file = ROOT / "data" / "espresso.txt"

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=512)
ids = torch.tensor([tokenizer.encode("That's that me espresso")]) # (1, seq_len)

model = GPT(
    vocab_size=len(tokenizer.vocab),
    d_model=64,
    num_heads=4,
    n_layers=2,
    max_len=512
)
logits = model(ids)

print(f"Token IDs shape: {ids.shape}")
print(f"Logits shape: {logits.shape}") # (1, seq_len, vocab_size)

