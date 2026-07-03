from llm.tokenizer.bpe import BPETokenizer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data_file = ROOT / "data" / "espresso.txt"

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=512)
ids = tokenizer.encode("That's that me espresso")
print(f"Encoded: {ids}")
print(f"Decoded: {tokenizer.decode(ids)}")