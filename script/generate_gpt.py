import torch
from pathlib import Path
from llm.tokenizer.bpe import BPETokenizer
from llm.model.gpt import GPT

ROOT = Path(__file__).resolve().parent.parent
ckpt = torch.load(ROOT / "checkpoints" / "gpt_wikibooks.pt", weights_only=False,)

tokenizer = BPETokenizer()
for pair in ckpt["merges"]:
    tokenizer.merges.append(pair)
    tokenizer._add_to_vocab(pair[0] + pair[1])

cfg = ckpt["config"]
model = GPT(
    vocab_size = cfg["vocab_size"],
    d_model = cfg["d_model"],
    num_heads = cfg["num_heads"],
    n_layers = cfg["n_layers"],
    max_len = cfg["max_len"],
)
model.load_state_dict(ckpt["model"])
model.eval()

prompt = "Paris is the capital of"
ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
out = model.generate(ids, max_new_tokens = 50, temperature = 0.8, top_k = 40)
print(tokenizer.decode(out[0].tolist()))