import torch
from pathlib import Path
from llm.tokenizer.bpe import BPETokenizer
from llm.model.gpt import GPT

ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "data" / "wikibooks.txt").read_text(encoding="utf-8")

tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=512)
tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)

block_size = 128
batch_size = 4

def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

model = GPT(
    vocab_size=len(tokenizer.vocab),
    d_model=64,
    num_heads=4,
    n_layers=2,
    max_len=block_size,
)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

for step in range(1000):
    x, y = get_batch(train_data, block_size, batch_size)
    logits = model(x)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()  # clear old gradients
    loss.backward()  # compute new gradients
    optimizer.step()  # update new weights

    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_data, block_size, batch_size)
            val_logits = model(x_val)
            val_loss = loss_fn(
                val_logits.view(-1, val_logits.size(-1)),
                y_val.view(-1),
            )
        model.train()
        print(step, "train", loss.item(), "val", val_loss.item())
