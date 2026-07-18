# Building an LLM from Scratch

Follow-along code for the [YouTube series](https://www.youtube.com/playlist?list=PLB-OkaiiwoiI) on building a GPT-3-style language model from scratch with Python and PyTorch.

## Follow along

1. Clone this repo.
2. Check out the tag for the episode you are watching.

```bash
git clone https://github.com/arnaublanco/building-an-llm-from-scratch.git
cd building-an-llm-from-scratch
git checkout episode-2   # use episode-2, episode-3, etc.
```

Each tag (`episode-2`, `episode-3`, …) matches the code **at the end of that episode**. To work with the latest code from the whole series, stay on `main`.

## Episodes

| Episode | Topic | Watch | Code |
|---------|-------|-------|------|
| 1 | Introduction & history of LLMs | [YouTube](https://www.youtube.com/watch?v=AA1OozCxg4M) | — |
| 2 | Tokenization & embeddings | [YouTube](https://www.youtube.com/watch?v=QR5wXrY1rto) | [`episode-2`](../../tree/episode-2) |
| 3 | *Coming soon* | — | — |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Scripts live in `script/`. Run them from the **repo root**:

```bash
PYTHONPATH=. python script/<script_name>.py
```

`requirements.txt` grows as the series adds new dependencies.

## Project layout

```text
data/       # training text
llm/        # model and training code
script/     # entry points you run
```

## License

MIT — see [LICENSE](LICENSE).
