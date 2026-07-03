import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/espresso.txt",
    model_prefix="models/sp_espresso",
    vocab_size=256,
    model_type="bpe",
)

sp = spm.SentencePieceProcessor(model_file="models/sp_espresso.model")
print(sp.encode_as_pieces("That's that me espresso"))