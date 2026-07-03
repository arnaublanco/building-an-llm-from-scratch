from collections import Counter

import regex

GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.inverse_vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _add_to_vocab(self, token: bytes) -> None:
        if token not in self.vocab:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token

    def _text_chunk_to_byte_tokens(self, chunk: str) -> list[bytes]:
        return [bytes([byte]) for byte in chunk.encode("utf-8")]

    def _count_pairs(self, chunks: list[list[bytes]]) -> Counter:
        pair_counts: Counter = Counter()
        for chunk in chunks:
            for left, right in zip(chunk, chunk[1:]):
                pair_counts[(left, right)] += 1
        return pair_counts    

    def _merge_pair_in_chunk(
            self, chunk: list[bytes], pair: tuple[bytes, bytes]
    ) -> list[bytes]:
        merged_chunk: list[bytes] = []
        index = 0
        while index < len(chunk):
            if (
                index < len(chunk) - 1
                and chunk[index] == pair[0]
                and chunk[index + 1] == pair[1]
            ):
                merged_chunk.append(chunk[index] + chunk[index + 1])
                index += 2
            else:
                merged_chunk.append(chunk[index])
                index += 1
        return merged_chunk
    
    
    def train(self, text: str, vocab_size: int) -> None:
        chunks = [
            self._text_chunk_to_byte_tokens(chunk)
            for chunk in GPT2_PATTERN.findall(text)
        ]
        num_merges = max(0, vocab_size - 256)
        for _ in range(num_merges):
            pair_counts = self._count_pairs(chunks)
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0][0]
            chunks = [self._merge_pair_in_chunk(chunk, best_pair) for chunk in chunks]
            self.merges.append(best_pair)
            self._add_to_vocab(best_pair[0] + best_pair[1])

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in GPT2_PATTERN.findall(text):
            tokens = self._text_chunk_to_byte_tokens(chunk)
            for pair in self.merges:
                tokens = self._merge_pair_in_chunk(tokens, pair)
            ids.extend(self.vocab[token] for token in tokens)
        return ids
    
    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(self.inverse_vocab[idx] for idx in ids)
        return byte_string.decode("utf-8", errors="replace")
    
    def tokenize(self, text: str) -> list[bytes]:
        tokens: list[bytes] = []
        for chunk in GPT2_PATTERN.findall(text):
            chunk_tokens = self._text_chunk_to_byte_tokens(chunk)
            for pair in self.merges:
                chunk_tokens = self._merge_pair_in_chunk(chunk_tokens, pair)
            tokens.extend(chunk_tokens)
        return tokens