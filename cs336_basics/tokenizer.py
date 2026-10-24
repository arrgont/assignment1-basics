
import pickle
from collections import deque
from collections.abc import Iterator, Iterable
from secrets import token_bytes

import regex as re

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
            self,
            vocab:dict[int, bytes],
            merges:list[tuple[bytes, bytes]],
            special_tokens:list[str] | None = None,
    ):
        # 浅拷贝，只复制字典这个容器，其中键值仍然引用原来的对象；但注意：虽然是浅拷贝但仍是创建了新字典
        self.vocab = dict(vocab)#避免添加特殊token时修改调用者词表
        self.merges = list(merges)

        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.special_tokens = set(special_tokens or [])
        #self.special_tokens = list(
        #    dict.fromkeys(special_tokens or [])
        #)
        if special_tokens:
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.bytes_to_id:
                    self.vocab[len(self.vocab)] = special_token_bytes
                    self.bytes_to_id[special_token_bytes] = len(self.vocab)

        ordered_specials = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )


        self.special_pattern = (
            "(" + "|".join(re.escape(s) for s in ordered_specials) + ")"
            if ordered_specials else None
        )

    @classmethod
    def from_files(
            cls,
            vocab_filepath:str,
            merges_filepath:str,
            special_tokens:list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)


    def _bpe(self, word):

        tokens = [
            bytes([b])
            for b in word.encode("utf-8")
        ]

        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")

            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(
                    pair, float("inf")
                )

                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_tokens = []
            i = 0

            while i < len(tokens):
                if(
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return [
            self.bytes_to_id[token]
            for token in tokens
        ]


    def encode(
            self,
            text: str,
    ) -> list[int]:
        ids = []
        parts = (
            re.split(self.special_pattern, text)
            if self.special_pattern else [text]
        )

        for part in parts:
            if part in self.special_tokens:
                ids.append(
                    self.bytes_to_id[part.encode("utf-8")]
                )
            else:
                for match in re.finditer(self.PAT, part):
                    ids.extend(self._bpe(match.group(0)))
        return ids



    def encode_iterable(
            self,
            iterable:Iterable[str]
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


    def decode(
            self,
            ids:list[int]
    ) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")



