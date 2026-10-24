# naive implementation
#可优化：预分词：分块后交给多个进程，各自计数，再汇总
#           ·multiprocessing、spawn、get_context、Pool、imap_unordered等
#       合并：维护pair频率和“哪些词包含该pair”的索引，每轮只处理受影响的词
#           ·倒排索引：根据某个内容，快速找到哪些对象包含它
#           ·pair_to_words:指出相应pair出现在哪些词中，避免遍历所有词
#           ·pair_freqs只去加减变化量，从而做到不全量更新

import os
from typing import BinaryIO
import regex as re
from cs336_basics.count_pretokens import count_pretoken

def initialize_vocab(special_tokens) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    num_special_tokens = len(special_tokens)

    # for i, string in enumerate(special_tokens, start=256):
    #    vocab[i] = string
    for i in range(num_special_tokens):
        vocab[i + 256] = bytes(special_tokens[i], "utf-8")
        # vocab.update{i + 256: bytes(special_tokens[i], "utf-8") for i in range(num_special_tokens)}
    # print(vocab)
    return vocab


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = initialize_vocab(special_tokens)
    #length_special_tokens = len(special_tokens)
    #vocablength = len(vocab)
    assert len(vocab) <= vocab_size, "vocab size should be within the max_num"
    counts = count_pretoken(input_path, special_tokens)
    #print(counts)
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []
    for step in range(num_merges):
        pair_freqs = {}
        for tokens, freq in counts.items():
            for pair in zip(tokens, tokens[1:]):
                  pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

        if not pair_freqs:
            break
         
        best_pair = max(
            pair_freqs,
            key=lambda pair: (pair_freqs[pair], pair),
        )

        merges.append(best_pair)
        mergedtoken = best_pair[0] + best_pair[1]
        new_counts = {}

        for tokens, freq in counts.items():
            i = 0
            new_tokens = []
            while i < len(tokens):
                if tokens[i:i + 2] == best_pair:
                    new_tokens.append(mergedtoken)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_counts[tuple(new_tokens)] = new_counts.get(tuple(new_tokens), 0) + freq
        counts = new_counts

        vocab[len(vocab)] = mergedtoken

    return vocab, merges

    #assert NotImplemented















