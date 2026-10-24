# 维护 pair 频率和词索引，每轮只更新受合并影响的词。
from collections import defaultdict

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
    pair_freqs = {}

    pair_to_words = defaultdict(set)

    def update_pairs(tokens, delta):
        # 重复出现的 pair 需要按出现次数累计频率。
        for pair in zip(tokens, tokens[1:]):
            pair_freqs[pair] = pair_freqs.get(pair, 0) + delta
            if pair_freqs[pair] == 0:
                del pair_freqs[pair]

        # 索引只记录一个词是否包含该 pair，因此先去重。
        for pair in set(zip(tokens, tokens[1:])):
            if delta > 0:
                pair_to_words[pair].add(tokens)
            else:
                pair_to_words[pair].remove(tokens)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

    # 全量统计只在训练开始时执行一次。
    for tokens, freq in counts.items():
        update_pairs(tokens, freq)

    for step in range(num_merges):
        if not pair_freqs:
            break
         
        best_pair = max(
            pair_freqs,
            key=lambda pair: (pair_freqs[pair], pair),
        )

        merges.append(best_pair)
        mergedtoken = best_pair[0] + best_pair[1]

        # 更新过程中会修改索引，先保存本轮受影响的词。
        for tokens in list(pair_to_words[best_pair]):
            freq = counts.pop(tokens)
            update_pairs(tokens, -freq)

            i = 0
            new_tokens = []
            while i < len(tokens):
                if tokens[i:i + 2] == best_pair:
                    new_tokens.append(mergedtoken)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens = tuple(new_tokens)
            counts[new_tokens] = counts.get(new_tokens, 0) + freq
            update_pairs(new_tokens, freq)

        vocab[len(vocab)] = mergedtoken

    return vocab, merges

    #assert NotImplemented















