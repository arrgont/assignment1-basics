import os
from typing import BinaryIO
import regex as re
from collections import Counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_pretoken(
        input_path: str,
        special_tokens:list[str]
) -> dict[tuple[bytes, ...], int]:
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 4
        counts = {}
        word_counts = Counter()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        special_pattern = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            if special_pattern:
                text_parts = re.split(special_pattern, chunk)
            else:
                text_parts = [chunk]
            for text in text_parts:
                for match in re.finditer(PAT, text):
                    word_counts[match.group(0)] += 1

        while word_counts:
            pre_token, frequency = word_counts.popitem()

            pre_tokens = tuple(
                bytes([b])
                for b in pre_token.encode("utf-8")
            )

            counts[pre_tokens] = frequency

        return counts




                    #pretoken = match.group(0)
                    #pretoken = tuple(
                    #    bytes([b])
                    #    for b in match.group(0).encode("utf-8")
                    #)
                    #counts[pretoken] = counts.get(pretoken, 0) + 1



                    #可优化：反复转换相同的词片段，可以先统计字符串出现次数然后再进行转换
                    # # 放在所有分块循环之外，让各块共用这份计数
                    # text_counts = {}
                    #
                    # # 在原来的正则匹配循环中
                    # for match in re.finditer(PAT, text):
                    #     token_text = match.group(0)  # 直接取字符串
                    #     text_counts[token_text] = text_counts.get(token_text, 0) + 1
                    #
                    # counts = {}
                    #
                    # for token_text, freq in text_counts.items():
                    #     pretoken = tuple(
                    #         bytes([b])
                    #         for b in token_text.encode("utf-8")
                    #     )
                    #     counts[pretoken] = freq

        #print(counts)
        return counts
