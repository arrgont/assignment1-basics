from pathlib import Path
import random

from cs336_basics.tokenizer import Tokenizer

PROJECT = Path("/Users/zyyy/Desktop/assignment1-basics")
DATA = PROJECT / "data"
EOS = "<|endoftext|>"


def iter_documents(filepath, read_chars = 1024*1024):
    buffer = ""

    with open(
        filepath,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        while True:
            chunk = f.read(read_chars)

            if not chunk:
                break

            buffer += chunk
            parts = buffer.split(EOS)

            buffer = parts.pop()

            for part in parts:
                yield part + EOS

        if buffer:
            yield buffer


def sample_documents(filepath, n=10, seed=22):
    rng = random.Random(seed)
    samples = []
    seen = 0

    for document in iter_documents(filepath):
        if not document.removesuffix(EOS).strip():
            continue

        seen += 1

        if len(samples) < n:
            samples.append(document)

        else:
            j = rng.randrange(seen)

            if j < n:
                samples[j] = document

    if len(samples) < n:
        raise ValueError(f"only have {len(samples)} not-empty documents")

    return samples


ts_documents = sample_documents(
    DATA / "TinyStoriesV2-GPT4-valid.txt",
    n=10,
    seed=22,
)

owt_documents = sample_documents(
    DATA / "owt_valid.txt",
    n=10,
    seed=22,
)

print("TinyStories documents:", len(ts_documents))
print("OpenWebText documents:", len(owt_documents))

ts_tokenizer = Tokenizer.from_files(
    vocab_filepath=str(DATA / "tinystories_vocab.pkl"),
    merges_filepath=str(DATA / "tinystories_merges.pkl"),
    special_tokens=[EOS],
)

assert len(ts_tokenizer.vocab) == 10000

owt_tokenizer = Tokenizer.from_files(
    vocab_filepath=str(DATA / "owt_vocab.pkl"),
    merges_filepath=str(DATA / "owt_merges.pkl"),
    special_tokens=[EOS],
)

assert len(owt_tokenizer.vocab) == 32000

def compression_stats(tokenizer,documents):
    total_bytes = 0
    total_tokens = 0

    for document in documents:
        tokens = tokenizer.encode(document)

        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokens)

    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
    }


ts_result = compression_stats(
    ts_tokenizer,
    ts_documents,
)

print("TinyStories:", ts_result)

owt_result = compression_stats(
    owt_tokenizer,
    owt_documents,
)

print("OpenWebText:", owt_result)

cross_result = compression_stats(
    ts_tokenizer,
    owt_documents,
)

print("TinyStories tokenizer on OWT:", cross_result)


