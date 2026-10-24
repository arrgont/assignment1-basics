import time
from pathlib import Path
import lib
import resource
import pickle

from cs336_basics.BPE_training import train_bpe
from tests.conftest import vocab_size

start = time.perf_counter()

vocab, merges = train_bpe(
    input_path=str("/Users/zyyy/Desktop/assignment1-basics/data/owt_train.txt"),
    vocab_size=32000,
    special_tokens=["<|endoftext|>"],
)

training_seconds = time.perf_counter() - start

peak_memory_gb = (
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000000
)

print(f"Training time: {training_seconds:.2f} seconds")
print(f"Peak process memory: {peak_memory_gb:.3f} GB")
print(f"Vocabulary size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")

assert len(vocab) <= 32000
assert b"<|endoftext|>" in vocab.values()

with open("/Users/zyyy/Desktop/assignment1-basics/data/owt_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("/Users/zyyy/Desktop/assignment1-basics/data/owt_merges.pkl", "wb") as f:
    pickle.dump(merges, f)


longest_id, longest_token = max(
    vocab.items(),
    key=lambda item: len(item[1]),
)

print("Longest token ID:", longest_id)
print("Length in bytes:", len(longest_token))
print("Raw bytes:", repr(longest_token))
print("Readable:", longest_token.decode("utf-8", errors="replace"))