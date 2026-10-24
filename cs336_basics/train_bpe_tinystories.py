import time
#from pathlib import Path
#import lib
import resource
import pickle

from cs336_basics.BPE_training import train_bpe
#from tests.conftest import vocab_size

start = time.perf_counter()

vocab, merges = train_bpe(
    input_path=str("/Users/zyyy/Desktop/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"),
    vocab_size=10000,
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

assert len(vocab) <= 10000
assert b"<|endoftext|>" in vocab.values()

with open("/Users/zyyy/Desktop/assignment1-basics/data/tinystories_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("/Users/zyyy/Desktop/assignment1-basics/data/tinystories_merges.pkl", "wb") as f:
    pickle.dump(merges, f)