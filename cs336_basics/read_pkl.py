import pickle

path = "/Users/zyyy/Desktop/assignment1-basics/data/tinystories_vocab.pkl"

with open(path, "rb") as f:
    vocab = pickle.load(f)

longest_id, longest_token = max(
    vocab.items(),
    key=lambda item: len(item[1]),
)
print("Token ID:", longest_id)
print("Length in bytes:", len(longest_token))
print("Raw bytes:", repr(longest_token))
print("Readable text:", longest_token.decode("utf-8", errors="replace"))
