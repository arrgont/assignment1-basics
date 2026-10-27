import time
import statistics

from cs336_basics import tokenizer
from cs336_basics.tokenizer_experiments import ts_tokenizer, ts_documents, owt_tokenizer, owt_documents


def measure_throughput(
    tokenizer,
    documents,
    repeat=3,
    min_seconds=1.0,
):
    bytes_per_round = sum(
        len(document.encode("utf-8"))
        for document in documents
    )

    for document in documents:
        tokenizer.encode(document)

    rates = []

    for _ in range(repeat):
        rounds = 0
        elapsed = 0.0
        start = time.perf_counter()

        while elapsed < min_seconds:
            for document in documents:
                tokenizer.encode(document)

            rounds += 1
            elapsed = time.perf_counter() - start

        processed_bytes = bytes_per_round * rounds
        rates.append(processed_bytes / elapsed)

    return statistics.median(rates)


for name, tokenizer, documents in [
    ("TinyStories", ts_tokenizer, ts_documents),
    ("OpenWebText", owt_tokenizer, owt_documents),
]:
    speed = measure_throughput(tokenizer, documents)

    pile_bytes = 825 * 10**9
    estimated_seconds = pile_bytes / speed

    print(name)
    print(f"Throughput: {speed:,.0f} bytes/second")
    print(f"Throughput: {speed / 10 ** 6:.3f} MB/second")
    print(
        f"Estimated Pile time: "
        f"{estimated_seconds / 3600:.2f} hours"
    )
    print(
        f"Estimated Pile time: "
        f"{estimated_seconds / 86400:.2f} days"
    )
