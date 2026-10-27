from itertools import islice
import numpy as np
from pathlib import Path

from cs336_basics.tokenizer_experiments import iter_documents, owt_tokenizer


PROJECT = Path("/Users/zyyy/Desktop/assignment1-basics")
DATA = PROJECT / "data"

def encode_dataset(
        tokenizer,
        input_path,
        output_path,
        batch_size=262144,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    doucuments = iter_documents(input_path)
    id_stream = tokenizer.encode_iterable(doucuments)

    total_tokens = 0
    batches = 0

    with open(output_path, "wb") as f:
        while True:
            ids = list(islice(id_stream, batch_size))

            if not ids:
                break

            array = np.asarray(ids, dtype=np.uint16)
            array.tofile(f)

            total_tokens += array.size
            batches += 1

            if batches % 20 == 0:
                print(
                    f"{input_path.name}: "
                    f"{total_tokens:,} tokens written",
                    flush=True,
                )

    expected_bytes = total_tokens * np.dtype(np.uint16).itemsize
    assert output_path.stat().st_size == expected_bytes

    print(
        f"Finished {output_path.name}: "
        f"{total_tokens:,} tokens, "
        f"{expected_bytes:,} bytes"
    )

    return total_tokens



jobs = [
    (
        owt_tokenizer,
        DATA / "owt_valid.txt",
        DATA / "owt_valid.bin",
    ),
    (
        owt_tokenizer,
        DATA / "owt_train.txt",
        DATA / "owt_train.bin",
    ),
]

for tokenizer, input_path, output_path in jobs:
    encode_dataset(
        tokenizer,
        input_path,
        output_path,
    )
