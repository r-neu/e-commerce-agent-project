from __future__ import annotations

import argparse
import json
import random
from typing import Iterator


def reservoir_sample(iterable: Iterator[str], sample_size: int, seed: int) -> list[str]:
    """Reservoir sampling algorithm for random selection."""
    random.seed(seed)
    reservoir = []
    for i, line in enumerate(iterable):
        if i < sample_size:
            reservoir.append(line)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = line
    return reservoir


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample rows from JSONL")
    parser.add_argument("--input", default="data/sft_amazon_qa_organized.jsonl")
    parser.add_argument("--output", default="data/sft_amazon_qa_organized_filtered.jsonl")
    parser.add_argument("--sample-size", type=int, default=48000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Read and sample
    with open(args.input, "r", encoding="utf-8") as f:
        lines = (line for line in f if line.strip())
        sampled = reservoir_sample(lines, args.sample_size, args.seed)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line)

    print(f"Sampled {len(sampled)} rows from {args.input} → {args.output}")


if __name__ == "__main__":
    main() 