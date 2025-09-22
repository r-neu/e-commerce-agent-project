"""
Merge filtered QA and review datasets into final SFT file.
"""
import os

def main() -> None:
    qa_file = "data/sft_amazon_qa_organized_filtered.jsonl"
    review_file = "data/sft_amazon_review.jsonl"
    output_file = "data/sft_full_merged_final.jsonl"

    # Verify input files exist
    for f in [qa_file, review_file]:
        if not os.path.exists(f):
            print(f"Error: Input file not found: {f}")
            return

    # Count input lines
    qa_count = sum(1 for _ in open(qa_file, "r"))
    review_count = sum(1 for _ in open(review_file, "r"))
    print(f"Input counts: QA filtered={qa_count}, Review={review_count}, Total={qa_count + review_count}")

    # Concatenate
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        # Copy filtered QA data
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                out.write(line)
        # Copy review data
        with open(review_file, "r", encoding="utf-8") as f:
            for line in f:
                out.write(line)

    # Verify output count
    output_count = sum(1 for _ in open(output_file, "r"))
    print(f"Output: {output_count} rows → {output_file}")
    if output_count == qa_count + review_count:
        print("match")
    else:
        print("mismatch")


if __name__ == "__main__":
    main() 