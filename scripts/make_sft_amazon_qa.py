"""
Create SFT dataset for Amazon QA tasks.
"""
import argparse
import json
import random
from datasets import load_dataset
from typing import Dict, Any

def load_amazon_qa_dataset() -> Any:
    try:
        dataset = load_dataset("sentence-transformers/amazon-qa", split="train")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def sample_and_reformat(dataset: Any, num_samples: int, seed: int) -> list:
    random.seed(seed)
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")
    
    if num_samples > total_samples:
        print(f"Warning: Requested {num_samples} samples but only {total_samples} available. Using all samples.")
        num_samples = total_samples
    
    sampled_indices = random.sample(range(total_samples), num_samples)
    
    reformatted_samples = []
    for idx in sampled_indices:
        sample = dataset[idx]
        reformatted_sample = {
            "user": sample["query"],
            "assistant": sample["answer"]
        }
        reformatted_samples.append(reformatted_sample)
    
    return reformatted_samples


def save_to_jsonl(samples: list, output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(samples)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create SFT Amazon QA dataset")
    parser.add_argument("--output", type=str, default="data/sft_amazon-qa.jsonl",
                       help="Output SFT dataset file")
    parser.add_argument("--max-rows", type=int, default=96000,
                       help="Maximum number of samples to extract")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Creating SFT Amazon QA dataset with {args.max_rows} samples")
    print(f"Random seed: {args.seed}")
    print(f"Output file: {args.output}")
    
    try:
        dataset = load_amazon_qa_dataset()
        
        samples = sample_and_reformat(dataset, args.max_rows, args.seed)
        
        save_to_jsonl(samples, args.output)
        
        print("Dataset creation completed")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
