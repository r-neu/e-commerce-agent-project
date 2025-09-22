"""
Build a demo store.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Dict, Any


def load_product_library(file_path: str) -> List[Dict[str, Any]]:
    products = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                products.append(json.loads(line))
    return products


def select_random_products(products: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    return random.sample(products, min(count, len(products)))


def save_demo_products(products: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for product in products:
            f.write(json.dumps(product, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo store")
    parser.add_argument("--input", default="data/product_library.jsonl", 
                       help="Path to input product library file")
    parser.add_argument("--output", default="data/demo_store_products_100.jsonl",
                       help="Path to output demo store file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-rows", type=int, default=100, help="Number of products to select")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    products = load_product_library(args.input)

    if len(products) < args.max_rows:
        args.max_rows = len(products)

    selected_products = select_random_products(products, args.max_rows, args.seed)

    categories = {}
    for product in selected_products:
        category = product.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1

    print(f"Selected products by category:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    print(f"Saving demo store to: {args.output}")
    save_demo_products(selected_products, args.output)

if __name__ == "__main__":
    main() 