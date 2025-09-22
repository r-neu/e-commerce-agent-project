"""
Generate pick_one queries using synthetic generation.
"""

import json
import time
import random
import math
from typing import Dict, List, Any
from collections import Counter
from build_product_context import build_product_context


def _choose_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _feature_freq(ctx: List[Dict[str, Any]]) -> Counter:
    feats = []
    for p in ctx:
        feats.extend(p.get("features") or [])
    return Counter(feats)


def _matches(p: Dict[str, Any], c: Dict[str, Any]) -> bool:
    if c.get("max_price") is not None and p.get("price") is not None:
        if p["price"] > c["max_price"]:
            return False
    if c.get("brand") and p.get("brand") != c["brand"]:
        return False
    if c.get("features"):
        pf = set(p.get("features") or [])
        if not c["features"].issubset(pf):
            return False
    if c.get("min_rating") is not None and p.get("avg_rating") is not None:
        if p["avg_rating"] < c["min_rating"]:
            return False
    return True


def _evaluate(ctx: List[Dict[str, Any]], cons: Dict[str, Any]) -> List[str]:
    return [p["product_id"] for p in ctx if _matches(p, cons)]


def create_pick_one_unique_query(product_context: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """
    Generate a pick_one query that is decidable using ONLY the given product_context
    and has EXACTLY ONE satisfying product (the target).
    """
    rng = _choose_rng(seed)
    target = rng.choice(product_context)

    # 1) Prefer rare features to quickly isolate the target
    freq = _feature_freq(product_context)
    t_feats = [f for f in (target.get("features") or [])]
    rare_feats = [f for f in t_feats if freq[f] == 1]
    common_feats = [f for f in t_feats if freq[f] > 1]

    chosen_feats = set()
    if rare_feats:
        chosen_feats.add(rng.choice(rare_feats))
    elif common_feats:
        chosen_feats.add(rng.choice(common_feats))

    max_price = None
    if target.get("price") is not None:
        # ceiling slightly above target, so higher-priced items are excluded
        max_price = float(target["price"]) + rng.uniform(5, 25)

    cons = {
        "max_price": max_price,
        "brand": None,
        "features": chosen_feats,
        "min_rating": None
    }

    def satisfying_ids():
        return _evaluate(product_context, cons)

    # 2) Tighten until unique (or we hit a fallback)
    for _ in range(6):
        sids = satisfying_ids()
        if sids == [target["product_id"]]:
            break  # unique
        if len(sids) > 1:
            # Add another feature (prefer rare, else common)
            pool = [f for f in rare_feats if f not in cons["features"]] or \
                   [f for f in common_feats if f not in cons["features"]]
            if pool:
                cons["features"].add(rng.choice(pool))
                continue
            # Add brand if available
            if not cons["brand"] and target.get("brand"):
                cons["brand"] = target["brand"]
                continue
            # Tighten price ceiling just below the cheapest non-target candidate
            others = [p for p in product_context
                      if p["product_id"] in sids
                      and p["product_id"] != target["product_id"]
                      and p.get("price") is not None]
            if target.get("price") is not None and others:
                min_other = min(p["price"] for p in others)
                ceiling = min(min_other - 0.01, float(target["price"]) + 1e-3)
                if ceiling >= target["price"]:
                    cons["max_price"] = ceiling
                    continue
            # Raise rating floor above max non-target rating but ≤ target rating
            if target.get("avg_rating") is not None:
                non_t = [p for p in product_context
                        if p["product_id"] in sids
                        and p["product_id"] != target["product_id"]
                        and p.get("avg_rating") is not None]
                if non_t:
                    max_non_t = max(p["avg_rating"] for p in non_t)
                    new_min = min(target["avg_rating"], max_non_t + 1e-3)
                    if new_min <= target["avg_rating"]:
                        cons["min_rating"] = new_min
                        continue
            # Last resort: brand + all target features
            if target.get("brand"):
                cons["brand"] = target["brand"]
            if len(cons["features"]) < len(t_feats):
                cons["features"] = set(t_feats)
        else:
            # Zero matches (over-tightened): relax gently
            if cons["min_rating"] is not None:
                cons["min_rating"] = None
            elif cons["max_price"] is not None:
                cons["max_price"] = cons["max_price"] + 10.0

    # Final check; if still not unique, enforce strict fallback
    sids = satisfying_ids()
    if target["product_id"] not in sids or len(sids) != 1:
        cons["brand"] = target.get("brand")
        cons["features"] = set(t_feats)
        if target.get("price") is not None:
            cons["max_price"] = float(target["price"]) + 1e-3
        sids = satisfying_ids()

    # 3) Render natural-language query using ONLY CONTEXT fields
    parts = [f"Need a {target['category'].replace('_',' ')}"]
    if cons["max_price"] is not None:
        parts.append(f"under ${math.floor(cons['max_price'])}")
    if cons["brand"]:
        parts.append(f"from {cons['brand']}")
    if cons["features"]:
        parts.append("with " + ", ".join(sorted(cons["features"])))
    if cons["min_rating"] is not None:
        parts.append(f"with rating at least {cons['min_rating']:.1f}")
    query = " ".join(parts) + "."

    return {
        "task": "pick_one",
        "query": query,
        "product_context": product_context,
        "satisfying_ids": [target["product_id"]] if target["product_id"] in sids and len(sids) == 1 else sids
    }


def generate_pick_one_queries_fast(total_queries: int, seed: int) -> List[Dict[str, Any]]:
    queries = []
    
    print(f"Generating {total_queries} pick_one queries")
    start_time = time.time()
    
    for i in range(total_queries):
        query_seed = seed + i
        
        # Build product context (fresh context for each query)
        context_data = build_product_context(query_seed)
        product_context = context_data["products"]
        
        if i % 100 == 0:  
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_queries - i - 1) / rate if rate > 0 else 0
            print(f"Progress: {i+1}/{total_queries} queries generated. Rate: {rate:.1f} queries/sec. ETA: {eta:.1f} seconds")
        
        query_data = create_pick_one_unique_query(product_context, query_seed)
        queries.append(query_data)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Generation completed in {total_time:.2f} seconds")
    print(f"Average time per query: {total_time / total_queries:.3f} seconds")
    
    return queries


def save_to_jsonl(queries: List[Dict[str, Any]], output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(queries)} queries to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pick_one queries using synthetic generation")
    parser.add_argument("--output", type=str, default="data/amazon_review_pick_one_query.jsonl",
                       help="Output file path")
    parser.add_argument("--max-rows", type=int, default=18000,
                       help="Number of queries to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Generating {args.max_rows} pick_one queries")
    print(f"Base seed: {args.seed}")
    print(f"Output file: {args.output}")
    
    try:
        queries = generate_pick_one_queries_fast(args.max_rows, args.seed)
        
        print(f"\nSaving {len(queries)} queries to JSONL file...")
        save_to_jsonl(queries, args.output)
        
        print("\nQuery generation completed successfully!")
        print(f"Total queries generated: {len(queries)}")
        
        if queries:
            print("\nSample queries:")
            for i, query in enumerate(queries[:3]):
                print(f"  {i+1}. {query['query']}")
                print(f"     Satisfying IDs: {query['satisfying_ids']}")
        
    except Exception as e:
        print(f"Error generating queries: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
