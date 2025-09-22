"""
Generate reject queries using synthetic generation.
"""
import json
import time
import random
import math
from typing import Dict, List, Any, Tuple
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
    if c.get("features_all"):
        pf = set(p.get("features") or [])
        if not c["features_all"].issubset(pf):
            return False
    if c.get("min_rating") is not None and p.get("avg_rating") is not None:
        if p["avg_rating"] < c["min_rating"]:
            return False
    return True


def _evaluate(ctx: List[Dict[str, Any]], cons: Dict[str, Any]) -> List[str]:
    return [p["product_id"] for p in ctx if _matches(p, cons)]


def _cooccur_map(ctx: List[Dict[str, Any]]) -> Counter:
    pairs = Counter()
    for p in ctx:
        feats = list(set(p.get("features") or []))
        feats.sort()
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                pairs[(feats[i], feats[j])] += 1
    return pairs


def _closest_products(ctx: List[Dict[str, Any]], desired_features: List[str], desired_category: str, desired_price: float, k: int = 2) -> List[Dict[str, Any]]:
    def score(p: Dict[str, Any]) -> Tuple[float, float, float]:
        feats = set(p.get("features") or [])
        overlap = len(set(desired_features) & feats)
        cat_bonus = 1.0 if p.get("category") == desired_category else 0.0
        price = p.get("price") or 0.0
        price_dist = abs(price - desired_price)
        # Higher is better for overlap and category; lower for price distance
        # Use negative distance so higher total score is better
        return (overlap, cat_bonus, -price_dist)

    scored = sorted(ctx, key=score, reverse=True)
    # Filter out exact duplicates if any
    unique = []
    seen = set()
    for p in scored:
        pid = p.get("product_id")
        if pid and pid not in seen:
            unique.append(p)
            seen.add(pid)
        if len(unique) >= k:
            break
    return unique


def _render_query(item_phrase: str, cons: Dict[str, Any], rng: random.Random) -> str:
    styles = [
        "I'm hunting for {item} under ${price}{brand_clause}{feat_clause}{rating_clause}",
        "Do you have any {item}{feat_clause} for less than ${price}{brand_clause}{rating_clause}?",
        "Looking for {item}. Budget around ${price}{brand_clause}{feat_clause}{rating_clause}",
        "I need {item}{feat_clause}. Price cap is ${price}{brand_clause}{rating_clause}",
        "Could you recommend {item}{feat_clause} under ${price}{brand_clause}{rating_clause}?",
    ]
    feat_clause = ""
    if cons.get("features_all"):
        feats = list(cons["features_all"])
        feats.sort()
        if len(feats) == 1:
            feat_clause = f" with {feats[0]}"
        else:
            feat_clause = f" with {', '.join(feats[:-1])} and {feats[-1]}"
    brand_clause = f" from {cons['brand']}" if cons.get("brand") else ""
    rating_clause = ""
    if cons.get("min_rating") is not None and cons["min_rating"] <= 5.0:
        rating_clause = f" with rating at least {cons['min_rating']:.1f}"
    template = rng.choice(styles)
    return template.format(item=item_phrase, price=math.floor(cons.get("max_price") or 0), feat_clause=feat_clause, brand_clause=brand_clause, rating_clause=rating_clause).strip() + "."


def _derive_item_phrase(ctx: List[Dict[str, Any]], rng: random.Random) -> str:
    # Pick a random product title and clean it up
    candidates = [p.get("title") for p in ctx if p.get("title")]
    if not candidates:
        # Fallback to a generic but still concrete phrase
        return "blender"
    title = rng.choice(candidates)
    # Titles look like 'Pyrex Blender - Kitchen Appliances'
    # Take the part before ' - '
    core = title.split(" - ")[0].strip()
    # Remove brand prefix if present (first word matches a brand in context)
    brands = {p.get("brand") for p in ctx if p.get("brand")}
    parts = core.split()
    if parts and parts[0] in brands and len(parts) > 1:
        core = " ".join(parts[1:])
    # Lowercase common shopping phrasing
    phrase = core.lower()
    return phrase


def _render_reject_reply(closest: List[Dict[str, Any]], query: str, rng: random.Random) -> str:
    openers = [
        "Sorry, we don't currently have an exact match for that.",
        "I couldn't find an exact match for your request right now.",
        "Looks like we're out of anything that fits those exact specs.",
        "I can't find a perfect match at the moment.",
        "We don't have that exact configuration in stock right now.",
    ]
    segways = [
        "However, here are the closest options I recommend:",
        "That said, these are the closest fits you might like:",
        "Meanwhile, these come closest to what you're after:",
        "In the meantime, consider these near-matches:",
        "Here are  some great alternatives that come close:",
    ]

    def product_line(p: Dict[str, Any]) -> str:
        title = p.get("title", "Product")
        brand = p.get("brand", "Brand")
        price = p.get("price", 0.0)
        rating = p.get("avg_rating", 0.0)
        feats = p.get("features", []) or []
        if feats:
            if len(feats) == 1:
                feat_text = f"features {feats[0]}"
            elif len(feats) == 2:
                feat_text = f"features {feats[0]} and {feats[1]}"
            else:
                feat_text = f"features {', '.join(feats[:-1])}, and {feats[-1]}"
        else:
            feat_text = "is a solid option"
        return f"- {title} by {brand} (${price:.2f}, {rating:.1f} stars), which {feat_text}."

    opener = rng.choice(openers)
    segway = rng.choice(segways)
    lines = [opener, segway]
    for p in closest:
        lines.append(product_line(p))
    return "\n".join(lines)


def create_reject_query(product_context: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """
    Generate a reject query that is decidable using ONLY the given product_context
    and has ZERO satisfying products, while recommending 1-2 closest alternatives.
    """
    rng = _choose_rng(seed)

    # Analyze context
    freq = _feature_freq(product_context)
    all_feats = list(freq.keys())
    cooccurs = _cooccur_map(product_context)

    # Baseline stats
    prices = [p.get("price", 0.0) for p in product_context if p.get("price") is not None]
    ratings = [p.get("avg_rating", 0.0) for p in product_context if p.get("avg_rating") is not None]
    min_price = min(prices) if prices else 0.0
    max_rating = max(ratings) if ratings else 5.0

    # Start with constraints likely to exclude all
    cons = {
        "max_price": None,
        "brand": None,
        "features_all": set(),
        "min_rating": None,
    }

    category = product_context[0].get("category", "product")

    # Strategy 1: require two features that never co-occur in any single product
    impossible_pairs = []
    if len(all_feats) >= 2:
        # Build set of co-occurring pairs present
        present_pairs = set(cooccurs.keys())
        # Consider random pairs from feature universe; choose ones not present
        candidates = []
        feats_sorted = sorted(set(all_feats))
        for i in range(len(feats_sorted)):
            for j in range(i + 1, len(feats_sorted)):
                pair = (feats_sorted[i], feats_sorted[j])
                if pair not in present_pairs:
                    candidates.append(pair)
        if candidates:
            impossible_pairs = candidates
    
    if impossible_pairs:
        pair = rng.choice(impossible_pairs)
        cons["features_all"] = set(pair)
    else:
        # Strategy 2: set price below minimum to exclude all
        cons["max_price"] = max(0.0, min_price - rng.uniform(5, 30))

    # Strategy 3: optionally add rating floor above max to exclude
    if rng.random() < 0.5:
        cons["min_rating"] = max_rating + rng.uniform(0.1, 0.5)

    # Ensure zero matches by tightening if needed
    for _ in range(6):
        sids = _evaluate(product_context, cons)
        if len(sids) == 0:
            break
        # Tighten further
        if cons.get("max_price") is None:
            cons["max_price"] = max(0.0, min_price - rng.uniform(1, 20))
            continue
        if cons.get("min_rating") is None and ratings:
            cons["min_rating"] = max_rating + rng.uniform(0.1, 0.5)
            continue
        # Add a third feature that makes it impossible
        available = [f for f in all_feats if f not in cons["features_all"]]
        if available:
            cons["features_all"].add(rng.choice(available))
        else:
            # As a last resort, set a brand not present in context
            brands = [p.get("brand") for p in product_context if p.get("brand")]
            brand_set = set(brands)
            fake_brands = ["Hamilton Beach", "Ninja", "Instant Pot", "Breville", "Calphalon"]
            cons["brand"] = rng.choice([b for b in fake_brands if b not in brand_set] or fake_brands)

    # Final assert: zero matches
    sids = _evaluate(product_context, cons)
    if len(sids) != 0:
        # Force reject by price + rating
        cons["max_price"] = max(0.0, min_price - 0.01)
        cons["min_rating"] = (max_rating + 0.1)
        sids = _evaluate(product_context, cons)

    # Construct query text using a realistic item phrase from titles
    max_price_for_text = cons.get("max_price")
    if max_price_for_text is None:
        # Put a reasonable price ask for the query rendering
        max_price_for_text = math.floor(min_price)
        cons["max_price"] = max_price_for_text
    item_phrase = _derive_item_phrase(product_context, rng)
    query = _render_query(item_phrase, cons, rng)

    # Choose 1-2 closest items to recommend
    desired_features = list(cons.get("features_all") or [])
    desired_price = float(cons.get("max_price") or 0.0)
    closest = _closest_products(product_context, desired_features, product_context[0].get("category", ""), desired_price, k=2)

    # Build reply text
    reply = _render_reject_reply(closest, query, rng)

    return {
        "task": "reject",
        "query": query,
        "product_context": product_context,
        "predicted_ids": [],
        "reply": reply
    }


def generate_reject_queries_fast(total_queries: int, seed: int, output_file: str) -> None:
    start_time = time.time()

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(total_queries):
            query_seed = seed + i
            
            context_data = build_product_context(query_seed)
            product_context = context_data["products"]
            
            query_data = create_reject_query(product_context, query_seed)
            
            f.write(json.dumps(query_data, ensure_ascii=False) + '\n')
            f.flush()
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_queries - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{total_queries} generated. Rate: {rate:.1f}/s. ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Generation completed in {total_time:.2f} seconds")
    print(f"Average time per query: {total_time / total_queries:.3f} seconds")
    print(f"All results saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reject queries using synthetic generation")
    parser.add_argument("--output", type=str, default="data/amazon_review_reject_combined.jsonl",
                       help="Output file path")
    parser.add_argument("--max-rows", type=int, default=14400,
                       help="Number of queries to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Generating {args.max_rows} reject queries")
    print(f"Base seed: {args.seed}")
    print(f"Output file: {args.output}")
    
    try:
        generate_reject_queries_fast(args.max_rows, args.seed, args.output)
        
        print("\nQuery generation completed successfully!")
        print(f"Total queries generated: {args.max_rows}")
        
        print("\nSample queries from output file:")
        with open(args.output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                query_data = json.loads(line.strip())
                print(f"  {i+1}. Query: {query_data['query']}")
                print(f"     Predicted IDs: {query_data['predicted_ids']}")
                print(f"     Reply: {query_data['reply'].splitlines()[0]}")
                print()
    except Exception as e:
        print(f"Error generating queries: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
