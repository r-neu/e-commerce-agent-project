"""
Generate clarify queries.
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


def _derive_item_phrase(ctx: List[Dict[str, Any]], rng: random.Random) -> str:
    candidates = [p.get("title") for p in ctx if p.get("title")]
    if not candidates:
        return "blender"
    title = rng.choice(candidates)
    core = title.split(" - ")[0].strip()
    brands = {p.get("brand") for p in ctx if p.get("brand")}
    parts = core.split()
    if parts and parts[0] in brands and len(parts) > 1:
        core = " ".join(parts[1:])
    phrase = core.lower()
    return phrase


def _context_diversity(ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    brands = [p.get("brand") for p in ctx if p.get("brand")]
    prices = [p.get("price") for p in ctx if p.get("price") is not None]
    ratings = [p.get("avg_rating") for p in ctx if p.get("avg_rating") is not None]
    feats = _feature_freq(ctx)
    return {
        "brand_set": set(brands),
        "min_price": min(prices) if prices else 0.0,
        "max_price": max(prices) if prices else 0.0,
        "min_rating": min(ratings) if ratings else 0.0,
        "max_rating": max(ratings) if ratings else 0.0,
        "feature_freq": feats,
    }


def _pick_common_feature(feat_freq: Counter, min_count: int = 2) -> str:
    candidates = [f for f, c in feat_freq.items() if c >= min_count]
    if not candidates:
        return ""
    return max(candidates, key=lambda f: feat_freq[f])


def _render_ambiguous_query(item_phrase: str, ambiguity: Dict[str, Any], rng: random.Random) -> str:
    styles = [
        "I'm shopping for a {item}{feat_hint}. My budget is flexible.{brand_hint}",
        "Looking for a {item}{feat_hint}. What's a good option?{brand_hint}",
        "Need a {item}{feat_hint}. Hoping to keep it affordable.{brand_hint}",
        "I'm after a {item}{feat_hint}. Not sure about price yet.{brand_hint}",
        "Can you suggest a {item}{feat_hint}? I can adjust the budget.{brand_hint}",
    ]
    feat_hint = ""
    if ambiguity.get("feature_name"):
        # leave preference unspecified to force clarify
        feat_hint = f" with {ambiguity['feature_name']} (if that's worth it)"
    brand_hint = ""
    if ambiguity.get("brand_choice_needed"):
        brand_hint = " Any brand is fine, I think."
    template = rng.choice(styles)
    return template.format(item=item_phrase, feat_hint=feat_hint, brand_hint=brand_hint).strip()


def _build_ambiguity(ctx: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    div = _context_diversity(ctx)
    needs_budget = (div["max_price"] - div["min_price"]) >= 80.0
    feature_name = _pick_common_feature(div["feature_freq"], min_count=2)
    needs_feature_choice = bool(feature_name)
    brands = div["brand_set"]
    needs_brand_choice = len(brands) >= 2

    # Choose 1-2 ambiguous slots that actually matter for this context
    slots = []
    if needs_budget:
        slots.append("budget")
    if needs_feature_choice:
        slots.append("feature")
    if needs_brand_choice:
        slots.append("brand")

    rng.shuffle(slots)
    chosen = slots[: rng.randint(1, min(2, len(slots)) or 1)]

    return {
        "budget_needed": "budget" in chosen,
        "feature_name": feature_name if "feature" in chosen else "",
        "brand_choice_needed": "brand" in chosen,
        "div": div,
    }


def _render_clarify_reply(ambiguity: Dict[str, Any], rng: random.Random) -> str:
    openings = [
        "Happy to help! A couple of quick questions so I can dial this in:",
        "Got it. To recommend the best fit, could you clarify a few things?",
        "I can narrow this down. Mind answering a couple of quick questions?",
        "Thanks! A few details will help me tailor the options:",
    ]
    qs: List[str] = []

    # Context stats
    div = ambiguity["div"]
    price_low = float(div["min_price"]) if div["min_price"] else 0.0
    price_high = float(div["max_price"]) if div["max_price"] else 0.0
    price_span = price_high - price_low
    rating_low = float(div["min_rating"]) if div["min_rating"] else 0.0
    rating_high = float(div["max_rating"]) if div["max_rating"] else 0.0
    rating_span = rating_high - rating_low

    # Budget question only if price spread exists
    if ambiguity.get("budget_needed") and price_span >= 20.0:
        mid = math.floor((price_low + price_high) / 2) if price_high else 100
        qs.append(f"- Do you have a rough budget cap (e.g., under ${mid})?")

    # Feature must-have question only if feature is present in context
    if ambiguity.get("feature_name"):
        fname = ambiguity["feature_name"]
        qs.append(f"- Is {fname} a must-have, a nice-to-have, or not needed?")

    # Brand preference question only if multiple brands
    asked_brand = False
    if ambiguity.get("brand_choice_needed") and len(div["brand_set"]) >= 2:
        qs.append("- Any brands you prefer or want to avoid?")
        asked_brand = True

    # Derive extra questions from context features for relevance
    feat_freq: Counter = div["feature_freq"]
    features_lower = {f.lower(): c for f, c in feat_freq.items()}

    def has_any(keys: List[str]) -> bool:
        return any(any(k in f for k in keys) for f in features_lower.keys())

    extras_pool: List[str] = []
    if has_any(["compact", "mini", "portable", "capacity", "large", "xl", "capacity"]):
        extras_pool.append("- Do you lean toward compact size or larger capacity?")
    if has_any(["stainless", "steel", "glass", "plastic", "ceramic"]):
        extras_pool.append("- Any preference on material (e.g., stainless steel, glass, plastic)?")
    if has_any(["easy clean", "dishwasher", "non-stick", "removable"]):
        extras_pool.append("- Is ease of cleaning important for you?")
    if has_any(["programmable", "smart", "preset", "timer", "multi-speed", "variable speed", "speed settings"]):
        extras_pool.append("- Do you prefer simpler controls or more advanced features?")
    if has_any(["quiet", "low noise"]):
        extras_pool.append("- Is quiet operation important, or is noise level not a concern?")
    if has_any(["lightweight", "heavy-duty"]):
        extras_pool.append("- Do you want something lightweight or heavy-duty?")

    rng.shuffle(extras_pool)
    for _ in range(3):
        if len(qs) >= 3:
            break
        if extras_pool:
            qs.append(extras_pool.pop())
        else:
            break

    # Only add generic tradeoff if the context supports a meaningful tradeoff
    if len(qs) < 3 and (price_span >= 20.0 or rating_span >= 0.4):
        qs.append("- Are you prioritizing lower price, higher rating, or specific features?")

    # Fallback brand openness only if brand not already asked and multiple brands exist
    if len(qs) < 3 and not asked_brand and len(div["brand_set"]) >= 2:
        qs.append("- Would you prefer a well-known brand or are you open to lesser-known ones?")

    # Ensure at least 2 questions
    if len(qs) < 2 and price_span >= 20.0:
        mid = math.floor((price_low + price_high) / 2) if price_high else 100
        if all("budget cap" not in q for q in qs):
            qs.append(f"- Do you have a rough budget cap (e.g., under ${mid})?")
    if len(qs) < 2 and rating_span >= 0.4:
        if all("prioritizing" not in q for q in qs):
            qs.append("- Are you prioritizing lower price, higher rating, or specific features?")

    # Truncate to max 3 questions
    qs = qs[:3]

    opening = rng.choice(openings)
    closing = rng.choice([
        "Once I have these, I'll shortlist the top matches.",
        "With that, I can recommend spot-on options.",
        "That'll help me zero in on the perfect picks.",
    ])

    return "\n".join([opening, *qs, closing])


def create_clarify_query(product_context: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    rng = _choose_rng(seed)
    item_phrase = _derive_item_phrase(product_context, rng)
    ambiguity = _build_ambiguity(product_context, rng)

    # If context has low diversity, still produce generic ambiguity that makes sense
    if not (ambiguity["budget_needed"] or ambiguity["feature_name"] or ambiguity["brand_choice_needed"]):
        ambiguity["budget_needed"] = True
        # pick any feature present
        any_feat = next(iter(_feature_freq(product_context).keys()), "")
        ambiguity["feature_name"] = any_feat

    query = _render_ambiguous_query(item_phrase, ambiguity, rng)
    reply = _render_clarify_reply(ambiguity, rng)

    return {
        "task": "clarify",
        "query": query,
        "product_context": product_context,
        "reply": reply,
    }


def generate_clarify_queries_fast(total_queries: int, seed: int, output_file: str) -> None:
    start_time = time.time()

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(total_queries):
            query_seed = seed + i
            context_data = build_product_context(query_seed)
            product_context = context_data["products"]

            data = create_clarify_query(product_context, query_seed)
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
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
    parser = argparse.ArgumentParser(description="Generate clarify queries")
    parser.add_argument("--output", type=str, default="data/amazon_review_clarify_combined.jsonl",
                       help="Output file path")
    parser.add_argument("--max-rows", type=int, default=3600,
                       help="Number of queries to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    
    args = parser.parse_args()
    
    print(f"Generating {args.max_rows} clarify queries")
    print(f"Base seed: {args.seed}")
    print(f"Output file: {args.output}")

    try:
        generate_clarify_queries_fast(args.max_rows, args.seed, args.output)
        print("\nQuery generation completed")
        print(f"Total queries generated: {args.max_rows}")

        print("\nSample queries from output file:")
        with open(args.output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                row = json.loads(line.strip())
                print(f"  {i+1}. Query: {row['query']}")
                print(f"     Reply: {row['reply'].splitlines()[0]}")
                print()
    except Exception as e:
        print(f"Error generating queries: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
