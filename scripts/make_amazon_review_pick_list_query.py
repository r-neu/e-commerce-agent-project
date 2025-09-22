"""
Generate pick_list queries.
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


def generate_diverse_human_query(seed: int) -> str:
    rng = _choose_rng(seed)
    
    # Different query styles for variety
    query_styles = [
        "I'm looking for {item} {rest}.",
        "Can you show me some {item} {rest}?",
        "I need {item} {rest}",
        "What {item} do you suggest {rest}?",
        "I want to compare {item} {rest}.",
        "Show me different {item} {rest}.",
        "I'm shopping for {item} {rest} Can you recommend a few?",
        "Looking for {item} alternatives {rest}.",
        "I need to choose between {item} {rest}.",
        "Can I see some {item} {rest}?",
    ]
    
    return rng.choice(query_styles)


def _derive_item_phrase(ctx: List[Dict[str, Any]], rng: random.Random) -> str:
    candidates = [p.get("title") for p in ctx if p.get("title")]
    if not candidates:
        return "product"
    title = rng.choice(candidates)
    core = title.split(" - ")[0].strip()
    brands = {p.get("brand") for p in ctx if p.get("brand")}
    parts = core.split()
    if parts and parts[0] in brands and len(parts) > 1:
        core = " ".join(parts[1:])
    return core.lower()


def _render_query_with_style(item_phrase: str, cons: Dict[str, Any], rng: random.Random, style_template: str) -> str:
    frags = []
    if cons.get("max_price") is not None:
        frags.append(f"under ${math.floor(cons['max_price'])}")
    if cons.get("brand"):
        frags.append(f"from {cons['brand']}")
    if cons.get("features"):
        frags.append("with " + ", ".join(sorted(cons["features"])) )
    if cons.get("min_rating") is not None:
        frags.append(f"rating at least {cons['min_rating']:.1f}")

    rng.shuffle(frags)
    if frags:
        if len(frags) == 1:
            rest = frags[0]
        else:
            rest = ", ".join(frags[:-1]) + f" and {frags[-1]}"
        rest = rest.strip()
        if not (rest.startswith("under") or rest.startswith("from") or rest.startswith("with") or rest.startswith("rating")):
            rest = "with " + rest
        rest = " " + rest
    else:
        rest = ""

    q = style_template.format(item=item_phrase, rest=rest).strip()
    if not q.endswith("?") and not q.endswith("."):
        if style_template.endswith("?"):
            q += "?"
        else:
            q += "."
    return q


def generate_shopping_assistant_pick_list_reply(satisfying_products: List[Dict[str, Any]], query: str, seed: int) -> str:
    rng = _choose_rng(seed)
    
    # Different reply templates for variety
    templates = [
        "I found {count} great options that match your requirements! {product_recommendations}",
        
        "Here are {count} excellent choices for you: {product_recommendations}",
        
        "Perfect! I've identified {count} products that fit your criteria: {product_recommendations}",
        
        "Great news! I found {count} options that meet your needs: {product_recommendations}",
        
        "I've got {count} solid recommendations for you: {product_recommendations}",
        
        "Here's what I recommend from your options: {product_recommendations}",
        
        "I found {count} products that should work well for you: {product_recommendations}",
        
        "Let me suggest {count} great choices: {product_recommendations}",
        
        "I've picked out {count} options that fit your requirements: {product_recommendations}",
        
        "Here are {count} products I think you'll like: {product_recommendations}"
    ]
    
    # Select random template
    template = rng.choice(templates)
    
    # Sort products by price for logical comparison
    sorted_products = sorted(satisfying_products, key=lambda x: x.get("price", 0))
    
    # Generate product recommendations text for all satisfying products
    product_texts = []
    for product in sorted_products:
        features = product.get("features", [])
        if features:
            if len(features) == 1:
                feature_text = f"features {features[0]}"
            elif len(features) == 2:
                feature_text = f"features {features[0]} and {features[1]}"
            else:
                feature_text = f"features {', '.join(features[:-1])}, and {features[-1]}"
        else:
            feature_text = "is a quality product"
        
        product_text = f"the {product.get('title', 'Product')} from {product.get('brand', 'Brand')} (${product.get('price', 0):.2f}, {product.get('avg_rating', 4.0):.1f} stars) which {feature_text}"
        product_texts.append(product_text)
    
    # Combine product recommendations
    if len(product_texts) == 2:
        product_recommendations = f"{product_texts[0]}, and {product_texts[1]}"
    else:
        product_recommendations = ", ".join(product_texts[:-1]) + f", and {product_texts[-1]}"
    
    reply = template.format(
        count=len(sorted_products),
        product_recommendations=product_recommendations
    )
    
    return reply


def create_pick_list_query(product_context: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """
    Generate a pick_list query that is decidable using ONLY the given product_context
    and has AT LEAST 2 satisfying products for meaningful comparison.
    """
    rng = _choose_rng(seed)
    
    # 1) Analyze product context to find common features and price ranges
    freq = _feature_freq(product_context)
    common_features = [f for f, count in freq.items() if count >= 2]
    
    # Get price range for reasonable constraints
    prices = [p.get("price", 0) for p in product_context if p.get("price") is not None]
    if prices:
        min_price = min(prices)
        max_price = max(prices)
        target_price = min_price + (max_price - min_price) * 0.7  # 70th percentile
    else:
        target_price = 200  # fallback
    
    # 2) Create constraints that will satisfy multiple products
    chosen_features = set()
    if common_features:
        # Pick 1-2 common features to ensure multiple matches
        num_features = rng.randint(1, min(2, len(common_features)))
        chosen_features = set(rng.sample(common_features, num_features))
    
    # Set price ceiling to include multiple products
    price_ceiling = target_price + rng.uniform(20, 100)
    
    # Brand constraint
    brand_constraint = None
    if rng.random() < 0.3:  # 30% chance to add brand constraint
        brands = [p.get("brand") for p in product_context if p.get("brand")]
        brand_counts = Counter(brands)
        # Only add brand if at least 2 products have it
        suitable_brands = [b for b, count in brand_counts.items() if count >= 2]
        if suitable_brands:
            brand_constraint = rng.choice(suitable_brands)
    
    # Rating constraint 
    rating_constraint = None
    if rng.random() < 0.4:  # 40% chance to add rating constraint
        ratings = [p.get("avg_rating", 0) for p in product_context if p.get("avg_rating") is not None]
        if ratings:
            min_rating = min(ratings)
            rating_constraint = min_rating + (max(ratings) - min_rating) * 0.3  # 30th percentile
    
    cons = {
        "max_price": price_ceiling,
        "brand": brand_constraint,
        "features": chosen_features,
        "min_rating": rating_constraint
    }
    
    # 3) Evaluate constraints and adjust if needed
    satisfying_ids = _evaluate(product_context, cons)
    
    # Ensure we have at least 2 products satisfying
    max_attempts = 10
    for _ in range(max_attempts):
        if len(satisfying_ids) >= 2:
            break
        
        if cons["min_rating"] is not None:
            cons["min_rating"] = None
        elif cons["brand"] is not None:
            cons["brand"] = None
        elif cons["features"]:
            if len(cons["features"]) > 1:
                cons["features"].discard(rng.choice(list(cons["features"])))
            else:
                cons["features"].clear()
        else:
            cons["max_price"] = cons["max_price"] + 100
        satisfying_ids = _evaluate(product_context, cons)
    
    if len(satisfying_ids) < 2:
        cons = {
            "max_price": price_ceiling + 200,
            "brand": None,
            "features": set(),
            "min_rating": None
        }
        satisfying_ids = _evaluate(product_context, cons)
    
    # 4) Generate natural language query
    item_phrase = _derive_item_phrase(product_context, rng)
    style_template = generate_diverse_human_query(seed)
    query = _render_query_with_style(item_phrase, cons, rng, style_template)
    
    # 5) Generate shopping assistant reply
    satisfying_products = [p for p in product_context if p["product_id"] in satisfying_ids]
    reply = generate_shopping_assistant_pick_list_reply(satisfying_products, query, seed)
    
    return {
        "task": "pick_list",
        "query": query,
        "product_context": product_context,
        "predicted_ids": satisfying_ids,
        "reply": reply
    }


def generate_pick_list_queries_fast(total_queries: int, seed: int, output_file: str) -> None:
    print(f"Generating {total_queries} pick_list queries")
    start_time = time.time()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(total_queries):
            query_seed = seed + i
            
            # Build product context (fresh context for each query)
            context_data = build_product_context(query_seed)
            product_context = context_data["products"]
            
            query_data = create_pick_list_query(product_context, query_seed)
            
            f.write(json.dumps(query_data, ensure_ascii=False) + '\n')
            f.flush()  
            
            if i % 100 == 0:  
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_queries - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{total_queries} queries generated. Rate: {rate:.1f} queries/sec. ETA: {eta:.1f} seconds")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Generation completed in {total_time:.2f} seconds")
    print(f"Average time per query: {total_time / total_queries:.3f} seconds")
    print(f"All results saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pick_list queries using synthetic generation")
    parser.add_argument("--output", type=str, default="data/amazon_review_pick_list_combined.jsonl",
                       help="Output file path")
    parser.add_argument("--max-rows", type=int, default=27000,
                       help="Number of queries to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Generating {args.max_rows} pick_list queries")
    print(f"Base seed: {args.seed}")
    print(f"Output file: {args.output}")
    
    try:
        generate_pick_list_queries_fast(args.max_rows, args.seed, args.output)
        
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
                print(f"     Reply: {query_data['reply'][:100]}...")
                print()
        
    except Exception as e:
        print(f"Error generating queries: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
