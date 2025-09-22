import json
import random
from typing import Dict, List, Any

def load_product_library(file_path: str = "data/product_library.jsonl") -> List[Dict[str, Any]]:
    products = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                products.append(json.loads(line.strip()))
    return products

def get_categories_and_subcategories(products: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    category_map = {}
    for product in products:
        category = product.get("category", "")
        sub_category = product.get("sub_category", "")
        
        if category and sub_category:
            if category not in category_map:
                category_map[category] = set()
            category_map[category].add(sub_category)
    
    return {cat: list(subcats) for cat, subcats in category_map.items()}

def select_random_category_and_subcategory(category_map: Dict[str, List[str]]) -> tuple:
    if not category_map:
        raise ValueError("No categories available")
    
    selected_category = random.choice(list(category_map.keys()))
    selected_subcategory = random.choice(category_map[selected_category])
    
    return selected_category, selected_subcategory

def select_products_from_subcategory(products: List[Dict[str, Any]], 
                                   category: str, 
                                   sub_category: str, 
                                   count: int = 8) -> List[Dict[str, Any]]:
    matching_products = [
        product for product in products
        if product.get("category") == category and product.get("sub_category") == sub_category
    ]
    
    if len(matching_products) < count:
        # If not enough products, return all available
        return matching_products
    
    selected_products = random.sample(matching_products, count)
    
    return selected_products


def format_product_context(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted_products = []
    
    for product in products:
        formatted_product = {
            "product_id": product.get("product_id", ""),
            "category": product.get("sub_category", "").lower().replace(" ", "_"),
            "title": product.get("title", ""),
            "price": product.get("price", 0.0),
            "brand": product.get("brand", ""),
            "features": product.get("features", []),
            "avg_rating": product.get("avg_rating", 0.0)
        }
        formatted_products.append(formatted_product)
    
    return formatted_products


def build_product_context(seed: int = None) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    
    products = load_product_library()
    
    category_map = get_categories_and_subcategories(products)
    
    selected_category, selected_subcategory = select_random_category_and_subcategory(category_map)
    
    # Select 8 products from the subcategory
    selected_products = select_products_from_subcategory(
        products, selected_category, selected_subcategory, 8
    )
    
    formatted_products = format_product_context(selected_products)
    
    return {
        "category": selected_category,
        "subcategory": selected_subcategory,
        "products": formatted_products
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build product context")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    try:
        context = build_product_context(args.seed)
        
        print(f"Selected Category: {context['category']}")
        print(f"Selected Subcategory: {context['subcategory']}")
        print(f"Selected Products: {len(context['products'])}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, ensure_ascii=False)
            print(f"Context saved to {args.output}")
        else:
            print("\nProduct Context:")
            print(json.dumps(context, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error building product context: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
