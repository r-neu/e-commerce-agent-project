"""
Filter Amazon review data for e-commerce agent training.
"""

import argparse
import json
import random
import requests
import gzip
from typing import Dict, Any, List, Optional

TARGET_CATEGORIES = [
    "Home_and_Kitchen",
    "Grocery_and_Gourmet_Food", 
    "Arts_Crafts_and_Sewing",
    "Health_and_Household",
    "Tools_and_Home_Improvement"
]

# Category mapping from dataset categories to my target categories
CATEGORY_MAPPING = {
    # Home_and_Kitchen
    "home_and_kitchen": "Home_and_Kitchen",
    "kitchen_and_dining": "Home_and_Kitchen",
    "home_improvement": "Home_and_Kitchen",
    "patio_lawn_and_garden": "Home_and_Kitchen",
    
    # Grocery_and_Gourmet_Food
    "grocery_and_gourmet_food": "Grocery_and_Gourmet_Food",
    "food_and_beverage": "Grocery_and_Gourmet_Food",
    
    # Arts_Crafts_and_Sewing
    "arts_crafts_and_sewing": "Arts_Crafts_and_Sewing",
    "arts_and_crafts": "Arts_Crafts_and_Sewing",
    
    # Health_and_Household
    "health_and_household": "Health_and_Household",
    "beauty_and_personal_care": "Health_and_Household",
    "all_beauty": "Health_and_Household",
    "health_and_beauty": "Health_and_Household",
    "personal_care_appliances": "Health_and_Household",
    "baby_products": "Health_and_Household",
    
    # Tools_and_Home_Improvement
    "tools_and_home_improvement": "Tools_and_Home_Improvement",
    "automotive": "Tools_and_Home_Improvement",
    "industrial_and_scientific": "Tools_and_Home_Improvement",
    "tools": "Tools_and_Home_Improvement",
}

# Alternative URLs to try for the dataset
DATASET_URLS = [
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta.jsonl.gz",
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Home_and_Kitchen.jsonl.gz",
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz",
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Arts_Crafts_and_Sewing.jsonl.gz",
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Health_and_Household.jsonl.gz",
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Tools_and_Home_Improvement.jsonl.gz"
]


def try_download_dataset() -> Optional[List[Dict[str, Any]]]:
    all_items = []
    
    for url in DATASET_URLS:
        try:
            print(f"Trying to download from: {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            print(f"Downloaded from: {url}")
            
            # Process the gzipped content
            with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item_meta = json.loads(line.strip())
                        all_items.append(item_meta)
                        
                        if len(all_items) % 10000 == 0:
                            print(f"  Loaded {len(all_items)} items...")
                            
                        # Limit to prevent memory issues
                        if len(all_items) >= 1000000:  
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        if line_num % 10000 == 0:
                            print(f"Error processing line {line_num}: {e}")
                        continue
            
            print(f"Loaded {len(all_items)} items from {url}")
            return all_items
            
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    return None


def belongs_to_category(item_meta: Dict[str, Any], target_category: str) -> bool:
    main_category = item_meta.get("main_category", "").lower()
    mapped_category = CATEGORY_MAPPING.get(main_category, "")
    return mapped_category == target_category


def extract_product_metadata(item_meta: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
    try:
        # Extract sub-category from categories list
        categories = item_meta.get("categories", [])
        sub_category = ""
        if categories and len(categories) > 0:
            # Get the most specific category (last in the hierarchy)
            if isinstance(categories[0], list) and len(categories[0]) > 0:
                sub_category = categories[0][-1]  # Last item in the first category list
            elif isinstance(categories, list) and len(categories) > 0:
                sub_category = categories[-1] if isinstance(categories[-1], str) else str(categories[-1])
        
        # Extract features
        features = item_meta.get("features", [])
        if not isinstance(features, list):
            features = []
        
        # Extract price
        price = item_meta.get("price", None)
        if price == "None" or price is None:
            price = None
        else:
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = None
        
        # Extract brand from details
        brand = ""
        details = item_meta.get("details", {})
        if isinstance(details, dict):
            brand = details.get("Brand", details.get("brand", ""))
        elif isinstance(details, str):
            # Try to parse JSON string
            try:
                details_dict = json.loads(details)
                brand = details_dict.get("Brand", details_dict.get("brand", ""))
            except:
                brand = ""
        
        # Create product metadata
        product_meta = {
            "product_id": item_meta.get("parent_asin", ""),
            "category": category,
            "sub_category": sub_category,
            "title": item_meta.get("title", ""),
            "price": price,
            "brand": brand,
            "features": features,
            "avg_rating": item_meta.get("average_rating", None)
        }
        
        # Validate required fields
        if not product_meta["product_id"] or not product_meta["title"]:
            return None
            
        return product_meta
        
    except Exception as e:
        print(f"Error processing item metadata: {e}")
        return None


def sample_products_by_category(all_items: List[Dict[str, Any]], samples_per_category: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    
    # Initialize category counters
    category_counts = {cat: 0 for cat in TARGET_CATEGORIES}
    sampled_products = []
    
    print(f"Sampling {samples_per_category} products per category from {len(all_items)} total items...")
    
    # Shuffle items for random sampling
    random.shuffle(all_items)
    
    for item_meta in all_items:
        # Check if we have enough samples for all categories
        if all(count >= samples_per_category for count in category_counts.values()):
            break
        
        # Check which category this item belongs to
        for target_category in TARGET_CATEGORIES:
            if category_counts[target_category] >= samples_per_category:
                continue
                
            if belongs_to_category(item_meta, target_category):
                product_meta = extract_product_metadata(item_meta, target_category)
                
                if product_meta:
                    sampled_products.append(product_meta)
                    category_counts[target_category] += 1
                    
                    # Print progress
                    if sum(category_counts.values()) % 1000 == 0:
                        print(f"Sampled {sum(category_counts.values())} products so far...")
                        print(f"Category counts: {category_counts}")
                    break
    
    return sampled_products


def save_to_jsonl(products: List[Dict[str, Any]], output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        for product in products:
            f.write(json.dumps(product, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(products)} products to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Filter Amazon review data")
    parser.add_argument("--input", type=str, default="McAuley-Lab/Amazon-Reviews-2023",
                       help="Input dataset name")
    parser.add_argument("--output", type=str, default="data/product_library.jsonl",
                       help="Output product library file")
    parser.add_argument("--max-rows", type=int, default=108000,
                       help="Maximum number of products to extract")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    
    samples_per_category = args.max_rows // 5
    print(f"Extracting {args.max_rows} products total")
    print(f"Samples per category: {samples_per_category}")
    print(f"Random seed: {args.seed}")
    print(f"Output file: {args.output}")
    
    try:
        all_items = try_download_dataset()
        
        if all_items:
            products = sample_products_by_category(all_items, samples_per_category, args.seed)
        
        print(f"\nSaving {len(products)} products to file")
        save_to_jsonl(products, args.output)
        
        category_counts = {}
        for product in products:
            cat = product["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\nDataset creation completed")
        print(f"Total products: {len(products)}")
        print("Category distribution:")
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
