#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import numpy as np
import pickle
import re
from evaluation import ImprovedEvaluator

MODEL_PATH = "models/llama-8b-ecommerce-Q4_K_M.gguf"
PRODUCTS_PATH = "data/demo_store_products_100.jsonl"
EMBEDDINGS_PATH = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "product_embeddings.pkl")
METADATA_PATH = os.path.join(EMBEDDINGS_PATH, "product_metadata.pkl")
TEST_DATA_PATH = "data/product_test_queries.jsonl"
OUTPUT_PATH = "data/agent_test_results.jsonl"

# RAG Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 8
SIMILARITY_THRESHOLD = 0.1

# Configuration for my M1 Max
MODEL_CONFIG = {
    "n_ctx": 4096,
    "n_gpu_layers": -1,
    "n_threads": 8,
    "verbose": False,
    "use_mmap": True,
    "use_mlock": True,
}

SYSTEM_PROMPT = """You are a helpful, concise shopping assistant.
CRITICAL INSTRUCTIONS:
- NEVER make assumptions about what products exist or their prices
- If product_context is provided, analyze it carefully before responding, use the provided product_context to answer questions about products
- If no relevant products are found in the context, say so clearly
- Do not invent or guess product details
- Keep responses short, concise and complete - do not repeat yourself
- When you finish answering, STOP immediately - do not continue generating
- NEVER ask any questions - provide information only
- Do not say "Based on the provided product context" or "the provided product context" or things like that any time

QUERY HANDLING:
- GREETINGS, and user general reply like "Hi", "Hello", "Hey", "Good morning", "ok", etc: Only respond "Hi! I am here to help you with your shopping needs." Do not repeat yourself.
- PRODUCT SEARCHES: Use the provided product_context to recommend relevant products
- SHIPPING/DELIVERY QUESTIONS: Use [TOOL: get_shipping_speed] to get delivery information
- RETURN QUESTIONS: Use [TOOL: get_return_policy] to get return information
- For anything and anything you are not sure, only reply "I am here to help you with your shopping needs."

Please keep reply and answer in short. When you finish your response, STOP. Do not say "Based on the provided product context" or "the provided product context" or things like that any time.
For anything and anything you are not sure, only reply "I am here to help you with your shopping needs."

AVAILABLE TOOLS:
- get_shipping_speed: Get delivery options (standard/express/overnight) and delivery time
- get_return_policy: Get return window, condition, restocking fee, and notes

TOOL USAGE:
- If user asks about shipping speed/delivery time: respond first with [TOOL: get_shipping_speed]
- If user asks about returns/refunds: respond first with [TOOL: get_return_policy]
- After a tool result is provided, summarize it shortly and clearly for the user.
"""

class AgentTester:
    def __init__(self):
        self.llm = None
        self.evaluator = ImprovedEvaluator()
        self.embedding_model = None
        self.product_embeddings = None
        self.product_metadata = None
        self.products = []
        self.shipping_data = {}
        self.return_policy_data = {}
        
    def load_model(self):
        if self.llm is not None:
            return
        
        print("Loading finetuned Llama model")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        try:
            self.llm = Llama(
                model_path=MODEL_PATH,
                **MODEL_CONFIG
            )
            print("Model loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_products(self) -> List[Dict[str, Any]]:
        products = []
        if os.path.exists(PRODUCTS_PATH):
            with open(PRODUCTS_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        products.append(json.loads(line))
        return products
    
    def create_product_text(self, product: Dict[str, Any]) -> str:
        title = product.get('title','')
        brand = product.get('brand','')
        category = product.get('category','')
        sub_category = product.get('sub_category','')
        price = product.get('price', 0)
        features = ' '.join(product.get('features', []))
        return f"{title} {brand} {category} {sub_category} ${price} {features}"

    def load_embedding_model(self):
        if self.embedding_model is None:
            print("Loading BGE-M3 embedding model")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded")
    
    def build_embeddings(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            print("Loading existing embeddings")
            with open(INDEX_PATH, 'rb') as f:
                self.product_embeddings = pickle.load(f)
            with open(METADATA_PATH, 'rb') as f:
                self.product_metadata = pickle.load(f)
            print(f"Loaded embeddings for {len(self.product_metadata)} products")
            return
        
        print("Building new embeddings")
        self.load_embedding_model()
        
        # Create embeddings
        product_texts = [self.create_product_text(product) for product in self.products]
        embeddings = self.embedding_model.encode(product_texts)
        
        # Save embeddings and metadata
        os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(self.products, f)
        
        self.product_embeddings = embeddings
        self.product_metadata = self.products
        print(f"Built embeddings for {len(self.products)} products")
    
    def search_products(self, query: str, top_k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
        if self.product_embeddings is None or self.product_metadata is None:
            return []
        self.load_embedding_model()
        q_emb = self.embedding_model.encode([query])
        sims = cosine_similarity(q_emb, self.product_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx in top_idx:
            score = float(sims[idx]) if sims[idx] == sims[idx] else 0.0
            results.append((self.product_metadata[idx], score))
        return results
    
    def format_product_context(self, products: List[Tuple[Dict[str, Any], float]]) -> str:
        if not products:
            return "No products found in the database."
        
        formatted_products = []
        for product, score in products:
            formatted_products.append({
                "product_id": product.get("product_id", ""),
                "category": product.get("category", ""),
                "title": product.get("title", ""),
                "price": product.get("price", 0),
                "brand": product.get("brand", ""),
                "features": product.get("features", []),
                "avg_rating": product.get("avg_rating", 0)
            })
        
        return f"Available products: {json.dumps(formatted_products, indent=2)}"
    
    def format_messages_for_llama(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        
        formatted += "<|assistant|>\n"
        return formatted

    def _clean_model_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = (text or "").strip()
        
        STOP = {"|stop|", "| Stop |", "| Finish |", "STOP", "END", "Finish", "Stop"}
        if cleaned in STOP:
            return ""
        for m in STOP:
            if cleaned.endswith(m):
                cleaned = cleaned[:-len(m)].rstrip()
        
        cleaned = re.sub(r"[|]end[^|\n]*[|]", "", cleaned)
        cleaned = re.sub(r"<[|][^>]*[|]>", "", cleaned)
        cleaned = re.sub(r"\bSTOP\.?\b", "", cleaned, flags=re.IGNORECASE)
        
        cleaned = cleaned.replace('|', ' ')
        
        cleaned = re.sub(r"Available products:\s*\[.*?\]\s*", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"(Would you like|Do you need|Can I help|Is there anything|Anything else|Let me know|Please let me|Feel free to|If you have|If you need|If you want|If you're looking|If you're interested|Any questions)[^.?!]*[.?!]?", "", cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r"I see you provided[^.]*\.?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"I will provide[^.]*\.?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Please note that shipping times may vary[^.]*\.?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bfinished\.?\b", "", cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r"\b(What|How|Why|Where|When|Which)\s*$", "", cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def load_tool_data(self):
        self.shipping_data = {}
        self.return_policy_data = {}
        shipping_path = "data/shipping_speed_data.jsonl"
        if os.path.exists(shipping_path):
            with open(shipping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    cat = data.get('category')
                    if cat:
                        self.shipping_data[cat] = data
        policy_path = "data/return_policy_data.jsonl"
        if os.path.exists(policy_path):
            with open(policy_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    cat = data.get('category')
                    if cat:
                        self.return_policy_data[cat] = data
        print(f"Loaded shipping data for {len(self.shipping_data)} categories")
        print(f"Loaded return policy data for {len(self.return_policy_data)} categories")

    def extract_category_from_query(self, query: str) -> str:
        q = (query or '').lower()
        mapping = {
            'Health_and_Household': ['health', 'vitamin', 'skincare', 'moisturizer', 'cleanser', 'personal care'],
            'Home_and_Kitchen': ['kitchen', 'cookware', 'dinnerware', 'appliance', 'cooking', 'food'],
            'Tools_and_Home_Improvement': ['tool', 'drill', 'hardware', 'power tool', 'safety', 'measuring'],
            'Grocery_and_Gourmet_Food': ['food', 'grocery', 'coffee', 'snack', 'organic', 'beverage'],
            'Arts_Crafts_and_Sewing': ['art', 'craft', 'sewing', 'painting', 'paint', 'crayola', 'creative'],
        }
        for cat, kws in mapping.items():
            if any(k in q for k in kws):
                return cat
        return 'Home_and_Kitchen'

    def safe_call_tool(self, tool_name: str, category: str) -> str:
        try:
            if tool_name == 'get_shipping_speed':
                data = self.shipping_data.get(category)
                if not data:
                    return f"Shipping information not available for {category}."
                parts = [
                    'Shipping options:',
                    f"- Standard: {data.get('standard_days')} days (free)",
                    f"- Express: {data.get('express_days')} days (${data.get('express_cost')})",
                    f"- Overnight: {data.get('overnight_days')} days (${data.get('overnight_cost')})",
                ]
                return "\n".join(parts)
            if tool_name == 'get_return_policy':
                data = self.return_policy_data.get(category)
                if not data:
                    return f"Return policy not available for {category}."
                parts = [
                    'Return policy:',
                    f"- Return window: {data.get('return_window')}",
                    f"- Condition: {data.get('condition')}",
                    f"- Restocking fee: ${data.get('restocking_fee')}",
                    f"- Special notes: {data.get('special_notes')}",
                ]
                return "\n".join(parts)
            return 'Tool not found.'
        except Exception as e:
            return f"Error retrieving information: {e}"

    def generate_response(self, query: str) -> str:
        messages: List[Dict[str, str]] = []
        messages.append({'role': 'system', 'content': SYSTEM_PROMPT})
        
        # Search for relevant products
        relevant_products = self.search_products(query, TOP_K)
        product_context = self.format_product_context(relevant_products)
        messages.append({'role': 'user', 'content': f"{query}\n\nHere are some products that might help:\n{product_context}"})
        
        prompt = self.format_messages_for_llama(messages)
        
        resp = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.2,
            stop=['<|user|>', '<|system|>', '<|assistant|>', 'STOP', 'STOP.', '|stop|', '| Stop |', '| Finish |', '|endoftool|', '|endoftoolresult|',
                  'Available products:', 'Would you like', 'Do you need', 'Can I help', 'Is there anything', 'Anything else', 'Let me know',
                  'Please let me', 'Feel free to', 'If you have', 'If you need', 'If you want', "If you're looking", "If you're interested", 'Any questions',
                  'I see you provided', 'I will provide', 'Please note that shipping times may vary', 'finished.'],
            echo=False
        )
        
        raw_first = (resp.get('choices') or [{}])[0].get('text', '').strip()
        first = self._clean_model_text(raw_first)
        
        # Handle tool calls
        ql = (query or '').lower()
        shipping_keywords = [
            "ship", "shipping", "delivery", "deliver", "arrive", "arrival", "eta", "etas",
            "how long", "when will", "lead time", "transit time", "ship time", "delivery time",
            "express", "overnight", "standard shipping", "expedited", "prime shipping", "same day", "next day"
        ]
        return_keywords = [
            "return", "refund", "exchange", "policy", "returning", "returns", "refunds", "exchanges",
            "return window", "restocking", "restocking fee", "money back", "can I return", "how to return",
            "return period", "return policy", "warranty", "guarantee"
        ]
        asked_shipping = any(k in ql for k in shipping_keywords)
        asked_returns = any(k in ql for k in return_keywords)
        
        if (('[TOOL: get_shipping_speed]' in raw_first or '[TOOL:get_shipping_speed]' in raw_first) and asked_shipping):
            category = self.extract_category_from_query(query)
            tool_text = self.safe_call_tool('get_shipping_speed', category)
            follow = (f"{prompt}{raw_first}\n\n" + f"Tool result: {tool_text}\n\n" + "Please provide a concise, helpful answer based on the tool result.")
            final = self.llm(
                follow,
                max_tokens=256,
                temperature=0.2,
                stop=['<|user|>', '<|system|>', '<|assistant|>', 'STOP', 'STOP.', '|stop|', '| Stop |', '| Finish |', '|endoftool|', '|endoftoolresult|',
                      'Available products:', 'Would you like', 'Do you need', 'Can I help', 'Is there anything', 'Anything else', 'Let me know',
                      'Please let me', 'Feel free to', 'If you have', 'If you need', 'If you want', "If you're looking", "If you're interested", 'Any questions',
                      'I see you provided', 'I will provide', 'Please note that shipping times may vary', 'finished.'],
                echo=False
            )
            final_text = (final.get('choices') or [{}])[0].get('text', '').strip()
            final_text = self._clean_model_text(final_text)
            final_text = re.sub(r"\[TOOL:[^\]]*\]", "", final_text).strip()
            return final_text or tool_text
        
        if (('[TOOL: get_return_policy]' in raw_first or '[TOOL:get_return_policy]' in raw_first) and asked_returns):
            category = self.extract_category_from_query(query)
            tool_text = self.safe_call_tool('get_return_policy', category)
            follow = (f"{prompt}{raw_first}\n\n" + f"Tool result: {tool_text}\n\n" + "Please provide a concise, helpful answer based on the tool result.")
            final = self.llm(
                follow,
                max_tokens=256,
                temperature=0.2,
                stop=['<|user|>', '<|system|>', '<|assistant|>', 'STOP', 'STOP.', '|stop|', '| Stop |', '| Finish |', '|endoftool|', '|endoftoolresult|',
                      'Available products:', 'Would you like', 'Do you need', 'Can I help', 'Is there anything', 'Anything else', 'Let me know',
                      'Please let me', 'Feel free to', 'If you have', 'If you need', 'If you want', "If you're looking", "If you're interested", 'Any questions',
                      'I see you provided', 'I will provide', 'Please note that shipping times may vary', 'finished.'],
                echo=False
            )
            final_text = (final.get('choices') or [{}])[0].get('text', '').strip()
            final_text = self._clean_model_text(final_text)
            final_text = re.sub(r"\[TOOL:[^\]]*\]", "", final_text).strip()
            return final_text or tool_text
        
        return first

    def evaluate_response(self, query: str, response: str, expected_product: Dict[str, Any]) -> Dict[str, Any]:
        result = self.evaluator.evaluate_response(query, response, expected_product)
        
        evaluation = {
            'query_type': self.classify_query(query),
            'is_helpful': result.is_helpful,
            'is_accurate': result.is_accurate,
            'is_complete': result.is_complete,
            'is_relevant': result.is_relevant,
            'confidence_score': result.confidence_score,
            'issues': result.issues,
            'strengths': result.strengths,
            'response_length': len(response),
            'is_empty': len(response.strip()) == 0,
            'contains_error': 'error' in response.lower() or 'not available' in response.lower()
        }
        
        return evaluation

    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['price', 'cost', 'much']):
            return 'price'
        elif any(word in query_lower for word in ['rating', 'rated']):
            return 'rating'
        elif any(word in query_lower for word in ['brand', 'makes', 'company', 'produces']):
            return 'brand'
        elif 'features' in query_lower:
            return 'features'
        elif any(word in query_lower for word in ['ship', 'shipping', 'delivery', 'deliver']):
            return 'shipping'
        elif any(word in query_lower for word in ['return', 'refund', 'policy']):
            return 'return_policy'
        elif any(word in query_lower for word in ['is', 'does', 'has']):
            return 'feature_check'
        else:
            return 'general'

    def run_tests(self, max_tests: int = None, seed: int = None) -> List[Dict[str, Any]]:
        print("Loading test data...")
        test_data = []
        with open(TEST_DATA_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        
        if max_tests:
            test_data = test_data[:max_tests]
        
        print(f"Running {len(test_data)} tests...")
        
        results = []
        for i, test_case in enumerate(test_data):
            print(f"Test {i+1}/{len(test_data)}: {test_case['user_query'][:50]}...")
            
            start_time = time.time()
            try:
                response = self.generate_response(test_case['user_query'])
                response_time = time.time() - start_time
                
                evaluation = self.evaluate_response(
                    test_case['user_query'], 
                    response, 
                    test_case
                )
                
                result = {
                    'test_id': i + 1,
                    'product_id': test_case['product_id'],
                    'title': test_case['title'],
                    'user_query': test_case['user_query'],
                    'agent_reply': response,
                    'response_time': response_time,
                    'evaluation': evaluation
                }
                
                results.append(result)
                
                test_case['agent_reply'] = response
                
            except Exception as e:
                print(f"Error in test {i+1}: {e}")
                result = {
                    'test_id': i + 1,
                    'product_id': test_case['product_id'],
                    'title': test_case['title'],
                    'user_query': test_case['user_query'],
                    'agent_reply': f"ERROR: {e}",
                    'response_time': 0,
                    'evaluation': {'error': True}
                }
                results.append(result)
        
        return results

    def calculate_accuracy_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_tests = len(results)
        successful_tests = len([r for r in results if not r['evaluation'].get('error', False)])
        
        # Response quality metrics
        empty_responses = len([r for r in results if r['evaluation'].get('is_empty', False)])
        error_responses = len([r for r in results if r['evaluation'].get('contains_error', False)])
        
        # Query type accuracy
        query_types = {}
        for result in results:
            query_type = result['evaluation'].get('query_type', 'unknown')
            if query_type not in query_types:
                query_types[query_type] = {'total': 0, 'accurate': 0}
            query_types[query_type]['total'] += 1
            
            eval_data = result['evaluation']
            if eval_data.get('is_accurate', False):
                query_types[query_type]['accurate'] += 1
        
        # Calculate accuracy rates
        accuracy_rates = {}
        for query_type, data in query_types.items():
            if data['total'] > 0:
                accuracy_rates[query_type] = data['accurate'] / data['total']
        
        response_times = [r['response_time'] for r in results if r['response_time'] > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        metrics = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'empty_response_rate': empty_responses / total_tests if total_tests > 0 else 0,
            'error_response_rate': error_responses / total_tests if total_tests > 0 else 0,
            'query_type_accuracy': accuracy_rates,
            'avg_response_time': avg_response_time,
            'response_time_compliance': len([t for t in response_times if t <= 2.0]) / len(response_times) if response_times else 0
        }
        
        return metrics

    def save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]):
        with open(OUTPUT_PATH, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        metrics_path = OUTPUT_PATH.replace('.jsonl', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to {OUTPUT_PATH}")
        print(f"Metrics saved to {metrics_path}")

    def print_summary(self, metrics: Dict[str, Any]):
        print(f"Total Tests: {metrics['total_tests']}")
        print(f"Successful Tests: {metrics['successful_tests']}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Empty Response Rate: {metrics['empty_response_rate']:.2%}")
        print(f"Error Response Rate: {metrics['error_response_rate']:.2%}")
        print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"Response Time Compliance (≤2s): {metrics['response_time_compliance']:.2%}")
        
        print("\nQuery Type Accuracy:")
        for query_type, accuracy in metrics['query_type_accuracy'].items():
            print(f"  {query_type}: {accuracy:.2%}")
        

def main():
    parser = argparse.ArgumentParser(description='Test agent accuracy')
    parser.add_argument('--max-tests', type=int, help='Maximum number of tests to run')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    tester = AgentTester()
    
    tester.load_model()
    tester.products = tester.load_products()
    tester.build_embeddings()
    tester.load_tool_data()
    
    results = tester.run_tests(max_tests=args.max_tests, seed=args.seed)
    
    metrics = tester.calculate_accuracy_metrics(results)
    
    tester.save_results(results, metrics)
    tester.print_summary(metrics)

if __name__ == "__main__":
    main()
