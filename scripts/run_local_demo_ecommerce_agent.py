#!/usr/bin/env python3
import os
import json
import pickle
import numpy as np
import gradio as gr
from typing import List, Dict, Any, Tuple, Generator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import re

MODEL_PATH = "models/llama-8b-ecommerce-Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/rri02/llama-8b-ecommerce-gguf/resolve/main/llama-8b-ecommerce-Q4_K_M.gguf"
# Data paths
PRODUCTS_PATH = "data/demo_store_products_100.jsonl"
EMBEDDINGS_PATH = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "product_embeddings.pkl")
METADATA_PATH = os.path.join(EMBEDDINGS_PATH, "product_metadata.pkl")
# RAG Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 8
SIMILARITY_THRESHOLD = 0.1  
# Configuration for my M1 Max
MODEL_CONFIG = {
    "n_ctx": 4096,  # Context window
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

class LocalEcommerceAgent:
    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.product_embeddings = None
        self.product_metadata = None
        self.products = []
        self.shipping_data = {}
        self.return_policy_data = {}
        self.return_policy_data = {}
        
    def download_model(self):
        """Download the finetuned model if not present"""
        if os.path.exists(MODEL_PATH):
            print(f"Model already exists at {MODEL_PATH}")
            return
        
        print(f"Downloading model from {MODEL_URL}")
        
        import requests
        from tqdm import tqdm
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(MODEL_PATH, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Model downloaded to {MODEL_PATH}")
    
    def load_model(self):
        if self.llm is not None:
            return
        
        print("Loading finetuned Llama model")
        
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
            print("Loading BGE-M3 embedding model...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded")
    
    def build_embeddings(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            print("Loading existing embeddings...")
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
        
        import re
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
    
    def generate_response_with_tools(self, query: str, history: List[Dict[str, str]]) -> str:
        messages: List[Dict[str, str]] = []
        messages.append({'role': 'system', 'content': SYSTEM_PROMPT})
        for msg in (history or []):
            role = msg.get('role'); content = msg.get('content')
            if content and role in ('user', 'assistant'):
                messages.append({'role': role, 'content': content})
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
        # Only allow tool execution if the user asked about that topic
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
        if not first:
            ql = (query or "").lower()
            if any(k in ql for k in ["ship","shipping","delivery","deliver"]):
                category = self.extract_category_from_query(query)
                return self.safe_call_tool("get_shipping_speed", category)
            if any(k in ql for k in ["return","refund","exchange","policy"]):
                category = self.extract_category_from_query(query)
                return self.safe_call_tool("get_return_policy", category)
        
        first = re.sub(r"\[TOOL:[^\]]*\]", "", first).strip()
        return first or "Sorry, I couldn't find the information."
    
    def generate_response_stream(self, query: str, history: List[Tuple[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> Generator[str, None, None]:
        # Build conversation history
        messages = []
        
        # Use the system prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        # Search for relevant products for every query
        relevant_products = self.search_products(query)
        
        # Format query with product context
        if relevant_products:
            product_context = self.format_product_context(relevant_products)
            formatted_query = f"{query}\n\nHere are some products that might help:\n{product_context}"
        else:
            formatted_query = f"{query}\n\nI couldn't find any products matching your request in our store."
        
        # Add conversation history
        for msg in history:
            if msg.get("role") == "user" and msg.get("content"):
                messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") == "assistant" and msg.get("content"):
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add current query
        messages.append({"role": "user", "content": formatted_query})
        
        try:
            
            prompt = self.format_messages_for_llama(messages)
            
            # Check for tool usage intent first
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
            
            if asked_shipping:
                category = self.extract_category_from_query(query)
                tool_text = self.safe_call_tool('get_shipping_speed', category)
                yield tool_text
                return
            elif asked_returns:
                category = self.extract_category_from_query(query)
                tool_text = self.safe_call_tool('get_return_policy', category)
                yield tool_text
                return   
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|user|>", "<|system|>", "<|assistant|>", "|", "| Finish |", "<|end|>", "END", "STOP", "Based on the provided product context", 
         "What", "How", "Would you", "would you", "Do you", "do you", "Can you", "can you", "Could you", "could you", "Are you", "are you", "If you", "if you", "Is there", "is there", "Which", "Where", "When", "Why",
         "<|", "|>", "<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|start|>", "<|stop|>",
         "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|sep|>", "<|cls|>", "<|mask|>",
         "<|special|>", "<|token|>", "<|prompt|>", "<|response|>", "<|completion|>", "<|generation|>",
         "<|input|>", "<|output|>", "<|context|>", "<|instruction|>", "<|task|>", "<|query|>",
         "<|answer|>", "<|result|>", "<|summary|>", "<|conclusion|>", "<|end|>", "<|finish|>",
         "<|done|>", "<|complete|>", "<|stop|>", "<|halt|>", "<|terminate|>", "<|abort|>",
         "In the context", "From the context", 
         "Would you like", "Do you need", "Can I help", "Is there anything", "Anything else",
         "Let me know", "Please let me", "Feel free to", "If you have",
         "If you need", "If you want", "If you're looking", "If you're interested",
         "Anything specific", "Any preferences", "Any questions"],
                echo=False,
                stream=True  
            )
                      
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("text", "")
                    if delta:
                        yield delta
            
        except Exception as e:
            yield f"Error generating response: {e}\n\nPlease try again."

def create_ui():
    agent = LocalEcommerceAgent()  
    agent.download_model()
    agent.load_model() 
    agent.products = agent.load_products()
    agent.load_tool_data()
    agent.build_embeddings()
    
    css = """
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');
        
        .gradio-container {
            max-width: 400px !important;
            margin: 0 auto !important;
            padding: 10px !important;
            background: linear-gradient(135deg, #fefefe 0%, #f9fafb 100%) !important;
            min-height: 100vh !important;
            font-family: 'Crimson Text', serif !important;
        }
        
        .header-section {
            text-align: center !important;
            margin-bottom: 15px !important;
            padding: 15px 10px !important;
            background: linear-gradient(135deg, #e8f4fd 0%, #d1e7dd 100%) !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 15px rgba(232, 244, 253, 0.3) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .header-section::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            background: linear-gradient(45deg, rgba(255,255,255,0.2) 0%, transparent 50%, rgba(255,255,255,0.2) 100%) !important;
            pointer-events: none !important;
        }
        
        .welcome-title {
            font-size: 2.0rem !important;
            font-weight: 600 !important;
            color: #4a5568 !important;
            margin: 0 0 8px 0 !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            letter-spacing: -0.01em !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        .welcome-subtitle {
            font-size: 1.0rem !important;
            color: #718096 !important;
            margin: 0 !important;
            font-weight: 400 !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        .preset-section {
            display: flex !important;
            gap: 8px !important;
            justify-content: center !important;
            margin-bottom: 15px !important;
            flex-wrap: wrap !important;
        }
        
        .preset-btn {
            padding: 8px 12px !important;
            border: none !important;
            border-radius: 10px !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            min-width: 120px !important;
            text-align: center !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
            position: relative !important;
            overflow: hidden !important;
            font-family: 'Crimson Text', serif !important;
        }
        
        .preset-btn::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
            transition: left 0.5s ease !important;
        }
        
        .preset-btn:hover::before {
            left: 100% !important;
        }
        
        .preset-btn:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2) !important;
        }
        
        .preset-btn:active {
            transform: translateY(-1px) !important;
        }
        
        .preset-btn-1 {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #ff9a9e 100%) !important;
            color: #2d3748 !important;
        }
        
        .preset-btn-2 {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 50%, #a8edea 100%) !important;
            color: #2d3748 !important;
        }

        .chat-container {
            background: white !important;
            border-radius: 12px !important;
            padding: 15px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08) !important;
            min-height: 350px !important;
            border: 1px solid rgba(226, 232, 240, 0.8) !important;
        }
        
        .input-section {
            display: flex !important;
            gap: 12px !important;
            align-items: flex-end !important;
        }
        
        .input-textbox {
            flex: 1 !important;
            border-radius: 14px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 12px 16px !important;
            font-size: 1.05rem !important;
            font-family: 'Crimson Text', serif !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            background: white !important;
            resize: none !important;
        }
        
        .input-textbox:focus {
            border-color: #a8c8ec !important;
            box-shadow: 0 6px 20px rgba(168, 200, 236, 0.15) !important;
            outline: none !important;
        }
        
        .input-textbox::placeholder {
            color: #a0aec0 !important;
            font-weight: 400 !important;
        }
        
        .submit-btn {
            padding: 12px 24px !important;
            border: none !important;
            border-radius: 14px !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            background: linear-gradient(135deg, #e8f4fd 0%, #d1e7dd 100%) !important;
            color: #4a5568 !important;
            box-shadow: 0 6px 18px rgba(232, 244, 253, 0.3) !important;
            font-family: 'Crimson Text', serif !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .submit-btn::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
            transition: left 0.5s ease !important;
        }
        
        .submit-btn:hover::before {
            left: 100% !important;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 25px rgba(232, 244, 253, 0.4) !important;
        }
        
        .submit-btn:active {
            transform: translateY(-1px) !important;
        }
        
        /* Chat message styling - much smaller blocks */
        .message {
            margin-bottom: 4px !important;
            padding: 4px 8px !important;
            border-radius: 4px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        }
        
        .message.user {
            background: linear-gradient(135deg, #e8f4fd 0%, #d1e7dd 100%) !important;
            color: #4a5568 !important;
            margin-left: 5% !important;
        }
        
        .message.assistant {
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%) !important;
            color: #2d3748 !important;
            margin-right: 5% !important;
            border: 1px solid #e2e8f0 !important;
        }
    """
    
    with gr.Blocks(css=css, title="Shopping Assistant") as demo:
        # Preset buttons
        with gr.Row():
            preset1 = gr.Button("Show me paint set products under $300", elem_classes=["preset-btn", "preset-btn-1"])
            preset2 = gr.Button("Cookware sets with good ratings", elem_classes=["preset-btn", "preset-btn-2"])

        # Chat interface
        chatbot = gr.Chatbot(
            value=[],
            elem_id="chatbot",
            elem_classes=["chat-container"],
            height=400,
            type="messages",
            show_label=False
        )
        
        # Input section
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me about products, find recommendations ~",
                elem_classes=["input-textbox"],
                lines=2,
                show_label=False,
                scale=4
            )
            submit_btn = gr.Button("Submit", elem_classes=["submit-btn"], scale=1)
        
        # Event handlers
        def respond(message, history):
            if not message.strip():
                return history, ""
            
            # Add user message to history
            history.append({"role": "user", "content": message})
            
            # Generate response with streaming
            response_text = ""
            for chunk in agent.generate_response_stream(message, history[:-1]):
                response_text += chunk
                # Update the assistant's response in history
                if len(history) > 0 and history[-1]["role"] == "user":
                    history.append({"role": "assistant", "content": response_text})
                else:
                    history[-1] = {"role": "assistant", "content": response_text}
                yield history, ""
            
            return history, ""
        
        def preset_click(preset_text, history):
            if not preset_text.strip():
                return history, ""
            
            # Add user message to history
            history.append({"role": "user", "content": preset_text})
            
            # Generate response with streaming
            response_text = ""
            for chunk in agent.generate_response_stream(preset_text, history[:-1]):
                response_text += chunk
                # Update the assistant's response in history
                if len(history) > 0 and history[-1]["role"] == "user":
                    history.append({"role": "assistant", "content": response_text})
                else:
                    history[-1] = {"role": "assistant", "content": response_text}
                yield history, ""
            
            return history, ""
        
        # Connect events
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])

        def preset1_click(history):
            if history is None:
                history = []
            history.append({"role": "user", "content": "Show me paint set products under $300"})
            
            # Generate response with streaming
            response_text = ""
            for chunk in agent.generate_response_stream("Show me paint set products under $300", history[:-1]):
                response_text += chunk
                # Update the assistant's response in history
                if len(history) > 0 and history[-1]["role"] == "user":
                    history.append({"role": "assistant", "content": response_text})
                else:
                    history[-1] = {"role": "assistant", "content": response_text}
                yield history, ""
            
            return history, ""
        
        def preset2_click(history):
            if history is None:
                history = []
            history.append({"role": "user", "content": "Find me premium cookware and kitchen sets with good ratings"})
            
            # Generate response with streaming
            response_text = ""
            for chunk in agent.generate_response_stream("Find me premium cookware and kitchen sets with good ratings", history[:-1]):
                response_text += chunk
                # Update the assistant's response in history
                if len(history) > 0 and history[-1]["role"] == "user":
                    history.append({"role": "assistant", "content": response_text})
                else:
                    history[-1] = {"role": "assistant", "content": response_text}
                yield history, ""
            
            return history, ""

        preset1.click(preset1_click, inputs=[chatbot], outputs=[chatbot, msg])
        preset2.click(preset2_click, inputs=[chatbot], outputs=[chatbot, msg])

    return demo

if __name__ == "__main__":  
    demo = create_ui()
    print("Open browser to: http://127.0.0.1:7860")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )