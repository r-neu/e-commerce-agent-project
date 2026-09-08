# Shopping Assistant

Shopping Assistant helps online shoppers search a product catalog in everyday language. They can ask for recommendations, compare product details, and check shipping or return policies in the same conversation.

## Why this problem

Product search works well when shoppers know the product name or category. It is less useful when a request combines a budget, specific features, and a minimum rating. A query such as “show me a well-rated cookware set under $300” may take several searches.

The shopper describes what they need, and the assistant finds products that match the request. Shipping and return information is included in the same flow because it can affect the final choice.

## User flow

1. A shopper asks a product, shipping, or return question.
2. The system searches the catalog or retrieves the relevant store policy.
3. The shopper receives an answer and can refine the request in the same chat.

## Technical workflow

Product requests are embedded with BGE-M3 and compared with the 100-product demo catalog. The eight closest product records are added to the prompt with their title, brand, price, features, and rating. The fine-tuned Llama model writes the answer from those records.

Shipping and return questions use category-based lookups from structured policy files. The Gradio interface streams each response, and `llama.cpp` runs the quantized model locally.

## Product decisions

I focused the MVP on product discovery and common policy questions before purchase. Checkout, payment, order tracking, and account support need separate commerce systems, so I left them out.

I used fine-tuning to improve how the model handles shopping questions. The live assistant still retrieves product facts from the catalog for every request. Shipping and return answers come from policy files. Changing a price or policy does not require another fine-tuning run.

I chose a quantized local model so the demo does not depend on a paid inference API.

## Training

I fine-tuned `Meta-Llama-3.1-8B-Instruct` with QLoRA. The 120,000-example training set contains 48,000 product Q&A examples and 72,000 review-based tasks. Those tasks cover choosing one product, building a shortlist, rejecting mismatched options, and asking for clarification.

I then merged the adapter with the base model and converted it to a Q4_K_M GGUF file for local inference.

## Testing

The automated test set contains 100 questions covering price, brand, rating, features, shipping, returns, and general product queries. The saved run returned a response for every question. Shipping scored highest. Brand and rating questions were the weakest areas.

A separate set of 20 edge cases covers greetings, ambiguous wording, and unsupported requests.

Run a small test sample after launching the app once:

```bash
python scripts/test_agent.py --max-tests 10
```

## Tech stack

Python, Llama 3.1, QLoRA, `llama.cpp`, BGE-M3, Sentence Transformers, scikit-learn, and Gradio.

## Run locally

The current `MODEL_CONFIG` is tuned for Apple Silicon. Other hardware may need different GPU and thread settings.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_local_demo_ecommerce_agent.py
```

The first launch downloads the fine-tuned model and BGE-M3, then builds the catalog embeddings. Open `http://127.0.0.1:7860` after the server starts.
