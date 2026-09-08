# Shopping Assistant

Shopping Assistant is an e-commerce agent for an online product catalog. A shopper describes what they need in plain language, and the agent looks for matching products. It also answers questions about shipping and returns.

## Why this problem

Shoppers do not always begin with a product name or a clear set of filters. Sometimes they only have a rough idea and work out the details as they browse. At that stage, it can be difficult to know what to search for or which filters to use.

The Shopping Assistant agent starts with that rough description. The shopper can add more detail after seeing the first set of results.

## User flow

1. A shopper describes what they need.
2. The agent searches the catalog and returns a shortlist.
3. The shopper adds more detail to narrow the results or asks about shipping and returns.

## Technical workflow

BGE-M3 creates an embedding for each product in the 100-product demo catalog. For a product question, the agent retrieves the eight closest matches. Their title, brand, price, features, and rating are passed to the fine-tuned Llama model, which writes the answer.

For shipping and returns, the agent reads the policy for the relevant product category from JSON files. Gradio provides the chat interface, and `llama.cpp` runs the quantized model.

## Product decisions

I limited the first version of the agent to questions that come before a purchase. I left out checkout, payment, order tracking, and account support because they require customer, payment, or order data that is not part of this prototype.

I kept the live catalog data outside the model. The agent retrieves product details when a question arrives, so changing a price does not require another fine-tuning run.

Shipping and return answers come directly from policy files.

## Training

I fine-tuned `Meta-Llama-3.1-8B-Instruct` with QLoRA. The training file contains 120,000 examples. It combines 48,000 product Q&A conversations with 72,000 review-based tasks for product selection and clarification.

I then merged the adapter with the base model and converted it to a Q4_K_M GGUF file for local inference.

## Testing

The test set contains 100 questions about price, brand, rating, features, shipping, returns, and general product information. The saved run produced a response for every question. Accuracy was highest for shipping questions and lowest for brand and rating questions.

Another 20 cases cover greetings, ambiguous wording, and unsupported requests.

Run a small test sample after launching the app once:

```bash
python scripts/test_agent.py --max-tests 10
```

## Tech stack

Python, Llama 3.1, QLoRA, `llama.cpp`, BGE-M3, Sentence Transformers, scikit-learn, and Gradio.

## Run locally

The current `MODEL_CONFIG` is set up for Apple Silicon. Other hardware may need different GPU and thread settings.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_local_demo_ecommerce_agent.py
```

The first launch downloads the fine-tuned model and BGE-M3, then builds the catalog embeddings. Open `http://127.0.0.1:7860` after the server starts.
