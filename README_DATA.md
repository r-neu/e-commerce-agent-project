# Data Files Documentation

Below is a comprehensive explanation of each file:

## Core Product Data

### `demo_store_products_100.jsonl`
- **Purpose**: Main product catalog with 100 sample products
- **Format**: JSONL (one JSON object per line)
- **Fields**: `product_id`, `category`, `sub_category`, `title`, `price`, `brand`, `features`, `avg_rating`
- **Usage**: Primary product database for RAG (Retrieval Augmented Generation)

### `product_library.jsonl`
- **Purpose**: Extended product library (larger dataset)
- **Format**: JSONL
- **Usage**: Source for generating demo products and training data

## Tool Data

### `shipping_speed_data.jsonl`
- **Purpose**: Shipping options and delivery times by product category
- **Format**: JSONL
- **Fields**: `category`, `standard_days`, `express_days`, `overnight_days`, `standard_cost`, `express_cost`, `overnight_cost`
- **Usage**: Powers the `get_shipping_speed` tool function

### `return_policy_data.jsonl`
- **Purpose**: Return policies and conditions by product category
- **Format**: JSONL
- **Fields**: `category`, `return_window`, `condition`, `restocking_fee`, `special_notes`
- **Usage**: Powers the `get_return_policy` tool function

## Training Data

### `sft_full_merged_final.jsonl`
- **Purpose**: Complete Supervised Fine-Tuning dataset (merged)
- **Size**: ~228MB
- **Usage**: Final training dataset for the e-commerce model

### `sft_amazon_qa_organized.jsonl` & `sft_amazon_qa_organized_filtered.jsonl`
- **Purpose**: Amazon Q&A dataset for training
- **Size**: 68MB (organized), 34MB (filtered)
- **Usage**: Source training data for question-answering capabilities

### `sft_amazon_review.jsonl`
- **Purpose**: Amazon product reviews dataset
- **Size**: ~195MB
- **Usage**: Training data for review analysis and product understanding

## Amazon Review Task Data

### `amazon_review_clarify_organized.jsonl`
- **Purpose**: Clarification task data
- **Usage**: Training data for handling ambiguous queries

### `amazon_review_pick_list_organized.jsonl`
- **Purpose**: List selection task data
- **Usage**: Training data for multi-product recommendations

### `amazon_review_pick_one_organized.jsonl`
- **Purpose**: Single product selection task data 
- **Usage**: Training data for single product recommendations

### `amazon_review_reject_organized.jsonl`
- **Purpose**: Query rejection task data 
- **Usage**: Training data for handling inappropriate or unclear queries

## Testing Data

### `product_test_queries.jsonl`
- **Purpose**: Test queries for agent evaluation
- **Format**: JSONL with `product_id`, `title`, `price`, `features`, `user_query`, `agent_reply`
- **Usage**: Automated testing of agent responses

### `edge_case_queries_results.jsonl`
- **Purpose**: Edge case queries and agent responses
- **Format**: JSONL with `query_id`, `user_query`, `agent_reply`
- **Usage**: Testing agent behavior on complex/boundary cases

## Evaluation Results

### `agent_test_results.jsonl`
- **Purpose**: Detailed test results from agent evaluation
- **Format**: JSONL with test results, evaluations, and metrics
- **Usage**: Analysis of agent performance

### `agent_test_results_metrics.json`
- **Purpose**: Aggregated performance metrics
- **Format**: JSON with summary statistics
- **Usage**: Quick overview of agent performance metrics

## Data Flow

1. **Raw Data**: Amazon datasets and product libraries
2. **Processing**: Scripts clean and organize data
3. **Training**: Processed data used for model fine-tuning
4. **Testing**: Test queries evaluate agent performance
5. **Results**: Evaluation outputs stored for analysis
