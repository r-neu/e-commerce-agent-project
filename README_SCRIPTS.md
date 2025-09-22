# Scripts Documentation

Below is a comprehensive explanation of each script:

## Main Application Scripts

### `run_local_demo_ecommerce_agent.py`
- **Purpose**: Main application script for running the e-commerce agent with Gradio UI
- **Features**: 
  - RAG (Retrieval Augmented Generation) for product search
  - Tool integration (shipping speed, return policy)
  - Streaming responses
  - Interactive web interface
- **Usage**: Primary script for user interaction with the agent

## Data Processing Scripts

### `build_demo_store_products_100.py`
- **Purpose**: Creates the demo product catalog (`demo_store_products_100.jsonl`)
- **Features**: Random sampling from product library with seed control
- **Usage**: Generate consistent demo datasets for testing

### `build_product_context.py`
- **Purpose**: Builds product context and embeddings for RAG
- **Features**: Creates embeddings index and metadata files
- **Usage**: Preprocessing for product search functionality

## Training Data Generation Scripts

### `make_sft_amazon_qa.py`
- **Purpose**: Creates Supervised Fine-Tuning dataset from Amazon Q&A data
- **Features**: Downloads and processes Amazon Q&A dataset
- **Usage**: Generate training data for question-answering capabilities

### `make_amazon_review_*.py` (Multiple files)
- **Purpose**: Generate training data for different Amazon review tasks
- **Files**:
  - `make_amazon_review_clarify_combined.py`: Clarification task data
  - `make_amazon_review_pick_list_query.py`: List selection queries
  - `make_amazon_review_pick_one_query.py`: Single product selection queries
  - `make_amazon_review_reject_query.py`: Query rejection data
  - `make_amazon_review_filter.py`: Data filtering utilities
- **Usage**: Create specialized training datasets for different agent behaviors

### `merge_final_sft.py`
- **Purpose**: Merges multiple training datasets into final SFT dataset
- **Usage**: Combine all training data sources

## Evaluation and Testing Scripts

### `test_agent.py`
- **Purpose**: Comprehensive agent testing and evaluation
- **Features**:
  - Automated testing with predefined queries
  - Performance metrics calculation
  - Response quality evaluation
- **Usage**: Evaluate agent performance on test dataset

### `evaluation.py`
- **Purpose**: Advanced evaluation logic and metrics
- **Features**: Intelligent response analysis, confidence scoring
- **Usage**: Detailed assessment of agent responses

### `generate_edge_case_responses.py`
- **Purpose**: Generates agent responses for edge case queries
- **Features**: Tests agent on complex/boundary scenarios
- **Usage**: Evaluate agent behavior on challenging queries

## Utility Scripts

### `sample_qa_dataset.py`
- **Purpose**: Samples and processes Q&A datasets
- **Usage**: Data preprocessing and sampling utilities

### `validate_chat_jsonl.py`
- **Purpose**: Validates JSONL chat data format
- **Usage**: Data quality assurance

## Google Colab

### `ecommerce_agent_train.ipynb`
- **Purpose**: Model training notebook
- **Features**: Complete training pipeline for fine-tuning
- **Usage**: Train the e-commerce agent model

### `compress_finetuned_model.ipynb`
- **Purpose**: Model compression and quantization
- **Features**: Converts trained model to GGUF format with Q4_K_M quantization
- **Usage**: Optimize model for deployment

## Script Categories

### Data Generation
- `build_*`: Create datasets and data structures
- `make_*`: Generate training data from sources
- `merge_*`: Combine datasets

### Testing & Evaluation
- `test_*`: Testing and evaluation scripts
- `generate_*`: Generate test responses
- `evaluation.py`: Evaluation logic

### Training
- `*_train.ipynb`: Training notebooks
- `compress_*`: Model optimization

### Utilities
- `sample_*`: Data sampling utilities
- `validate_*`: Data validation

## Command Line Usage

Most scripts support command-line arguments:

```bash
# Test agent with limited samples
python scripts/test_agent.py --max-tests 10 --seed 42

# Build demo products
python scripts/build_demo_store_products_100.py --max-rows 100 --seed 42

# Generate edge case responses
python scripts/generate_edge_case_responses.py
```

## Dependencies

Scripts require various libraries:
- **Core**: `llama-cpp-python`, `sentence-transformers`, `scikit-learn`
- **UI**: `gradio`
- **Data**: `numpy`, `pandas`, `json`
- **ML**: `transformers`, `datasets`, `peft`
