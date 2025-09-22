This guide will help you set up and run the E-commerce Agent system on your local machine.

## System Requirements

- **Operating System**: macOS (tested on macOS with M1 chip)
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB recommended)
- **Storage**: At least 5GB free space for models and data
- **GPU**: Apple Silicon

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd E-commerce-Agent
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install torch>=2.0.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0

# Install ML libraries
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install peft>=0.6.0
pip install accelerate>=0.24.0
pip install trl>=0.7.0
pip install bitsandbytes>=0.41.0

# Install embedding and search libraries
pip install faiss-cpu>=1.7.4
pip install FlagEmbedding>=1.2.0
pip install sentence-transformers
pip install scikit-learn

# Install model inference library
pip install llama-cpp-python

# Install UI library
pip install gradio

# Install utilities
pip install ujson>=5.8.0
pip install python-dotenv>=1.0.0
pip install requests
pip install tqdm
```

### 4. Download Required Models

The system will automatically download the following models when you first run it:

- **E-commerce Model**: `llama-8b-ecommerce-Q4_K_M.gguf` 
- **Embedding Model**: `BAAI/bge-m3` 

Models will be stored in:
- `models/` directory for the main model
- `hf_cache/` directory for Hugging Face models

## Running the Agent

### 1. Start the Application
```bash
# Make sure you're in the project directory and virtual environment is activated
python scripts/run_local_demo_ecommerce_agent.py
```

### 2. Access the Web Interface
- The script will start a Gradio web server
- Open your browser and go to: `http://localhost:7860`
- You should see the E-commerce Agent interface

### 3. Using the Agent

#### Basic Usage
- Type your questions in the chat input box
- The agent can help with:
  - Product searches and recommendations
  - Price inquiries
  - Feature questions
  - Shipping information
  - Return policies

#### Example Queries
```
"Show me kitchen appliances under $200"
"What's the price of the Cuisinart cookware set?"
"Find organic products"
"How long does shipping take for health products?"
"What's the return policy for tools?"
```

#### Preset Buttons
- Use the preset buttons for common queries:
  - "Show me kitchen appliances under $200"
  - "Find organic products under $100"

### Performance Optimization

#### For Apple Silicon (M1/M2)
- The script is optimized for Apple Silicon
- Uses Metal Performance Shaders (MPS) when available
- GPU layers are automatically configured

#### For NVIDIA GPUs
- Install CUDA-compatible PyTorch
- Modify `MODEL_CONFIG` to use CUDA:
```python
MODEL_CONFIG = {
    "n_ctx": 4096,
    "n_gpu_layers": -1,  # Use all GPU layers
    "n_threads": 8,
    "verbose": False,
    "use_mmap": True,
    "use_mlock": True,
}
```

## Testing the Agent

### Run Automated Tests
```bash
# Test with limited samples
python scripts/test_agent.py --max-tests 10

# Test all samples
python scripts/test_agent.py --max-tests 100
```

### Generate Edge Case Responses
```bash
python scripts/generate_edge_case_responses.py
```

## Advanced Configuration

### Model Configuration
Edit `MODEL_CONFIG` in the script to adjust:
- Context window size (`n_ctx`)
- GPU layers (`n_gpu_layers`)
- CPU threads (`n_threads`)

### RAG Configuration
Modify these parameters:
- `TOP_K`: Number of products to retrieve
- `SIMILARITY_THRESHOLD`: Minimum similarity score
- `EMBEDDING_MODEL`: Embedding model to use
