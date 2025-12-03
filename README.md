<p align="center">
  <img src="/assets/light2.svg" alt="RAGOpt Logo" width="1000"/>
</p>

<p align="center">
  RAGOpt eliminates manual hyperparameter tuning in your RAG pipelines with Bayesian optimization 🚀
</p>

<p align="center">
  <a href="https://github.com/your-repo/publish"><img src="https://img.shields.io/badge/Publish-passing-brightgreen" alt="Publish passing"></a>
  <a href="https://pypi.org/project/your-package/"><img src="https://img.shields.io/badge/release-v0.1.5-orange" alt="PyPI version"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python 3.11"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

---

- **Documentation**: https://ragopt.aboneda.com
- **Colab Notebook**: https://colab.research.google.com/drive/1hrfAHCfm3x0Ov-amCEHpptMiyoqC-McE

---

**RAGOpt** is a Python framework to optimize Retrieval-Augmented Generation (RAG) pipelines. It eliminates manual hyperparameter tuning using Bayesian optimization, automatically finding the best configuration for your dataset and use case.

## Key Features

- Optimizes 20+ RAG hyperparameters including chunk size, overlap, embedding strategies, and LLM selection.
- Flexible with any LangChain-compatible model or provider.
- Partially Opinionated - Smart defaults with full flexibility for customization
- Generates Pareto-optimal configurations for your specific data.
- Comprehensive Metrics - Quality (precision, recall, faithfulness), performance (latency, cost), and safety (toxicity, bias)

---

## Installation

```bash
pip install rag-opt
```

## 🔥 Quick Start

### 1. Generate Training Questions

```python
from langchain.chat_models import init_chat_model
from rag_opt.rag import DatasetGenerator

# Initialize LLM
llm = init_chat_model(
    model="gpt-3.5-turbo",
    model_provider="openai",
    api_key="sk-***"
)

# Generate Q&A pairs from your documents
data_gen = DatasetGenerator(llm, dataset_path="./data")
dataset = data_gen.generate(10)
dataset.to_json("./rag_dataset.json")
```

### 2. Define Your Search Space

Create `rag_config.yaml`:

```yaml
chunk_size:
  bounds: [512, 1024]
  dtype: int

max_tokens:
  bounds: [256, 512]
  dtype: int

chunk_overlap:
  bounds: [0, 200]
  dtype: int

temperature:
  bounds: [0.0, 1.0]
  dtype: float

search_type:
  choices: ["similarity", "mmr", "hybrid"]

vector_store:
  choices:
    faiss: {}
    pinecone:
      api_key: "YOUR_API_KEY"
      index_name: "your-index"

embedding:
  choices:
    openai:
      api_key: "YOUR_API_KEY"
      models:
        - "text-embedding-3-large"
        - "text-embedding-ada-002"
    huggingface:
      models:
        - "all-MiniLM-L6-v2"

llm:
  choices:
    openai:
      api_key: "YOUR_API_KEY"
      models:
        - "gpt-4o"
        - "gpt-3.5-turbo"

k:
  bounds: [1, 10]
  dtype: int

use_reranker: false
```

### 3. Run Optimization

```python
from rag_opt.dataset import TrainDataset
from rag_opt.optimizer import Optimizer

# Load dataset
train_dataset = TrainDataset.from_json("rag_dataset.json")

# Initialize optimizer
optimizer = Optimizer(
    train_dataset=train_dataset,
    config_path="rag_config.yaml",
    verbose=True
)

# Find optimal configuration
best_config = optimizer.optimize(n_trials=3, best_one=True)
best_config.to_json()
```

### 4. Get Your Optimized Config

Output example:

```yaml
{
"chunk_size": 500
"max_tokens": 100
"chunk_overlap": 200
"search_type": "hybrid"
"k": "1"
"temperature": 1.0
"embedding":
  "provider": "openai"
  "model": "text-embedding-3-large"
"llm":
  "provider": "openai"
  "model": "gpt-4o"
"vector_store":
  "provider": "faiss"
"use_reranker": "true"
}
```

## How It Works

1. **Dataset Generation** - Create synthetic Q&A pairs from your documents
2. **Search Space Definition** - Configure which parameters to optimize
3. **Bayesian Optimization** - Intelligently sample and evaluate configurations
4. **Multi-Metric Evaluation** - Assess quality, performance, and safety
5. **Pareto-Optimal Results** - Get the best configurations for your priorities

## RAGOpt vs Alternatives

- **AutoRAG**: Uses Bayesian optimization instead of grid search like AutoRAG
- **Ragas**: Flexible evaluation framework, not rigid—bring your own metrics
- **Manual Tuning**: Systematic, data-driven approach saves time and improves results

## License

This project is licensed under the terms of the MIT license.
