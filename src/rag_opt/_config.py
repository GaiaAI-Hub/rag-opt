from typing import Any, Union, Optional, Literal,  NamedTuple
from rag_opt.rag._pricing import (LLMTokenCost, 
                                  EmbeddingCost, 
                                  RerankerCost, 
                                  RerankerPricingType, 
                                  VectorStoreCost,
                                )
from dataclasses import dataclass, asdict
import json 

VectorStoreProvider = Literal["faiss", "chroma", "pinecone", "weaviate"]
SearchType = Literal["similarity",   "mmr", "bm25", "tfidf", "hybrid"]
LLMProvider = Literal["openai", "anthropic", "huggingface", "azure", "deepseek"]
EmbeddingProvider = Literal["openai", "huggingface", "sentence-transformers", "claude"]
RerankerType = Literal["cross_encoder", "colbert", "bge"]
SearchSpaceType = Literal["continuous", "categorical", "boolean"]


@dataclass
class LLMConfig:
    """Configuration for LLM settings with multiple provider support"""
    provider: LLMProvider
    models: list[str]
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[dict[str, LLMTokenCost]] = None  # model_name -> pricing

    def __post_init__(self):
        """Validate LLM configuration"""
        if not self.models:
            raise ValueError("LLM model cannot be empty")
        if self.pricing is None:
            self.pricing = {model: LLMTokenCost(input=0.0, output=0.0) for model in self.models}

@dataclass
class VectorStoreConfig:
    """Configuration for vector store settings with multiple provider support"""
    provider: VectorStoreProvider
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    index_name: Optional[str] = None
    cloud_config: Optional[dict[str, Any]] = None
    pricing: Optional[VectorStoreCost] = None

    def __post_init__(self):
        """Initialize pricing with zeros if not provided"""
        if self.pricing is None:
            self.pricing = VectorStoreCost()

@dataclass
class EmbeddingConfig:
    """Configuration for embedding settings with multiple provider support"""
    provider: EmbeddingProvider
    models: list[str]
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[dict[str, EmbeddingCost]] = None  # model_name -> pricing

    def __post_init__(self):
        """Validate embedding configuration"""
        if not self.models:
            raise ValueError("Embedding model cannot be empty")
        # Initialize pricing with zeros if not provided
        if self.pricing is None:
            self.pricing = {model: EmbeddingCost() for model in self.models}

@dataclass
class RerankerConfig:
    """Configuration for reranker settings"""
    provider: RerankerType
    models: list[str]
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[dict[str, RerankerCost]] = None  # model_name -> pricing

    def __post_init__(self):
        """Validate reranker configuration"""
        if not self.models:
            raise ValueError("Reranker model cannot be empty")
        if self.pricing is None:
            self.pricing = {
                model: RerankerCost(
                    pricing_type=RerankerPricingType.FREE, 
                    cost_per_unit=0.0
                ) for model in self.models
            }

class AIModel(NamedTuple):
    provider: EmbeddingProvider
    model: str
    api_key: Optional[str]
    api_base: Optional[str]
    pricing: Optional[Union[LLMTokenCost, EmbeddingCost, RerankerCost]] = None

class EmbeddingModel(AIModel):
    """Embedding model configuration"""
    pass

class LLMModel(AIModel):
    """LLM model configuration"""
    pass

class RerankerModel(AIModel):
    """Reranker model configuration"""
    pass

class VectorStoreItem(NamedTuple):
    provider: VectorStoreProvider
    index_name: Optional[str] = None
    api_key: Optional[str] = None
    pricing: Optional[VectorStoreCost] = None

@dataclass
class RAGConfig:
    """Individual RAG configuration instance (a sample from the search space)"""
    chunk_size: int
    max_tokens: int
    chunk_overlap: int
    search_type: SearchType
    k: int
    temperature: float
    embedding: EmbeddingModel
    llm: LLMModel
    vector_store: VectorStoreItem
    use_reranker: Optional[bool] = False
    reranker: Optional[RerankerModel] = None

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        return f"""
                chunk_size={self.chunk_size},
                max_tokens={self.max_tokens},
                chunk_overlap={self.chunk_overlap},
                search_type={self.search_type},
                vector_store={self.vector_store},
                embedding={self.embedding},
                k={self.k},
                temperature={self.temperature},
                use_reranker={self.use_reranker},
                reranker={self.reranker},
                llm={self.llm}
            """
