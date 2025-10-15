from typing import Any, Union, Optional, Literal
from rag_opt.rag._pricing import (LLMTokenCost, 
                                  EmbeddingCost, 
                                  RerankerCost, 
                                  RerankerPricingType, 
                                  VectorStoreCost,
                                )
from dataclasses import dataclass, asdict
import json 
import time
import os 

VectorStoreProvider = Literal["faiss", "chroma", "pinecone", "weaviate"]
SearchType = Literal["similarity", "mmr", "bm25", "tfidf", "hybrid"]
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
    pricing: Optional[dict[str, LLMTokenCost]] = None

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
    pricing: Optional[dict[str, EmbeddingCost]] = None

    def __post_init__(self):
        """Validate embedding configuration"""
        if not self.models:
            raise ValueError("Embedding model cannot be empty")
        if self.pricing is None:
            self.pricing = {model: EmbeddingCost() for model in self.models}

@dataclass
class RerankerConfig:
    """Configuration for reranker settings"""
    provider: RerankerType
    models: list[str]
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[dict[str, RerankerCost]] = None

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

@dataclass
class AIModel:
    provider: Union[EmbeddingProvider, LLMProvider, RerankerType]
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[Union[LLMTokenCost, EmbeddingCost, RerankerCost]] = None

@dataclass
class EmbeddingModel:
    """Embedding model configuration"""
    provider: EmbeddingProvider
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[EmbeddingCost] = None

@dataclass
class LLMModel:
    """LLM model configuration"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[LLMTokenCost] = None

@dataclass
class RerankerModel:
    """Reranker model configuration"""
    provider: RerankerType
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    pricing: Optional[RerankerCost] = None

@dataclass
class VectorStoreItem:
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
    
    @staticmethod
    def _remove_sensitive_fields(data: dict) -> dict:
        """Recursively remove sensitive fields from nested dictionaries"""
        if not isinstance(data, dict):
            return data
        
        cleaned = {}
        for key, value in data.items():
            if key in ("api_key", "pricing"):
                continue
            
            if isinstance(value, dict):
                cleaned[key] = RAGConfig._remove_sensitive_fields(value)
            elif value:
                cleaned[key] = value
        
        return cleaned
    
    @classmethod
    def from_json(cls, file_path: str):
        """Load RAG configuration from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reconstruct nested dataclass objects
        if "embedding" in data and isinstance(data["embedding"], dict):
            data["embedding"] = EmbeddingModel(**data["embedding"])
        if "llm" in data and isinstance(data["llm"], dict):
            data["llm"] = LLMModel(**data["llm"])
        if "vector_store" in data and isinstance(data["vector_store"], dict):
            data["vector_store"] = VectorStoreItem(**data["vector_store"])
        if "reranker" in data and data["reranker"] and isinstance(data["reranker"], dict):
            data["reranker"] = RerankerModel(**data["reranker"])
        
        return cls(**data)
    
    def to_json(self, path: str = "./best_config.json"):
        """Save configuration to JSON file"""
        if os.path.exists(path):
            base = path.replace(".json", "")
            path = f"{base}-{int(time.time())}.json"
        elif not path.endswith(".json"):
            path += ".json"
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        
    
    def to_dict(self, remove_sensitive: bool = True) -> dict:
        """Convert configuration to dictionary"""
        data = asdict(self)
        return self._remove_sensitive_fields(data) if remove_sensitive else data
    
    def __repr__(self):
        return f"""RAGConfig(
            chunk_size={self.chunk_size},
            max_tokens={self.max_tokens},
            chunk_overlap={self.chunk_overlap},
            search_type={self.search_type},
            vector_store={self.vector_store.provider},
            embedding={self.embedding.model},
            k={self.k},
            temperature={self.temperature},
            use_reranker={self.use_reranker},
            reranker={self.reranker.model if self.reranker else None},
            llm={self.llm.model}
        )"""