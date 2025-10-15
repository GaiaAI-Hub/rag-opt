from typing import Optional, Union, Literal, TypeAlias
from langchain_core.messages.ai import UsageMetadata
from langchain.schema import Document
from dataclasses import dataclass
from enum import Enum
import tiktoken
from loguru import logger

@dataclass
class LLMTokenCost:
    """Pricing structure for LLM tokens (per 1K tokens)"""
    input: float
    output: float
    cache_read: Optional[float] = None
    cache_creation: Optional[float] = None
    reasoning: Optional[float] = None
    audio: Optional[float] = None

    def cost_for(self, usage: UsageMetadata) -> float:
        """Calculate total cost from usage metadata"""
        cost = 0.0
        
        # Base input/output costs
        cost += (usage.get("input_tokens", 0) / 1000) * self.input
        cost += (usage.get("output_tokens", 0) / 1000) * self.output
        
        # Output token details (reasoning, audio)
        output_details = usage.get("output_token_details", {})
        if "reasoning" in output_details and self.reasoning:
            cost += (output_details["reasoning"] / 1000) * self.reasoning
        if "audio" in output_details and self.audio:
            cost += (output_details["audio"] / 1000) * self.audio

        # Input token details (cache, audio)
        input_details = usage.get("input_token_details", {})
        if "cache_read" in input_details and self.cache_read:
            cost += (input_details["cache_read"] / 1000) * self.cache_read
        if "cache_creation" in input_details and self.cache_creation:
            cost += (input_details["cache_creation"] / 1000) * self.cache_creation
        if "audio" in input_details and self.audio:
            cost += (input_details["audio"] / 1000) * self.audio
            
        return cost


@dataclass
class EmbeddingCost:
    """Pricing for embedding models (per 1K tokens)"""
    cost_per_1k_tokens: float

    def cost_for(self, usage: UsageMetadata) -> float:
        """Calculate embedding cost from usage metadata"""
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return (total_tokens / 1000) * self.cost_per_1k_tokens


class RerankerPricingType(Enum):
    """Different pricing models for rerankers"""
    TOKEN_BASED = "token_based"
    REQUEST_BASED = "request_based"
    DOCUMENT_BASED = "document_based"
    FREE = "free"


@dataclass
class RerankerCost:
    """Universal cost calculator for different reranker pricing models"""
    pricing_type: RerankerPricingType
    cost_per_unit: float  # Cost per token/request/document
    cost_unit: int = 1000  # For token-based: per 1K tokens, for others: per 1 unit

    def cost_for(self, docs: list[Document], num_requests: int = 1) -> float:
        """
        Calculate the total cost for reranking
        
        Args:
            docs: list of Document objects to be reranked
            num_requests: Number of API calls (for batching scenarios)
            
        Returns:
            Total cost in USD
        """
        if self.pricing_type == RerankerPricingType.FREE:
            return 0.0
        
        elif self.pricing_type == RerankerPricingType.REQUEST_BASED:
            return num_requests * self.cost_per_unit
        
        elif self.pricing_type == RerankerPricingType.DOCUMENT_BASED:
            return len(docs) * (self.cost_per_unit / self.cost_unit)
        
        elif self.pricing_type == RerankerPricingType.TOKEN_BASED:
            total_tokens = self._count_tokens(docs)
            return (total_tokens / self.cost_unit) * self.cost_per_unit
        
        else:
            raise ValueError(f"Unknown pricing type: {self.pricing_type}")
    
    def _count_tokens(self, docs: list[Document]) -> int:
        """Count total tokens across all documents"""
        total_tokens = 0
        
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            for doc in docs:
                content = self._extract_content(doc)
                if content:
                    total_tokens += len(encoding.encode(content))
        except Exception:
            # Fallback to simple estimation
            total_chars = sum(len(self._extract_content(doc)) for doc in docs)
            total_tokens = total_chars // 4  # 1 token ≈ 4 chars
        
        return total_tokens
    
    def _extract_content(self, doc: Union[Document, str]) -> str:
        """Extract text content from document"""
        if isinstance(doc, str):
            return doc
        elif hasattr(doc, 'page_content'):
            return doc.page_content or ""
        else:
            return str(doc)


@dataclass
class VectorStoreCost:
    """Pricing for vector store operations"""
    storage_per_gb_month: float = 0.0  # Cost per GB per month
    read_operations_per_1k: float = 0.0  # Cost per 1K read operations
    write_operations_per_1k: float = 0.0  # Cost per 1K write operations
    query_per_1k: float = 0.0  # Cost per 1K queries

    def cost_for(self, storage_gb: float = 0, read_ops: int = 0, 
                 write_ops: int = 0, queries: int = 0, months: float = 1.0) -> float:
        """
        Calculate vector store costs
        
        Args:
            storage_gb: Storage used in GB
            read_ops: Number of read operations
            write_ops: Number of write operations
            queries: Number of queries
            months: Time period for storage cost calculation
            
        Returns:
            Total cost in USD
        """
        cost = 0.0
        cost += storage_gb * self.storage_per_gb_month * months
        cost += (read_ops / 1000) * self.read_operations_per_1k
        cost += (write_ops / 1000) * self.write_operations_per_1k
        cost += (queries / 1000) * self.query_per_1k
        return cost


ServiceType: TypeAlias = Literal["llm", "embedding", "reranker", "vector_store", "all"]

from typing import Optional

class PricingRegistry:
    """Unified pricing registry for all service types required for RAG pipeline"""

    # 🔹 Shared registries across all instances
    llm_registry: dict[str, dict[str, LLMTokenCost]] = {}
    embedding_registry: dict[str, dict[str, EmbeddingCost]] = {}
    reranker_registry: dict[str, dict[str, RerankerCost]] = {}
    vector_store_registry: dict[str, VectorStoreCost] = {}  # provider > cost 

    # ---------------- LLM ----------------
    @classmethod
    def add_llm_provider(cls, provider: str, models: dict[str, LLMTokenCost]) -> None:
        """Add LLM provider with its models"""
        if provider in cls.llm_registry:
            cls.llm_registry[provider].update(models)
        else:
            cls.llm_registry[provider] = models
    
    @classmethod
    def get_llm_cost(cls, provider: str, model: str) -> Optional[LLMTokenCost]:
        """Get LLM pricing for a specific provider/model"""
        return cls.llm_registry.get(provider, {}).get(model)
    
    @classmethod
    def calculate_llm_cost(cls, provider: str, model: str, usage: UsageMetadata) -> Optional[float]:
        """Calculate LLM cost for usage"""
        pricing = cls.get_llm_cost(provider, model)
        return pricing.cost_for(usage) if pricing else 0.0
    
    # ---------------- Embedding ----------------
    @classmethod
    def add_embedding_provider(cls, provider: str, models: dict[str, EmbeddingCost]) -> None:
        if provider in cls.embedding_registry:
            cls.embedding_registry[provider].update(models)
        else:
            cls.embedding_registry[provider] = models
    
    @classmethod
    def get_embedding_cost(cls, provider: str, model: str) -> Optional[EmbeddingCost]:
        return cls.embedding_registry.get(provider, {}).get(model)
    
    @classmethod
    def calculate_embedding_cost(cls, provider: str, model: str, usage: UsageMetadata) -> Optional[float]:
        pricing = cls.get_embedding_cost(provider, model)
        return pricing.cost_for(usage) if pricing else None
    
    # ---------------- Reranker ----------------
    @classmethod
    def add_reranker_provider(cls, provider: str, rerankers: dict[str, RerankerCost]) -> None:
        if provider in cls.reranker_registry:
            cls.reranker_registry[provider].update(rerankers)
        else:
            cls.reranker_registry[provider] = rerankers
    
    @classmethod
    def get_reranker_cost(cls, provider: str, reranker: str) -> Optional[RerankerCost]:
        return cls.reranker_registry.get(provider, {}).get(reranker)
    
    @classmethod
    def calculate_reranker_cost(cls, provider: str, model: str, docs: list[Document], num_requests: int = 1) -> Optional[float]:
        pricing = cls.get_reranker_cost(provider, model)
        return pricing.cost_for(docs, num_requests) if pricing else None
    
    # ---------------- Vector store ----------------
    @classmethod
    def add_vector_store_provider(cls, provider: str, cost: VectorStoreCost) -> None:
        cls.vector_store_registry[provider] = cost
    
    @classmethod
    def get_vector_store_cost(cls, provider: str) -> Optional[VectorStoreCost]:
        return cls.vector_store_registry.get(provider)
    
    @classmethod
    def calculate_vector_store_cost(cls, provider: str, storage_gb: float = 0,
                                   read_ops: int = 0, write_ops: int = 0, queries: int = 0,
                                   months: float = 1.0) -> Optional[float]:
        pricing = cls.get_vector_store_cost(provider)
        return pricing.cost_for(storage_gb, read_ops, write_ops, queries, months) if pricing else None
    
    # ---------------- Utils ----------------
    @classmethod
    def list_providers(cls, service_type: ServiceType = "all") -> dict[str, list[str]]:
        providers = {}
        if service_type in ("all", "llm"):
            providers["llm"] = list(cls.llm_registry.keys())
        if service_type in ("all", "embedding"):
            providers["embedding"] = list(cls.embedding_registry.keys())
        if service_type in ("all", "reranker"):
            providers["reranker"] = list(cls.reranker_registry.keys())
        if service_type in ("all", "vector_store"):
            providers["vector_store"] = list(cls.vector_store_registry.keys())
        return providers
    
    @classmethod
    def list_models(cls, provider: str, service_type: ServiceType = "all") -> dict[str, list[str]]:
        models = {}
        if service_type in ("all", "llm") and provider in cls.llm_registry:
            models["llm"] = list(cls.llm_registry[provider].keys())
        if service_type in ("all", "embedding") and provider in cls.embedding_registry:
            models["embedding"] = list(cls.embedding_registry[provider].keys())
        if service_type in ("all", "reranker") and provider in cls.reranker_registry:
            models["reranker"] = list(cls.reranker_registry[provider].keys())
        return models
