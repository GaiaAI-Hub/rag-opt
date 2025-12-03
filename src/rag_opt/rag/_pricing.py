from typing import Optional, Literal
from langchain_core.messages.ai import UsageMetadata
from langchain.schema import Document
from dataclasses import dataclass
from enum import Enum
import tiktoken
from loguru import logger
import os
from functools import lru_cache


@dataclass
class LLMTokenCost:
    """Pricing for LLM tokens (per 1K tokens)"""
    input: float
    output: float
    cache_read: Optional[float] = None
    cache_creation: Optional[float] = None
    reasoning: Optional[float] = None
    audio: Optional[float] = None

    def cost_for(self, usage: UsageMetadata) -> float:
        """Calculate total cost from usage metadata"""
        cost = 0.0
        
        # Base costs
        cost += (usage.get("input_tokens", 0) / 1000) * self.input
        cost += (usage.get("output_tokens", 0) / 1000) * self.output
        
        # Output token details
        output_details = usage.get("output_token_details", {})
        if "reasoning" in output_details and self.reasoning:
            cost += (output_details["reasoning"] / 1000) * self.reasoning
        if "audio" in output_details and self.audio:
            cost += (output_details["audio"] / 1000) * self.audio

        # Input token details
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
    cost_per_1k_tokens: float = 0.0

    def cost_for(self, usage: UsageMetadata) -> float:
        """Calculate embedding cost"""
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return (total_tokens / 1000) * self.cost_per_1k_tokens


class RerankerPricingType(Enum):
    """Reranker pricing models"""
    TOKEN_BASED = "token_based"
    REQUEST_BASED = "request_based"
    DOCUMENT_BASED = "document_based"
    FREE = "free"


@dataclass
class RerankerCost:
    """Universal reranker pricing"""
    pricing_type: RerankerPricingType
    cost_per_unit: float = 0.0
    cost_unit: int = 1000

    def cost_for(self, docs: list[Document], num_requests: int = 1) -> float:
        """Calculate reranking cost"""
        if self.pricing_type == RerankerPricingType.FREE:
            return 0.0
        
        if self.pricing_type == RerankerPricingType.REQUEST_BASED:
            return num_requests * self.cost_per_unit
        
        if self.pricing_type == RerankerPricingType.DOCUMENT_BASED:
            return len(docs) * (self.cost_per_unit / self.cost_unit)
        
        if self.pricing_type == RerankerPricingType.TOKEN_BASED:
            total_tokens = self._count_tokens(docs)
            return (total_tokens / self.cost_unit) * self.cost_per_unit
        
        raise ValueError(f"Unknown pricing type: {self.pricing_type}")
    
    @staticmethod
    def _count_tokens(docs: list[Document]) -> int:
        """Count tokens across documents"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return sum(len(encoding.encode(RerankerCost._extract_content(doc))) for doc in docs)
        except Exception:
            total_chars = sum(len(RerankerCost._extract_content(doc)) for doc in docs)
            return total_chars // 4
    
    @staticmethod
    def _extract_content(doc: Document | str) -> str:
        """Extract text from document"""
        if isinstance(doc, str):
            return doc
        return getattr(doc, 'page_content', str(doc)) or ""


@dataclass
class VectorStoreCost:
    """Pricing for vector store operations"""
    storage_per_gb_month: float = 0.0
    read_operations_per_1k: float = 0.0
    write_operations_per_1k: float = 0.0
    query_per_1k: float = 0.0

    def cost_for(
        self, 
        storage_gb: float = 0, 
        read_ops: int = 0, 
        write_ops: int = 0, 
        queries: int = 0, 
        months: float = 1.0
    ) -> float:
        """Calculate vector store costs"""
        cost = storage_gb * self.storage_per_gb_month * months
        cost += (read_ops / 1000) * self.read_operations_per_1k
        cost += (write_ops / 1000) * self.write_operations_per_1k
        cost += (queries / 1000) * self.query_per_1k
        return cost


ServiceType = Literal["llm", "embedding", "reranker", "vector_store", "all"]


class AIGatewayClient:
    """Client for fetching pricing from Vercel AI Gateway"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def fetch_model_pricing(provider: str, model: str) -> Optional[dict]:
        """
        Fetch model pricing from Vercel AI Gateway.
        Results are cached to avoid repeated API calls.
        
        Returns dict with pricing info or None if unavailable.
        """
        try:
            from openai import OpenAI
            
            api_key = os.getenv('AI_GATEWAY_API_KEY')
            if not api_key:
                logger.debug("AI_GATEWAY_API_KEY not set, skipping gateway pricing fetch")
                return None
            
            client = OpenAI(
                api_key=api_key,
                base_url='https://ai-gateway.vercel.sh/v1'
            )
            
            # Support both "provider/model" and just "model" formats
            model_id = f"{provider}/{model}" if "/" not in model else model
            
            model_info = client.models.retrieve(model_id)
            
            if hasattr(model_info, 'pricing') and model_info.pricing:
                logger.success(f"Fetched pricing for {model_id} from AI Gateway")
                return model_info.pricing
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not fetch pricing from AI Gateway for {provider}/{model}: {e}")
            return None
    
    @staticmethod
    def parse_llm_pricing(pricing_dict: dict) -> Optional[LLMTokenCost]:
        """Parse AI Gateway pricing response into LLMTokenCost"""
        try:
            return LLMTokenCost(
                input=float(pricing_dict.get('input', 0)) * 1000,  # Convert to per 1K tokens
                output=float(pricing_dict.get('output', 0)) * 1000,
                cache_read=float(pricing_dict.get('input_cache_read', 0)) * 1000 if 'input_cache_read' in pricing_dict else None,
                cache_creation=float(pricing_dict.get('input_cache_write', 0)) * 1000 if 'input_cache_write' in pricing_dict else None,
            )
        except Exception as e:
            logger.warning(f"Failed to parse LLM pricing: {e}")
            return None
    
    @staticmethod
    def parse_embedding_pricing(pricing_dict: dict) -> Optional[EmbeddingCost]:
        """Parse AI Gateway pricing response into EmbeddingCost"""
        try:
            # Gateway typically returns per-token pricing, convert to per-1K
            cost = float(pricing_dict.get('input', 0)) * 1000
            return EmbeddingCost(cost_per_1k_tokens=cost)
        except Exception as e:
            logger.warning(f"Failed to parse embedding pricing: {e}")
            return None


class PricingRegistry:
    """Unified pricing registry with AI Gateway fallback"""

    _llm_registry: dict[str, dict[str, LLMTokenCost]] = {}
    _embedding_registry: dict[str, dict[str, EmbeddingCost]] = {}
    _reranker_registry: dict[str, dict[str, RerankerCost]] = {}
    _vector_store_registry: dict[str, VectorStoreCost] = {}

    # ============ LLM ============
    @classmethod
    def add_llm_provider(cls, provider: str, models: dict[str, LLMTokenCost]) -> None:
        """Add LLM provider with pricing"""
        if provider in cls._llm_registry:
            cls._llm_registry[provider].update(models)
        else:
            cls._llm_registry[provider] = models
    
    @classmethod
    def get_llm_cost(cls, provider: str, model: str) -> Optional[LLMTokenCost]:
        """
        Get LLM pricing with AI Gateway fallback.
        First checks local registry, then fetches from AI Gateway if needed.
        """
        # Check local registry first
        local_pricing = cls._llm_registry.get(provider, {}).get(model)
        if local_pricing:
            return local_pricing
        
        # Try AI Gateway
        gateway_pricing = AIGatewayClient.fetch_model_pricing(provider, model)
        if gateway_pricing:
            parsed = AIGatewayClient.parse_llm_pricing(gateway_pricing)
            if parsed:
                # Cache it locally for future use
                cls.add_llm_provider(provider, {model: parsed})
                return parsed
        
        logger.debug(f"No pricing found for {provider}/{model}")
        return None
    
    @classmethod
    def calculate_llm_cost(cls, provider: str, model: str, usage: UsageMetadata) -> float:
        """Calculate LLM cost with auto-fetch from AI Gateway"""
        pricing = cls.get_llm_cost(provider, model)
        return pricing.cost_for(usage) if pricing else 0.0
    
    # ============ Embedding ============
    @classmethod
    def add_embedding_provider(cls, provider: str, models: dict[str, EmbeddingCost]) -> None:
        """Add embedding provider with pricing"""
        if provider in cls._embedding_registry:
            cls._embedding_registry[provider].update(models)
        else:
            cls._embedding_registry[provider] = models
    
    @classmethod
    def get_embedding_cost(cls, provider: str, model: str) -> Optional[EmbeddingCost]:
        """Get embedding pricing with AI Gateway fallback"""
        # Check local registry first
        local_pricing = cls._embedding_registry.get(provider, {}).get(model)
        if local_pricing:
            return local_pricing
        
        # Try AI Gateway
        gateway_pricing = AIGatewayClient.fetch_model_pricing(provider, model)
        if gateway_pricing:
            parsed = AIGatewayClient.parse_embedding_pricing(gateway_pricing)
            if parsed:
                cls.add_embedding_provider(provider, {model: parsed})
                return parsed
        
        logger.debug(f"No pricing found for {provider}/{model}")
        return None
    
    @classmethod
    def calculate_embedding_cost(cls, provider: str, model: str, usage: UsageMetadata) -> float:
        """Calculate embedding cost with auto-fetch from AI Gateway"""
        pricing = cls.get_embedding_cost(provider, model)
        return pricing.cost_for(usage) if pricing else 0.0
    
    # ============ Reranker ============
    @classmethod
    def add_reranker_provider(cls, provider: str, rerankers: dict[str, RerankerCost]) -> None:
        """Add reranker provider with pricing"""
        if provider in cls._reranker_registry:
            cls._reranker_registry[provider].update(rerankers)
        else:
            cls._reranker_registry[provider] = rerankers
    
    @classmethod
    def get_reranker_cost(cls, provider: str, reranker: str) -> Optional[RerankerCost]:
        """Get reranker pricing (local registry only)"""
        return cls._reranker_registry.get(provider, {}).get(reranker)
    
    @classmethod
    def calculate_reranker_cost(
        cls, 
        provider: str, 
        model: str, 
        docs: list[Document], 
        num_requests: int = 1
    ) -> float:
        """Calculate reranker cost"""
        pricing = cls.get_reranker_cost(provider, model)
        return pricing.cost_for(docs, num_requests) if pricing else 0.0
    
    # ============ Vector Store ============
    @classmethod
    def add_vector_store_provider(cls, provider: str, cost: VectorStoreCost) -> None:
        """Add vector store provider with pricing"""
        cls._vector_store_registry[provider] = cost
    
    @classmethod
    def get_vector_store_cost(cls, provider: str) -> Optional[VectorStoreCost]:
        """Get vector store pricing"""
        return cls._vector_store_registry.get(provider)
    
    @classmethod
    def calculate_vector_store_cost(
        cls, 
        provider: str, 
        storage_gb: float = 0,
        read_ops: int = 0, 
        write_ops: int = 0, 
        queries: int = 0,
        months: float = 1.0
    ) -> float:
        """Calculate vector store cost"""
        pricing = cls.get_vector_store_cost(provider)
        return pricing.cost_for(storage_gb, read_ops, write_ops, queries, months) if pricing else 0.0
    
    # ============ Utilities ============
    @classmethod
    def list_providers(cls, service_type: ServiceType = "all") -> dict[str, list[str]]:
        """List providers by service type"""
        providers = {}
        if service_type in ("all", "llm"):
            providers["llm"] = list(cls._llm_registry.keys())
        if service_type in ("all", "embedding"):
            providers["embedding"] = list(cls._embedding_registry.keys())
        if service_type in ("all", "reranker"):
            providers["reranker"] = list(cls._reranker_registry.keys())
        if service_type in ("all", "vector_store"):
            providers["vector_store"] = list(cls._vector_store_registry.keys())
        return providers
    
    @classmethod
    def list_models(cls, provider: str, service_type: ServiceType = "all") -> dict[str, list[str]]:
        """List models for provider"""
        models = {}
        if service_type in ("all", "llm") and provider in cls._llm_registry:
            models["llm"] = list(cls._llm_registry[provider].keys())
        if service_type in ("all", "embedding") and provider in cls._embedding_registry:
            models["embedding"] = list(cls._embedding_registry[provider].keys())
        if service_type in ("all", "reranker") and provider in cls._reranker_registry:
            models["reranker"] = list(cls._reranker_registry[provider].keys())
        return models
    
    @classmethod
    def clear_all(cls) -> None:
        """Clear all pricing data (useful for testing)"""
        cls._llm_registry.clear()
        cls._embedding_registry.clear()
        cls._reranker_registry.clear()
        cls._vector_store_registry.clear()
        # Clear AI Gateway cache
        AIGatewayClient.fetch_model_pricing.cache_clear()
