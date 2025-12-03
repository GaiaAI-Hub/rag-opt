from langchain_core.documents import BaseDocumentCompressor, Document
from abc import ABC, abstractmethod
from loguru import logger
from typing import Optional


class BaseReranker(ABC):
    """Abstract base class for all rerankers"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self) -> BaseDocumentCompressor:
        """Initialize the specific reranker model"""
        pass
    
    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        """Rerank documents based on query relevance"""
        pass


class HuggingFaceCrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using HuggingFace models"""
    
    def _initialize_model(self):
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        model_name = self.kwargs.get("model_name", "BAAI/bge-reranker-base")
        return HuggingFaceCrossEncoder(model_name=model_name) 
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.score(pairs)
        
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]


class CohereReranker(BaseReranker):
    """Cohere reranker implementation"""
    
    def _initialize_model(self):
        from langchain.retrievers.document_compressors import CohereRerank
        
        api_key = self.kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key is required for Cohere reranker")
        
        model = self.kwargs.get("model_name", "rerank-english-v3.0")
        return CohereRerank(cohere_api_key=api_key, model=model, top_n=100)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        return self.model.compress_documents(documents, query)[:top_k]


class FlashRankReranker(BaseReranker):
    """FlashRank reranker implementation"""
    
    def _initialize_model(self):
        from flashrank import Ranker
        model = self.kwargs.get("model_name", "ms-marco-MiniLM-L-12-v2")
        return Ranker(model_name=model)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
        results = self.model.rerank(query, passages)
        return [documents[r["id"]] for r in results[:top_k]]


class JinaReranker(BaseReranker):
    """Jina AI reranker implementation"""
    
    def _initialize_model(self):
        from langchain_community.document_compressors import JinaRerank
        
        api_key = self.kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key is required for Jina reranker")
        
        model = self.kwargs.get("model_name", "jina-reranker-v1-base-en")
        return JinaRerank(jina_api_key=api_key, model=model, top_n=100)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        return self.model.compress_documents(documents, query)[:top_k]


class VoyageAIReranker(BaseReranker):
    """Voyage AI reranker implementation"""
    
    def _initialize_model(self):
        from langchain_voyageai import VoyageAIRerank
        
        api_key = self.kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key is required for Voyage AI reranker")
        
        model = self.kwargs.get("model_name", "rerank-lite-1")
        return VoyageAIRerank(voyageai_api_key=api_key, model=model, top_k=100)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        return self.model.compress_documents(documents, query)[:top_k]


class BM25Reranker(BaseReranker):
    """BM25 statistical reranker"""
    
    def _initialize_model(self):
        return None  # Initialized on-the-fly during rerank
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        from rank_bm25 import BM25Okapi
        
        # Tokenize corpus and query
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        
        # Return top-k documents
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]


class PineconeReranker(BaseReranker):
    """Pinecone Inference reranker"""
    
    def _initialize_model(self):
        from langchain_pinecone import PineconeRerank
        
        api_key = self.kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key is required for Pinecone reranker")
        
        model = self.kwargs.get("model_name", "bge-reranker-v2-m3")
        return PineconeRerank(api_key=api_key, model=model)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        docs_text = [doc.page_content for doc in documents]
        results = self.model.rerank(query=query, documents=docs_text, top_n=top_k)
        return [documents[r.index] for r in results]


# Registry of available rerankers
RERANKER_REGISTRY = {
    "cohere": CohereReranker,
    "flashrank": FlashRankReranker,
    "jina": JinaReranker,
    "voyageai": VoyageAIReranker,
    "bm25": BM25Reranker,
    "pinecone": PineconeReranker,
    "huggingface": HuggingFaceCrossEncoderReranker,
}


def init_reranker(
    provider: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseReranker:
    """
    Initialize a reranker based on provider.
    
    Args:
        provider: Reranker provider (cohere, flashrank, jina, voyageai, bm25, pinecone, huggingface)
        model_name: Specific model to use (optional)
        api_key: API key for cloud providers
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Initialized reranker instance
    
    Examples:
        >>> reranker = init_reranker("cohere", api_key="your-key")
        >>> reranker = init_reranker("huggingface", model_name="BAAI/bge-reranker-base")
        >>> reranker = init_reranker("bm25")  # No API key needed
    """
    provider = provider.lower()
    
    if provider not in RERANKER_REGISTRY:
        available = ", ".join(RERANKER_REGISTRY.keys())
        raise ValueError(f"Unsupported reranker: {provider}. Available: {available}")
    
    # Build config
    config = kwargs.copy()
    if model_name:
        config["model_name"] = model_name
    if api_key:
        config["api_key"] = api_key
    
    # Initialize reranker
    reranker_class = RERANKER_REGISTRY[provider]
    reranker = reranker_class(**config)
    
    logger.success(f"{provider.capitalize()} reranker loaded successfully")
    return reranker