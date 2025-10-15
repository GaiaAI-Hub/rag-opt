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
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]


class CohereReranker(BaseReranker):
    """Cohere reranker implementation"""
    
    def _initialize_model(self):
        from langchain.retrievers.document_compressors import CohereRerank
        
        api_key = self.kwargs.get("api_key")
        model = self.kwargs.get("model_name", "rerank-english-v3.0")
        
        if not api_key:
            raise ValueError("api_key is required for Cohere reranker")
        
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
        
        reranked_docs = [documents[r["id"]] for r in results[:top_k]]
        return reranked_docs


class JinaReranker(BaseReranker):
    """Jina AI reranker implementation"""
    
    def _initialize_model(self):
        from langchain_community.document_compressors import JinaRerank
        
        api_key = self.kwargs.get("api_key")
        model = self.kwargs.get("model_name", "jina-reranker-v1-base-en")
        
        if not api_key:
            raise ValueError("api_key is required for Jina reranker")
        
        return JinaRerank(jina_api_key=api_key, model=model, top_n=100)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        return self.model.compress_documents(documents, query)[:top_k]


class VoyageAIReranker(BaseReranker):
    """Voyage AI reranker implementation"""
    
    def _initialize_model(self):
        from langchain_voyageai import VoyageAIRerank
        
        api_key = self.kwargs.get("api_key")
        model = self.kwargs.get("model_name", "rerank-lite-1")
        
        if not api_key:
            raise ValueError("api_key is required for Voyage AI reranker")
        
        return VoyageAIRerank(voyageai_api_key=api_key, model=model, top_k=100)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        return self.model.compress_documents(documents, query)[:top_k]


class RankGPTReranker(BaseReranker):
    """RankGPT reranker using LLM for ranking"""
    
    def _initialize_model(self):
        from langchain.retrievers.document_compressors import LLMChainFilter
        from langchain_openai import ChatOpenAI
        
        api_key = self.kwargs.get("api_key")
        model = self.kwargs.get("model_name", "gpt-3.5-turbo")
        
        llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
        return LLMChainFilter.from_llm(llm)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        return self.model.compress_documents(documents, query)[:top_k]


class ColBERTReranker(BaseReranker):
    """ColBERT reranker implementation"""
    
    def _initialize_model(self):
        from ragatouille import RAGPretrainedModel
        
        model_name = self.kwargs.get("model_name", "colbert-ir/colbertv2.0")
        return RAGPretrainedModel.from_pretrained(model_name) 
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        docs_text = [doc.page_content for doc in documents]
        results = self.model.rerank(query, docs_text, k=top_k)
        
        reranked_docs = [documents[r["document_id"]] for r in results]
        return reranked_docs


class BM25Reranker(BaseReranker):
    """BM25 statistical reranker"""
    
    def _initialize_model(self):
        from rank_bm25 import BM25Okapi
        return {"bm25": None, "tokenized_corpus": None}
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        from rank_bm25 import BM25Okapi
        
        # Tokenize corpus
        corpus = [doc.page_content for doc in documents]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Get scores
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Sort and return top_k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]



class PineconeReranker(BaseReranker):
    """Pinecone Inference reranker"""
    
    def _initialize_model(self):
        from langchain_pinecone import PineconeRerank
        
        api_key = self.kwargs.get("api_key")
        model = self.kwargs.get("model_name", "bge-reranker-v2-m3")
        
        if not api_key:
            raise ValueError("api_key is required for Pinecone reranker")
        
        return PineconeRerank(api_key=api_key, model=model)
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        docs_text = [doc.page_content for doc in documents]
        results = self.model.rerank(query=query, documents=docs_text, top_n=top_k)
        
        reranked_docs = [documents[r.index] for r in results]
        return reranked_docs


# Registry of available rerankers
RERANKER_REGISTRY = {
    "cohere": CohereReranker,
    "flashrank": FlashRankReranker,
    "jina": JinaReranker,
    "voyageai": VoyageAIReranker,
    "rankgpt": RankGPTReranker,
    "colbert": ColBERTReranker,
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
        provider: Reranker provider (e.g., 'cohere', 'flashrank', 'cross_encoder')
        model_name: Specific model to use (optional, provider-dependent)
        api_key: API key for providers that require it
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Initialized reranker instance
    
    Raises:
        ValueError: If provider is not supported
    
    Examples:
        >>> reranker = init_reranker("cohere", api_key="your-key")
        >>> reranker = init_reranker("cross_encoder", model_name="BAAI/bge-reranker-base")
        >>> reranker = init_reranker("flashrank")
    """
    provider = provider.lower()
    
    if provider not in RERANKER_REGISTRY:
        available = ", ".join(RERANKER_REGISTRY.keys())
        raise ValueError(f"Unsupported reranker provider: {provider}. Available: {available}")
    
    reranker_class = RERANKER_REGISTRY[provider]
    
    config = kwargs.copy()
    if model_name:
        config["model_name"] = model_name
    if api_key:
        config["api_key"] = api_key
    
    reranker = reranker_class(**config)
    logger.success(f"{provider} Reranker Loaded successfully")
    return reranker
