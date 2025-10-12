from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from typing import Any, Optional
from pydantic import PrivateAttr, Field
from typing_extensions import override
from rag_opt._config import SearchType
from langchain.schema import Document
from loguru import logger
class Retriever(BaseRetriever):
    """
    Enhanced retrieval with multiple search strategies.
    
    Supports:
    - Semantic search (vector similarity, MMR)
    - Lexical search (BM25, TF-IDF)
    - Hybrid search (combination of semantic + lexical)
    """

    _vector_store: VectorStore = PrivateAttr()
    _lexical_retriever: Optional[Any] = PrivateAttr(default=None)
    search_type: SearchType = Field(default="similarity", description="Search strategy to use")
    
    search_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search parameters (k, fetch_k, lambda_mult, score_threshold, etc.)"
    )
    
    hybrid_weights: dict[str, float] = Field(
        default_factory=lambda: {"semantic": 0.7, "lexical": 0.3},
        description="Weights for hybrid search (must sum to 1.0)"
    )

    def __init__(
        self,
        vector_store: VectorStore,
        search_type: SearchType = "similarity",
        corpus_documents: Optional[list[Document]] = None,
        **search_kwargs,
    ):
        """
        Initialize retriever with specified search strategy.
        
        Args:
            vector_store: Vector store for semantic search
            search_type: Type of search strategy
            corpus_documents: Required for BM25/TF-IDF/hybrid search
            **search_kwargs: Additional parameters for search
        """
        super().__init__()
        self._vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        
        # Initialize lexical retriever if needed
        if search_type in ("bm25", "tfidf", "hybrid"):
            if not corpus_documents:
                logger.error(
                    f"corpus_documents required for {search_type} search")
                raise ValueError(
                    f"corpus_documents required for {search_type} search"
                )
            self._lexical_retriever = self._initialize_lexical_retriever(
                search_type, corpus_documents
            )
        
        # Validate hybrid weights
        if search_type == "hybrid":
            total_weight = sum(self.hybrid_weights.values())
            if not abs(total_weight - 1.0) < 1e-6:
                raise ValueError(
                    f"hybrid_weights must sum to 1.0, got {total_weight}"
                )

    def _initialize_lexical_retriever(
        self, search_type: str, documents: list[Document]
    ) -> Any:
        """Initialize BM25 or TF-IDF retriever"""
        if search_type in ("bm25", "hybrid"):
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                raise ImportError(
                    "rank-bm25 required for BM25 search. "
                    "Install with: pip install rank-bm25"
                )
             
            # Tokenize documents for BM25
            tokenized_docs = [
                doc.page_content.lower().split() for doc in documents
            ]
            return {
                "retriever": BM25Okapi(tokenized_docs),
                "documents": documents,
                "type": "bm25"
            }
        
        elif search_type == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
            except ImportError:
                raise ImportError(
                    "scikit-learn required for TF-IDF search. "
                    "Install with: pip install scikit-learn"
                )
            
            # Create TF-IDF vectorizer
            corpus = [doc.page_content for doc in documents]
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=self.search_kwargs.get("max_features", 5000)
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            return {
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "documents": documents,
                "type": "tfidf"
            }
        
        return None

    def _retrieve_semantic(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve using vector similarity"""
        search_kwargs = {"k": k, } 
        
        if self.search_type == "mmr":
            return self._vector_store.max_marginal_relevance_search(
                query, **search_kwargs
            )
        else:  # similarity
            return self._vector_store.similarity_search(query, **search_kwargs)

    def _retrieve_lexical(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve using BM25 or TF-IDF"""
        if not self._lexical_retriever:
            raise ValueError("Lexical retriever not initialized")
        
        retriever_type = self._lexical_retriever["type"]
        documents = self._lexical_retriever["documents"]
        
        if retriever_type == "bm25":
            # Tokenize query and get BM25 scores
            tokenized_query = query.lower().split()
            bm25 = self._lexical_retriever["retriever"]
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k documents
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:k]
            return [documents[i] for i in top_indices]
        
        elif retriever_type == "tfidf":
            # Transform query and compute similarity
            vectorizer = self._lexical_retriever["vectorizer"]
            tfidf_matrix = self._lexical_retriever["tfidf_matrix"]
            
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Get top-k documents
            top_indices = scores.argsort()[-k:][::-1]
            return [documents[i] for i in top_indices]
        
        return []

    def _retrieve_hybrid(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve using hybrid semantic + lexical search"""
        # Get results from both strategies
        semantic_docs = self._retrieve_semantic(query, k=k * 2)
        lexical_docs = self._retrieve_lexical(query, k=k * 2)

        
        # Combine and re-rank using weighted scores
        doc_scores: dict[str, float] = {}
        
        # Add semantic scores
        sem_weight = self.hybrid_weights.get("semantic", 0.7)
        for idx, doc in enumerate(semantic_docs):
            doc_id = doc.page_content
            # Normalize rank-based score
            score = (len(semantic_docs) - idx) / len(semantic_docs)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score * sem_weight
        
        # Add lexical scores
        lex_weight = self.hybrid_weights.get("lexical", 0.3)
        for idx, doc in enumerate(lexical_docs):
            doc_id = doc.page_content
            score = (len(lexical_docs) - idx) / len(lexical_docs)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score * lex_weight
        
        # Create document map
        doc_map = {doc.page_content: doc for doc in semantic_docs + lexical_docs}
        
        # Sort by combined score and return top-k
        sorted_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:k]]

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve documents for query using configured search strategy.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant documents
        """
        k = self.search_kwargs.get("k", 4)
        
        if self.search_type in ("similarity", "mmr"):
            docs =  self._retrieve_semantic(query, k)
            return docs
        elif self.search_type in ("bm25", "tfidf"):
            return self._retrieve_lexical(query, k)
        elif self.search_type == "hybrid":
            return self._retrieve_hybrid(query, k)
        
        raise ValueError(f"Unknown search type: {self.search_type}")
 
    @override
    def _get_relevant_documents(
        self,
        query: str,
        **kwargs
    ) -> list[Document]:
        """LangChain retriever interface implementation"""
        return self.retrieve(query)
