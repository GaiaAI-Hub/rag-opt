

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.schema import Document
from typing import Literal

class Retriever:
    """Enhanced retriever with multiple search strategies (Phase 5)"""

    # TODO:: allow different search strategies (tfidf, BM25, opensearch, ..)
    
    def __init__(self, 
                 vector_store: VectorStore, 
                 search_type: Literal["mmr", "similarity","similarity_score_threshold"] = "similarity",
                 **search_kwargs):
        self.vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.retriever = self._create_retriever()
    
    def _create_retriever(self) -> BaseRetriever:
        """Create retriever with specified search configuration"""
        return self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs
        )
    def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents for query"""
        return self.retriever.invoke(query)
