from langchain.schema import Document



class Reranker:
    """Base reranker class for different reranker types (Phase 4)"""
    # TODO:: recheck langchain docs and install those
    def __init__(self, reranker_type: str = "cross_encoder", **kwargs):
        self.reranker_type = reranker_type
        self.kwargs = kwargs
        self.reranker = self._init_reranker()
    
    def _init_reranker(self):
        """Initialize specific reranker based on type"""
        if self.reranker_type == "cross_encoder":
            from langchain_huggingface import HuggingFaceEmbeddings # TODO;: allow custom models from HF 
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            model_name = self.kwargs.get("model_name", "BAAI/bge-reranker-base" ) # "cross-encoder/ms-marco-MiniLM-L-6-v2")
            # return HuggingFaceEmbeddings(model_name=model_name) # TODO:: recheck this again
            return  HuggingFaceCrossEncoder(model_name=model_name)

        
        elif self.reranker_type == "cohere":
            import cohere
            api_key = self.kwargs.get("api_key")
            return cohere.Client(api_key)
        
        elif self.reranker_type == "flashrank":
            from flashrank import Ranker
            return Ranker()
        
        elif self.reranker_type == "bm25":
            from rank_bm25 import BM25Okapi
            return None  # Will be initialized with documents
        
        # TODO:: add TFIDFRetriever
        else:
            raise ValueError(f"Unsupported reranker type: {self.reranker_type}")
    
    def rerank(self, query: str, documents: list[Document], top_k: int = 10) -> list[Document]:
        """Rerank documents based on query relevance"""
        if self.reranker_type == "cross_encoder":
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.reranker.rerank(pairs)
            
            # Sort by scores and return top_k
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_k]]
        
        elif self.reranker_type == "cohere":
            docs_text = [doc.page_content for doc in documents]
            response = self.reranker.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=docs_text,
                top_n=top_k
            )
            
            reranked_docs = []
            for result in response.results:
                reranked_docs.append(documents[result.index])
            return reranked_docs
        
        else:
            # Fallback: return original order
            return documents[:top_k]
    

# TODO:: revise all of these rerankers

# https://python.langchain.com/docs/integrations/document_transformers/jina_rerank/
# https://python.langchain.com/docs/integrations/retrievers/cohere-reranker/
# https://python.langchain.com/docs/integrations/retrievers/contextual/
# https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/
# https://python.langchain.com/docs/integrations/retrievers/pinecone_rerank/
# https://python.langchain.com/docs/integrations/document_transformers/infinity_rerank/
# https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/
# https://python.langchain.com/docs/integrations/document_transformers/rankllm-reranker/
# https://python.langchain.com/docs/integrations/document_transformers/voyageai-reranker/
# MonoT5 Reranker
# TART Reranker
# UPR Reranker
# BM25 Reranker
# RankGPT
# Colbert Reranker
# Flag Embedding Reranker
# Time Reranker
# voyageai_reranker
# FlashRank Reranker