from pydantic.v1 import BaseModel
from typing import Optional, List
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import Pinecone
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from loguru import logger 
import faiss

def init_vectorstore(
    provider: str,
    embeddings: Embeddings,
    embedding_dim: int,
    documents: Optional[List[Document]] = None,
    **kwargs
) -> VectorStore:
    """
    Initialize vector store similar to init_embeddings and init_chat_model
    
    Args:
        provider: Vector store provider name
        embeddings: Embeddings instance
        documents: Optional documents to initialize with
        **kwargs: Provider-specific parameters
    """
    
    provider = provider.lower()
    
    if provider == "faiss":
        if documents:
            return FAISS.from_documents(documents, embeddings)
        else:
            # Create empty FAISS index
            index = faiss.IndexFlatL2(embedding_dim)
            return FAISS(
                embedding_function=embeddings,
                index=index,
                docstore={},
                index_to_docstore_id={}
            )
    
    elif provider == "chroma":
        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        collection_name = kwargs.get("collection_name", "default")
        
        if documents:
            return Chroma.from_documents(
                documents, 
                embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        else:
            return Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
    
    elif provider == "pinecone":
        # TODO:: see how to get api key , create index if not exist 
        index_name = kwargs.get("index_name", "default-index")
        if documents:
            return Pinecone.from_documents(documents, embeddings, index_name=index_name)
        else:
            return Pinecone.from_existing_index(index_name, embeddings)
    
    elif provider == "qdrant":
        url = kwargs.get("url", "http://localhost:6333")
        collection_name = kwargs.get("collection_name", "default")
        
        sample_embedding = embeddings.embed_query("test")
        
        # Initialize client - use provided URL or in-memory
        if url == ":memory:" or kwargs.get("in_memory", False):
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(url=url)
        
        # Create collection if it doesn't exist
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
        except Exception:
            # Collection might already exist
            pass

        if documents:
            return QdrantVectorStore.from_documents(
                documents, 
                embeddings, 
                client=client,
                collection_name=collection_name
            )
        else:
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings
            )
    
    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")
    



class Indexer:
    """ Take embeddings and store in vector db / index (Phase 3) """

    def __init__(self, 
                 chunk_size:int,
                 chunk_overlap:int,
                 vector_store:VectorStore,
                 ) -> None:
        """ Load the parsed documents and split them

            Args: (those args represents hyperparameters in our RAG pipeline)
                - chunk_size: Size of each chunk.
                - chunk_overlap: Overlap between chunks.
                - vector_store: Vector store to use.
                    https://python.langchain.com/docs/integrations/vectorstores/

        """
        self.vector_store = vector_store
        self.embeddings = vector_store.embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def store(self, documents: list[Document]):
        """Store documents in vector store"""
        if not documents:
            raise ValueError("No documents provided to store")
        
        self.vector_store.add_documents(documents)
        logger.success(f"Successfully stored {len(documents)} documents in vector store")