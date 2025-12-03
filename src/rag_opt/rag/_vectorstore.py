from typing import Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.docstore.in_memory import InMemoryDocstore
from loguru import logger
import faiss
import os

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


def _get_embedding_dimension(embeddings: Embeddings) -> int:
    """
    Cached function to get embedding dimension.
    Only runs once per embeddings instance.
    """
    try:
        return len(embeddings.embed_query("test"))
    except Exception as e:
        logger.warning(f"Could not determine embedding dimension: {e}. Using default 768.")
        return 768


def _create_pinecone_index(index_name: str, api_key: str, dimension: int, initialize: bool = False) -> None:
    """Create Pinecone index if it doesn't exist"""
    from pinecone import Pinecone, ServerlessSpec
    
    if not api_key:
        raise ValueError("api_key is required for Pinecone")
    
    pc = Pinecone(api_key=api_key)
    existing_indexes = pc.list_indexes().names()
    
    # Index already exists
    if index_name in existing_indexes:
        return
    
    # Check index limit
    if len(existing_indexes) >= 5:
        if initialize:
            logger.warning("Pinecone free tier limit (5 indexes) reached. Index will be created at runtime.")
            return
        # Delete oldest index to make room
        pc.delete_index(existing_indexes[-2])
        logger.info(f"Deleted old Pinecone index to make room for {index_name}")
    
    # Create new index
    logger.debug(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def init_vectorstore(
    provider: str,
    embeddings: Embeddings,
    documents: Optional[list[Document]] = None,
    api_key: Optional[str] = "",
    initialize: bool = False,
    **kwargs
) -> VectorStore:
    """
    Initialize vector store with optimized setup.
    
    Args:
        provider: Vector store provider (faiss, chroma, pinecone, qdrant)
        embeddings: Embeddings instance
        documents: Optional documents to initialize with
        api_key: API key for cloud providers
        initialize: If True, skip errors during registry loading
        **kwargs: Provider-specific parameters
    
    Returns:
        Initialized vector store instance
    """
    provider = provider.lower()
    
    # Get embedding dimension (cached)
    embedding_dim = _get_embedding_dimension(embeddings)
    
    # Generate index/collection name
    base_name = kwargs.get('index_name') or kwargs.get('collection_name', 'ragopt')
    index_name = f"{base_name}-{embedding_dim}"
    
    # Initialize vector store based on provider
    if provider == "faiss":
        vector_store = _init_faiss(embeddings, documents, embedding_dim)
    
    elif provider == "chroma":
        vector_store = _init_chroma(embeddings, documents, index_name, **kwargs)
    
    elif provider == "pinecone":
        vector_store = _init_pinecone(embeddings, documents, index_name, api_key, embedding_dim, initialize)
    
    elif provider == "qdrant":
        vector_store = _init_qdrant(embeddings, documents, index_name, embedding_dim, **kwargs)
    
    else:
        raise ValueError(
            f"Unsupported vector store: {provider}. "
            f"Supported: faiss, chroma, pinecone, qdrant"
        )
    
    logger.success(f"{provider.upper()} loaded successfully → {index_name}")
    return vector_store


def _init_faiss(embeddings: Embeddings, documents: Optional[list[Document]], 
                embedding_dim: int) -> FAISS:
    """Initialize FAISS vector store"""
    if documents:
        return FAISS.from_documents(documents, embeddings)
    
    # Create empty index
    index = faiss.IndexFlatL2(embedding_dim)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )


def _init_chroma(embeddings: Embeddings, documents: Optional[list[Document]],
                 collection_name: str, **kwargs) -> Chroma:
    """Initialize Chroma vector store"""
    persist_directory = kwargs.get("persist_directory", "./chroma_db")
    
    if documents:
        return Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    
    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )


def _init_pinecone(embeddings: Embeddings, documents: Optional[list[Document]],
                   index_name: str, api_key: str, embedding_dim: int, 
                   initialize: bool) -> VectorStore:
    """Initialize Pinecone vector store"""
    from langchain_pinecone import Pinecone
    
    if api_key:
        os.environ["PINECONE_API_KEY"] = api_key
    
    # Create index if needed
    _create_pinecone_index(index_name, api_key, embedding_dim, initialize)
    
    if documents:
        return Pinecone.from_documents(documents, embeddings, index_name=index_name)
    
    return Pinecone.from_existing_index(index_name, embeddings)


def _init_qdrant(embeddings: Embeddings, documents: Optional[list[Document]],
                 collection_name: str, embedding_dim: int, **kwargs) -> QdrantVectorStore:
    """Initialize Qdrant vector store"""
    url = kwargs.get("url", "http://localhost:6333")
    
    # Initialize client
    if url == ":memory:" or kwargs.get("in_memory", False):
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(url=url)
    
    # Create collection if needed
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    except Exception:
        # Collection likely exists
        pass
    
    if documents:
        return QdrantVectorStore.from_documents(
            documents,
            embeddings,
            client=client,
            collection_name=collection_name
        )
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )