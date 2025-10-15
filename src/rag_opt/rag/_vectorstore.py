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

def init_vectorstore(
    provider: str,
    embeddings: Embeddings,
    documents: Optional[list[Document]] = None,
    api_key: Optional[str] = "",
    initialize: bool = False,
    **kwargs
) -> VectorStore:
    """
    Initialize vector store similar to init_embeddings and init_chat_model
    
    Args:
        provider: Vector store provider name
        embeddings: Embeddings instance
        documents: Optional documents to initialize with
        initialize: if True means this is used to load the vectorstore for registery (could skip intializing in case of error)
        **kwargs: Provider-specific parameters
    """
    
    try:
        embedding_dim = len(embeddings.embed_query("test query"))
    except Exception as e:
        logger.warning(f"Could not determine actual embedding dimension: {e}. Defaulting to 768.")
        embedding_dim = 768

    provider = provider.lower()
    index_name = f"{kwargs.get('index_name', 'ragopt-index-') or kwargs.get('collection_name', 'ragopt-collection-')}{embedding_dim}"
    
    out = None
    if provider == "faiss":
        if documents:
            out =  FAISS.from_documents(documents, embeddings)
        else:
            # Create empty FAISS index with correct dimension
            index = faiss.IndexFlatL2(embedding_dim)
            out = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            

    elif provider == "chroma":
        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        collection_name = index_name

        if documents:

            out = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        else:
            out = Chroma(
                embedding_function=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )


    elif provider == "pinecone":
        from langchain_pinecone import Pinecone

        if api_key:
            os.environ["PINECONE_API_KEY"] = api_key

        _create_pinecone_index_if_not_exists(index_name=index_name, api_key=api_key, dimension=embedding_dim,initialize=initialize)

        if documents:
            out = Pinecone.from_documents(documents, embeddings, index_name=index_name)
        else:
            out =  Pinecone.from_existing_index(index_name, embeddings)


    elif provider == "qdrant":
        url = kwargs.get("url", "http://localhost:6333")
        collection_name = index_name

        # Initialize client
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
            out = QdrantVectorStore.from_documents(
                documents,
                embeddings,
                client=client,
                collection_name=collection_name
            )
        else:
            out = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings
            )

    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")
    
    _log_loading(provider, index_name)
    return out



def _log_loading(vector_store_name: str, index_name: str) -> None:
    logger.success(f"{vector_store_name} Loaded successfully, working with index: {index_name}")

def _create_pinecone_index_if_not_exists(index_name:str, api_key:str, dimension:int, initialize:bool) -> None:
    """ will try to create the index if it does not exist (only in case u didnt exceed the limit) other than that will skip and load later"""
    from pinecone import Pinecone, ServerlessSpec
    if not api_key:
        raise ValueError("api_key is required for Pinecone")
    pc = Pinecone(api_key=api_key)

    if not index_name in pc.list_indexes().names():
        if len(pc.list_indexes().names()) >= 5:
            if initialize:
                logger.warning("Pinecone has reached the limit of 5 indexes. Skipping index creation and will create it during optimization runtime.")
                return
            else:
                pc.delete_index(pc.list_indexes().names().pop(-2)) 
        logger.debug(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
