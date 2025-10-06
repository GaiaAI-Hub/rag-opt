from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from loguru import logger 


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
    



def index_documents(documents: list[Document], 
                    chunk_size:int, 
                    chunk_overlap:int,
                    vector_store:VectorStore,
                    ):
    """ Index documents in vector store """
    indexer = Indexer(chunk_size, chunk_overlap, vector_store)
    indexer.store(documents)
    logger.success(f"Successfully indexed {len(documents)} documents in vector store")



def index_data_files(dataset_path: str="./data", 
                    **kwargs,):
    from rag_opt.rag.parser import Parser
    parser = Parser(path=dataset_path, **kwargs)
    docs = parser.load_docs() 
    index_documents(docs, **kwargs)