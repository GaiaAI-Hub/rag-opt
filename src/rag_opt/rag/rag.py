from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import VectorStore
from langchain.schema import Document
from rag_opt.rag.retriever import Retriever
from rag_opt.rag.reranker import Reranker
from typing import Optional


# TODO:: recheck langchain docs for this 
# TODO:: add more specific methods within the pipeline like rerank , get relevant docs, ...

class GAIARAG:
    """Main RAG pipeline class"""
    
    def __init__(self, 
                 embeddings, 
                 vector_store: VectorStore, 
                 llm,
                 reranker: Optional[Reranker] = None,
                 retriever_config: Optional[dict] = None):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        
        # Initialize retriever
        retriever_config = retriever_config or {"search_type": "similarity", "k": 4}
        self.retriever = Retriever(vector_store, **retriever_config)
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self): # TODO:: add typing 
        """Create the complete RAG chain"""
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs: Document):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": self.retriever.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def get_answer(self, query: str, use_reranker: bool = True) -> str:
        """Process query through RAG pipeline"""
        documents = self.retriever.retrieve(query)
        
        # Apply reranking if available
        if use_reranker and self.reranker:
            documents = self.reranker.rerank(query, documents)
        
        # Generate answer using RAG chain
        response = self.rag_chain.invoke(query)
        return response
    
    def get_agentic_answer(self, query: str, use_reranker: bool = True) -> str:
        """Process query through Agent """
        # TODO:: use ReAct, langchain agent , langgraph
    

# TODO:: allow customization design of RAG pipeline and see how to make compatible with the BO optimizer