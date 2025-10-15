from rag_opt.dataset import EvaluationDatasetItem, GroundTruth, ComponentUsage, TrainDataset, EvaluationDataset
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import VectorStore
from typing_extensions import Annotated, Doc, Optional
from concurrent.futures import Future, Executor, as_completed
from rag_opt._utils import get_shared_executor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from rag_opt.rag.callbacks import RAGCallbackHandler
from rag_opt.rag.retriever import Retriever
from rag_opt.rag.reranker import BaseReranker
from langchain.schema import Document
from rag_opt.llm import RAGLLM
import time


class RAGWorkflow:
    """Main RAG pipeline class"""
    
    agent_executor: Annotated[AgentExecutor, Doc("The agent executor for handling RAG process using agent")] = None
    
    def __init__(self, 
                 embeddings, 
                 vector_store: VectorStore, 
                 llm: Annotated[RAGLLM, Doc("the llm to be used in the dataset evaluation process")],
                 reranker: Optional[BaseReranker] = None,
                 retrieval_config: Optional[dict] = None,
                 *,
                 embedding_provider_name: Annotated[str, Doc("Name of the embedding provider like openai,...")] = None,
                 embedding_model_name: Annotated[str, Doc("Name of the embedding model like text-embedding-ada-002,...")] = None,
                 llm_provider_name: Annotated[str, Doc("Name of the llm provider like openai,...")] = None,
                 llm_model_name: Annotated[str, Doc("Name of the llm model like gpt-4,...")] = None,
                 reranker_provider_name: Annotated[str, Doc("Name of the reranker provider like cohere,...")] = None,
                 reranker_model_name: Annotated[str, Doc("Name of the reranker model like rerank-english-v2.0,...")] = None,
                 vector_store_provider_name: Annotated[str, Doc("Name of the vector store provider like pinecone,...")] = None,
                 max_workers: Annotated[int, Doc("Maximum workers for parallel component loading")] = 5,
                 executor: Annotated[Optional[Executor], Doc("The thread pool executor for batch evaluation")] = None,
                 **kwargs):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

        # Provider and model names
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model_name = embedding_model_name
        self.llm_provider_name = llm_provider_name
        self.llm_model_name = llm_model_name
        self.reranker_provider_name = reranker_provider_name
        self.reranker_model_name = reranker_model_name
        self.vector_store_provider_name = vector_store_provider_name

        # Initialize retrieval
        retrieval_config = retrieval_config or {
            "search_type": kwargs.get("search_type", "similarity"), 
            "k": kwargs.get("k", 5)
        }
        
        self.retrieval = Retriever(
            vector_store, 
            corpus_documents=kwargs.get("corpus_documents", None),
            **retrieval_config
        )
        
        self.retrieval_tool = create_retriever_tool(
            self.retrieval,
            "retrieve_relative_context",
            "Search and return information required to answer the question",
        )

        self.executor = executor if executor is not None else get_shared_executor(max_workers)
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
        
        # Initialize agent components
        self._init_agent()

    def _init_agent(self):
        """Initialize agent and agent executor"""
        tools = [self.retrieval_tool]
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the retrieve_relative_context tool to get relevant information to answer questions."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        self.agent = create_tool_calling_agent(self.llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=False,
        )
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create RAG prompt template"""
        template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
        return ChatPromptTemplate.from_template(template)
    
    def _create_rag_chain(self) -> RunnableSequence:
        """Create the complete RAG chain with optional reranking"""
        prompt = self._create_rag_prompt()
        
        def format_docs(docs: list[Document]) -> str:
            """Format documents into a single string"""
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": self.retrieval | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def _generate_eval_metadata(self, callback_handler: RAGCallbackHandler) -> dict[str, ComponentUsage]:
        """Generate evaluation metadata for metrics evaluation"""
        return {
            "cost": ComponentUsage(
                llm=callback_handler.llm_stats.total_cost,
                embedding=callback_handler.embedding_stats.total_cost,
                vectorstore=0.0,
                reranker=callback_handler.reranker_stats.total_cost
            ),
            "latency": ComponentUsage(
                llm=callback_handler.llm_stats.total_latency,
                embedding=callback_handler.embedding_stats.total_latency,
                vectorstore=0.0,
                reranker=callback_handler.reranker_stats.total_latency
            )
        }

    def get_batch_answers(self, 
                          dataset: TrainDataset,
                          **kwargs) -> EvaluationDataset:
        """Get answers for all dataset questions - useful for preparing evaluation dataset"""
        futures: list[Future[EvaluationDatasetItem]] = []

        for item in dataset.items:
            futures.append(self.executor.submit(
                self.get_answer, 
                query=item.question, 
                ground_truth=item.to_ground_truth(), 
                **kwargs
            ))
        
        items: list[EvaluationDatasetItem] = []
        for future in as_completed(futures):
            items.append(future.result())
        
        return EvaluationDataset(items=items)

    def get_answer(
        self, 
        query: str,
        *,
        ground_truth: Optional[GroundTruth] = None,
        use_reranker: bool = False,
        rerank_top_k: int = 10,
        **kwargs
    ) -> EvaluationDatasetItem:
        """
        Process query through RAG pipeline with optional reranking.
        
        Args:
            query: Question to answer
            ground_truth: Optional ground truth for evaluation
            use_reranker: Whether to apply reranking
            rerank_top_k: Number of documents to keep after reranking
        """
        callback_handler = RAGCallbackHandler(
            verbose=kwargs.get("verbose", False),
            llm_provider_name=self.llm_provider_name,
            llm_model_name=self.llm_model_name,
            embedding_provider_name=self.embedding_provider_name,
            embedding_model_name=self.embedding_model_name,
            reranker_model_name=self.reranker_model_name,
            reranker_provider_name=self.reranker_provider_name,
            vector_store_provider_name=self.vector_store_provider_name
        )
        
        # Apply reranking if requested
        if use_reranker and self.reranker is not None:
            # Retrieve documents
            docs = self.retrieval.invoke(query)
            
            # Rerank documents
            start_time = time.time()
            reranked_docs = self.reranker.rerank(
                query=query, 
                documents=docs, 
                top_k=rerank_top_k
            )
            rerank_latency = time.time() - start_time
            
            # Track reranking cost
            callback_handler.track_reranking(docs, rerank_latency)
            
            # Create temporary chain with reranked docs
            def format_docs(docs: list[Document]) -> str:
                return "\n\n".join(doc.page_content for doc in docs)
            
            prompt = self._create_rag_prompt()
            temp_chain = (
                {"context": lambda _: format_docs(reranked_docs), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            response = temp_chain.invoke(query, config={"callbacks": [callback_handler]})
            contexts = [doc.page_content for doc in reranked_docs]
        else:
            # Standard RAG without reranking
            response = self.rag_chain.invoke(
                query, 
                config={"callbacks": [callback_handler]}
            )
            contexts = callback_handler.retrieved_contexts
        
        metadata = self._generate_eval_metadata(callback_handler)
        
        return EvaluationDatasetItem(
            question=query,
            answer=response,
            contexts=contexts,
            ground_truth=ground_truth,
            metadata=metadata
        )
    
    def get_agentic_answer(self, query: str, **kwargs) -> EvaluationDatasetItem:
        """Process query through Agentic RAG pipeline"""
        callback_handler = RAGCallbackHandler(
            verbose=kwargs.get("verbose", False),
            llm_provider_name=self.llm_provider_name,
            llm_model_name=self.llm_model_name,
            embedding_provider_name=self.embedding_provider_name,
            embedding_model_name=self.embedding_model_name,
            reranker_model_name=self.reranker_model_name,
            reranker_provider_name=self.reranker_provider_name,
            vector_store_provider_name=self.vector_store_provider_name
        )
        
        response = self.agent_executor.invoke(
            {"input": query}, 
            config={"callbacks": [callback_handler]}
        )
        
        contexts = callback_handler.retrieved_contexts
        metadata = self._generate_eval_metadata(callback_handler)
        
        return EvaluationDatasetItem(
            question=query,
            answer=response.get("output"),
            contexts=contexts,
            ground_truth=None,
            metadata=metadata
        )
    
    def get_relevant_docs(self, query: str) -> list[Document]:
        """Retrieve relevant documents for query"""
        return self.retrieval.invoke(query)

    def store_documents(self, documents: list[Document]):
        """Store documents in vector store"""
        self.vector_store.add_documents(documents)