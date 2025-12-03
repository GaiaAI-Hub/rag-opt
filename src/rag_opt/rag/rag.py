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
from loguru import logger
from enum import Enum
import time


class RAGMode(Enum):
    """Enumeration of supported RAG execution modes."""
    BASELINE = "baseline"
    AGENTIC = "agentic"
    HYBRID = "hybrid"
    RERANKING = "reranking"


class BaseRAGInterface:
    """
    Base interface for custom RAG implementations.
    Users can extend this class to create custom RAG workflows.
    """
    
    def get_answer(self, query: str, **kwargs) -> EvaluationDatasetItem:
        """Process a single query and return evaluation item."""
        raise NotImplementedError("Subclasses must implement get_answer()")
    
    def get_batch_answers(self, dataset: TrainDataset, **kwargs) -> EvaluationDataset:
        """Process multiple queries in batch."""
        raise NotImplementedError("Subclasses must implement get_batch_answers()")
    
    def get_relevant_docs(self, query: str, **kwargs) -> list[Document]:
        """Retrieve relevant documents for a query."""
        raise NotImplementedError("Subclasses must implement get_relevant_docs()")


class RAGWorkflow(BaseRAGInterface):
    """
    Production-ready RAG pipeline with multiple execution modes.
    
    Supports:
    - Baseline: Standard retrieval + generation
    - Agentic: Tool-calling agents with decision-making
    - Hybrid: Semantic + lexical search combination
    - Reranking: Document reranking for improved relevance
    """
    
    def __init__(
        self, 
        embeddings, 
        vector_store: VectorStore, 
        llm: Annotated[RAGLLM, Doc("LLM for answer generation")],
        reranker: Optional[BaseReranker] = None,
        retrieval_config: Optional[dict] = None,
        *,
        corpus_documents: Annotated[Optional[list[Document]], Doc("Corpus documents for lexical search")] = None,
        lexical_cache_dir: Annotated[Optional[str], Doc("Path to lexical retriever cache")] = None,
        embedding_provider_name: Annotated[Optional[str], Doc("Embedding provider name")] = None,
        embedding_model_name: Annotated[Optional[str], Doc("Embedding model name")] = None,
        llm_provider_name: Annotated[Optional[str], Doc("LLM provider name")] = None,
        llm_model_name: Annotated[Optional[str], Doc("LLM model name")] = None,
        reranker_provider_name: Annotated[Optional[str], Doc("Reranker provider name")] = None,
        reranker_model_name: Annotated[Optional[str], Doc("Reranker model name")] = None,
        vector_store_provider_name: Annotated[Optional[str], Doc("Vector store provider name")] = None,
        max_workers: Annotated[int, Doc("Maximum workers for parallel processing")] = 5,
        executor: Annotated[Optional[Executor], Doc("Thread pool executor")] = None,
        **kwargs
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

        # Provider and model metadata for tracking
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model_name = embedding_model_name
        self.llm_provider_name = llm_provider_name
        self.llm_model_name = llm_model_name
        self.reranker_provider_name = reranker_provider_name
        self.reranker_model_name = reranker_model_name
        self.vector_store_provider_name = vector_store_provider_name

        # Initialize retrieval configuration
        retrieval_config = retrieval_config or {
            "search_type": kwargs.get("search_type", "similarity"), 
            "k": kwargs.get("k", 5)
        }
        
        # Initialize retriever with caching support for hybrid mode
        self.retrieval = Retriever(
            vector_store, 
            corpus_documents=corpus_documents,
            lexical_cache_dir=lexical_cache_dir,
            **retrieval_config
        )
        
        # Executor for parallel batch processing
        self.executor = executor or get_shared_executor(max_workers)
        
        # Create baseline RAG chain
        self.rag_chain = self._create_rag_chain()
        
        # Initialize agentic components
        self.agent_executor: Optional[AgentExecutor] = None
        self._init_agent()

    def _init_agent(self) -> None:
        """Initialize agent executor for agentic mode."""
        retrieval_tool = create_retriever_tool(
            self.retrieval,
            "retrieve_relevant_context",
            "Search and return information required to answer the question accurately",
        )
        
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the retrieve_relevant_context tool to get relevant information to answer questions accurately."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, [retrieval_tool], agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=[retrieval_tool],
            verbose=False,
        )

    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create RAG prompt template for baseline mode."""
        template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
        return ChatPromptTemplate.from_template(template)
    
    def _format_docs(self, docs: list[Document]) -> str:
        """Format documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _create_rag_chain(self) -> RunnableSequence:
        """Create the baseline RAG chain."""
        prompt = self._create_rag_prompt()
        
        rag_chain = (
            {"context": self.retrieval | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def _generate_eval_metadata(self, callback_handler: RAGCallbackHandler) -> dict[str, ComponentUsage]:
        """Generate evaluation metadata for metrics tracking."""
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

    def _apply_reranking(
        self,
        query: str,
        docs: list[Document],
        rerank_top_k: int,
        callback_handler: RAGCallbackHandler
    ) -> list[Document]:
        """Apply reranking to retrieved documents."""
        if not self.reranker:
            logger.warning("Reranker not initialized, returning original documents")
            return docs
        
        start_time = time.time()
        reranked_docs = self.reranker.rerank(
            query=query, 
            documents=docs, 
            top_k=rerank_top_k
        )
        rerank_latency = time.time() - start_time
        
        callback_handler.track_reranking(docs, rerank_latency)
        
        return reranked_docs

    def _get_callback_handler(self, verbose: bool = False) -> RAGCallbackHandler:
        """Create a callback handler with current configuration."""
        return RAGCallbackHandler(
            verbose=verbose,
            llm_provider_name=self.llm_provider_name,
            llm_model_name=self.llm_model_name,
            embedding_provider_name=self.embedding_provider_name,
            embedding_model_name=self.embedding_model_name,
            reranker_model_name=self.reranker_model_name,
            reranker_provider_name=self.reranker_provider_name,
            vector_store_provider_name=self.vector_store_provider_name
        )

    def get_answer(
        self, 
        query: str,
        *,
        ground_truth: Optional[GroundTruth] = None,
        mode: RAGMode = RAGMode.BASELINE,
        use_reranker: bool = False,
        rerank_top_k: int = 10,
        verbose: bool = False,
        **kwargs
    ) -> EvaluationDatasetItem:
        """
        Process query through RAG pipeline with configurable execution mode.
        
        Args:
            query: Question to answer
            ground_truth: Optional ground truth for evaluation
            mode: Execution mode (BASELINE, AGENTIC, HYBRID, RERANKING)
            use_reranker: Whether to apply reranking (overrides mode if True)
            rerank_top_k: Number of documents to keep after reranking
            verbose: Enable verbose logging
            
        Returns:
            EvaluationDatasetItem with answer and metadata
        """
        # Determine effective mode
        if use_reranker:
            mode = RAGMode.RERANKING
        
        callback_handler = self._get_callback_handler(verbose)
        
        # Route to appropriate mode handler
        if mode == RAGMode.AGENTIC:
            return self._get_agentic_answer(query, ground_truth, callback_handler)
        elif mode in (RAGMode.RERANKING, RAGMode.HYBRID):
            return self._get_reranked_answer(query, ground_truth, rerank_top_k, callback_handler)
        else:  # BASELINE
            return self._get_baseline_answer(query, ground_truth, callback_handler)

    def _get_baseline_answer(
        self,
        query: str,
        ground_truth: Optional[GroundTruth],
        callback_handler: RAGCallbackHandler
    ) -> EvaluationDatasetItem:
        """Execute baseline RAG without reranking."""
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

    def _get_reranked_answer(
        self,
        query: str,
        ground_truth: Optional[GroundTruth],
        rerank_top_k: int,
        callback_handler: RAGCallbackHandler
    ) -> EvaluationDatasetItem:
        """Execute RAG with document reranking."""
        # Retrieve documents
        docs = self.retrieval.invoke(query)
        
        # Apply reranking
        reranked_docs = self._apply_reranking(query, docs, rerank_top_k, callback_handler)
        
        # Create temporary chain with reranked documents
        prompt = self._create_rag_prompt()
        temp_chain = (
            {"context": lambda _: self._format_docs(reranked_docs), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = temp_chain.invoke(query, config={"callbacks": [callback_handler]})
        contexts = [doc.page_content for doc in reranked_docs]
        metadata = self._generate_eval_metadata(callback_handler)
        
        return EvaluationDatasetItem(
            question=query,
            answer=response,
            contexts=contexts,
            ground_truth=ground_truth,
            metadata=metadata
        )

    def _get_agentic_answer(
        self,
        query: str,
        ground_truth: Optional[GroundTruth],
        callback_handler: RAGCallbackHandler
    ) -> EvaluationDatasetItem:
        """Execute agentic RAG with tool-calling agent."""
        if not self.agent_executor:
            raise RuntimeError("Agent executor not initialized")
        
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
            ground_truth=ground_truth,
            metadata=metadata
        )

    def get_batch_answers(
        self, 
        dataset: TrainDataset,
        mode: RAGMode = RAGMode.BASELINE,
        use_reranker: bool = False,
        rerank_top_k: int = 10,
        verbose: bool = False,
        **kwargs
    ) -> EvaluationDataset:
        """
        Get answers for all dataset questions in parallel.
        
        Args:
            dataset: Training dataset with questions
            mode: Execution mode for all queries
            use_reranker: Whether to apply reranking
            rerank_top_k: Number of documents to keep after reranking
            verbose: Enable verbose logging
            
        Returns:
            EvaluationDataset with all answers
        """
        futures: list[Future[EvaluationDatasetItem]] = []

        # Submit all tasks in parallel
        for item in dataset.items:
            future = self.executor.submit(
                self.get_answer, 
                query=item.question, 
                ground_truth=item.to_ground_truth(),
                mode=mode,
                use_reranker=use_reranker,
                rerank_top_k=rerank_top_k,
                verbose=verbose,
                **kwargs
            )
            futures.append(future)
        
        # Collect results as they complete
        items: list[EvaluationDatasetItem] = []
        for future in as_completed(futures):
            try:
                items.append(future.result(timeout=120)) 
            except Exception as e:
                logger.error(f"Error processing question: {e}")
        
        return EvaluationDataset(items=items)
    
    def get_relevant_docs(
        self, 
        query: str,
        use_reranker: bool = False,
        rerank_top_k: int = 10
    ) -> list[Document]:
        """
        Retrieve relevant documents for query with optional reranking.
        
        Args:
            query: Search query
            use_reranker: Whether to apply reranking
            rerank_top_k: Number of documents to keep after reranking
            
        Returns:
            List of relevant documents
        """
        docs = self.retrieval.invoke(query)
        
        if use_reranker and self.reranker:
            callback_handler = self._get_callback_handler(verbose=False)
            docs = self._apply_reranking(query, docs, rerank_top_k, callback_handler)
        
        return docs

    def store_documents(self, documents: list[Document]) -> None:
        """Store documents in vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully stored {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to store documents: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear lexical retriever cache if applicable."""
        if hasattr(self.retrieval, 'clear_cache'):
            self.retrieval.clear_cache()
            logger.info("Retrieval cache cleared")
