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

# NOTE:: use ReAct, langchain agent , langgraph (make custom agentic RAG possible)
# NOTE;: allow async and running server

class GAIARAG:
    """Main RAG pipeline class"""
    
    agent_executor: Annotated[AgentExecutor, Doc("The agent executor in case of running handling RAG process using agent")] = None
    
    def __init__(self, 
                 embeddings, 
                 vector_store: VectorStore, 
                 llm: Annotated[RAGLLM, Doc("the llm to be used in the dataset evaluation process")],
                 reranker: Optional[BaseReranker] = None,
                 retrieval_config: Optional[dict] = None,
                 *,
                 max_workers: Annotated[int, Doc("Maximum workers for parallel component loading")] = 5,
                 executor: Annotated[Optional[Executor], Doc("The thread pool executor for batch evaluation ")] = None,
                 **kwargs):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

        # Initialize retrieval
        retrieval_config = retrieval_config or {"search_type": kwargs.get("search_type", "similarity"), "k": kwargs.get("k", 5)} 
        
        self.retrieval = Retriever(vector_store, 
                                   corpus_documents=kwargs.get("corpus_documents", None),
                                   **retrieval_config)
        self.retrieval_tool = create_retriever_tool(
            self.retrieval,
            "retrieve_relative_context",
            "Search and return information required to answer the question",
        )

        if executor is None:
            self.executor = get_shared_executor(max_workers)
        
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
        self.agent_executor = AgentExecutor( # NOTE:: we wanna handle reranking for agent executor
            agent=self.agent,
            tools=tools,
            verbose=False,
        )
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        return prompt
    
    def _create_rag_chain(self, use_reranker: bool = False, reranker: Optional[BaseReranker] = None) -> RunnableSequence:
        """Create the complete RAG chain"""
        
        prompt = self._create_rag_prompt()
        
        def format_docs(docs: list[Document]):
            """Format documents into a single string"""
            return "\n\n".join(doc.page_content for doc in docs)
        
        if use_reranker and reranker is not None:
            def retrieve_and_rerank(query: str) -> list[Document]:
                docs = self.retrieval.invoke(query)
                reranked_docs = reranker.rerank(query=query, documents=docs, top_k=10)
                return reranked_docs
            
            retrieval_chain = retrieve_and_rerank
        else:
            retrieval_chain = self.retrieval
        
        rag_chain = (
            {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    

    def _generate_eval_metadata(self,callback_handler:RAGCallbackHandler) -> dict[str, ComponentUsage]:
        """ Generate evaluation metadata to be used later for metrics evaluation """
        # NOTE:: for now we will be using llm as alternative to total
        return {
            "cost": ComponentUsage(llm=callback_handler.llm_stats.total_cost, embedding=0.0, vectorstore=0.0, reranker=0.0), 
            "latency": ComponentUsage(llm=callback_handler.llm_stats.total_latency, embedding=0.0, vectorstore=0.0, reranker=0.0)
             }
     

    def get_batch_answers(self, 
                          dataset: TrainDataset,
                           **kwargs) -> EvaluationDataset:
        """ get answers for all dataset questions > useful for preparing evaluation dataset """

        futures: list[Future[EvaluationDatasetItem]] = []

        for item in dataset.items:
            futures.append(self.executor.submit(self.get_answer, 
                                            query=item.question, 
                                            ground_truth=item.to_ground_truth(), 
                                            **kwargs))
        
        items: list[EvaluationDatasetItem] = []
        for future in as_completed(futures):
            items.append(future.result())
        
        return EvaluationDataset(items=items)
    
    def get_agentic_batch_answer(self,**kwargs):
        raise NotImplementedError


    def get_answer(
        self, 
        query: str,
        *,
        ground_truth: Optional[GroundTruth] = None,
        **kwargs
    ) -> EvaluationDatasetItem:
        """
        Process query through RAG pipeline.
        
        Args:
            query: Question to answer
            ground_truth: Optional ground truth for evaluation
        """
        callback_handler = RAGCallbackHandler(verbose=False)
        response = self.rag_chain.invoke(
            query, config={"callbacks": [callback_handler]}
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
    
    def get_agentic_answer(self, query: str) -> EvaluationDatasetItem:
        """Process query through Agentic RAG pipeline"""
        callback_handler = RAGCallbackHandler()
        response = self.agent_executor.invoke(
            {"input": query}, 
            config={"callbacks": [callback_handler]}
        )
        contexts = getattr(callback_handler, 'retrieved_contexts', [])
        return EvaluationDatasetItem(
            question=query,
            answer=response.get("output"),
            cost=0.0,
            latency=0.0,
            contexts=contexts
        )
    
    def get_relevant_docs(self, query: str) -> list[Document]:
        """Retrieve relevant documents for query"""
        return self.retrieval.retrieve(query)

    def store_documents(self, documents: list[Document]):
        self.vector_store.add_documents(documents)
        


