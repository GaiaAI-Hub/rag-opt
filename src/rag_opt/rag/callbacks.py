from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from rag_opt.rag._pricing import PricingRegistry 
from langchain_core.messages import AIMessage
from langchain.schema import Document
from dataclasses import dataclass
from loguru import logger
from typing import Any, Optional
import threading
from uuid import UUID
import time


@dataclass
class LLMUsageStats:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    prompt_tokens_cached: int = 0
    reasoning_tokens: int = 0


@dataclass
class EmbeddingUsageStats:
    total_tokens: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0


@dataclass
class RerankerUsageStats:
    total_documents: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0


@dataclass
class ToolUsageStats:
    tool_calls: int = 0
    retrieval_calls: int = 0
    total_latency: float = 0.0


class RAGCallbackHandler(BaseCallbackHandler):
    """Unified callback handler for RAG pipeline to track cost, usage and latency"""
    
    def __init__(self,
                 embedding_provider_name: str = None,
                 embedding_model_name: str = None,
                 llm_provider_name: str = None,
                 llm_model_name: str = None,
                 reranker_provider_name: str = None,
                 reranker_model_name: str = None,
                 vector_store_provider_name: str = None,
                 verbose: bool = True) -> None:
        super().__init__()
        self._verbose = verbose
        self.usage_metadata: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._retrieved_contexts: list[str] = []
        
        self.llm_stats = LLMUsageStats()
        self.embedding_stats = EmbeddingUsageStats()
        self.reranker_stats = RerankerUsageStats()
        self.tool_stats = ToolUsageStats()
        self._start_times: dict[str, float] = {}

        # Provider and model names
        self.embedding_provider_name = embedding_provider_name
        self.embedding_model_name = embedding_model_name
        self.llm_provider_name = llm_provider_name
        self.llm_model_name = llm_model_name
        self.reranker_provider_name = reranker_provider_name
        self.reranker_model_name = reranker_model_name
        self.vector_store_provider_name = vector_store_provider_name

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def retrieved_contexts(self) -> list[str]:
        return self._retrieved_contexts
    
    @retrieved_contexts.setter
    def retrieved_contexts(self, value: list[str]):
        self._retrieved_contexts = value
    
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None and self.verbose:
            logger.info("RAG Pipeline Started")
        self._start_times[run_id] = time.time()

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if run_id in self._start_times:
            latency = time.time() - self._start_times.pop(run_id)
            if parent_run_id is None and self.verbose:
                logger.success(f"RAG Pipeline Completed - Latency: {latency:.3f}s")
                logger.warning(self.get_summary())

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start_times[run_id] = time.time()
        if self.verbose:
            logger.info("LLM Call Started")

    def on_llm_end(self, response: LLMResult, run_id=None, **kwargs: Any) -> None:
        """Track LLM usage and distinguish between tool calls and regular responses"""
        
        latency = time.time() - self._start_times.pop(run_id, time.time())
        
        try:
            generation = response.generations[0][0]
        except IndexError:
            if self.verbose:
                logger.warning("No generation found in LLM response")
            return
        
        if not isinstance(generation, ChatGeneration):
            return
        
        message = generation.message
        if not isinstance(message, AIMessage):
            return
        
        has_tool_calls = bool(getattr(message, 'tool_calls', None))
        
        # Track tool vs LLM latency
        with self._lock:
            if has_tool_calls:
                self.tool_stats.tool_calls += 1
                self.tool_stats.total_latency += latency
                if self.verbose:
                    logger.info(f"Tool Call - Latency: {latency:.3f}s")
                return
            else:
                self.llm_stats.total_latency += latency
                if self.verbose:
                    logger.info(f"LLM Response - Latency: {latency:.3f}s")
        
        # Extract usage data
        usage_metadata = self._extract_usage(message, response)
        
        if not usage_metadata:
            with self._lock:
                self.llm_stats.successful_requests += 1
            return
        
        # Extract model name
        model_name = self.llm_model_name or self._extract_model_name(message, response)
        
        # Update stats with cost calculation
        self._update_llm_stats(usage_metadata, model_name)
        self._log_current_stats()

    def _extract_model_name(self, message: AIMessage, response: LLMResult) -> str:
        """Extract model name from message or response"""
        if hasattr(message, 'response_metadata'):
            model = message.response_metadata.get('model_name') or message.response_metadata.get('model')
            if model:
                return model
        
        if response.llm_output:
            model = response.llm_output.get('model_name') or response.llm_output.get('model')
            if model:
                return model
        
        return "unknown"

    def _extract_usage(self, message: AIMessage, response: LLMResult) -> dict | None:
        """Extract usage metadata from various sources"""
        
        # Method 1: Modern usage_metadata
        usage_metadata = getattr(message, 'usage_metadata', None)
        if usage_metadata:
            return usage_metadata
        
        # Method 2: From response_metadata
        if hasattr(message, 'response_metadata') and 'usage' in message.response_metadata:
            usage_data = message.response_metadata['usage']
            return {
                'total_tokens': usage_data.get('total_tokens', 0),
                'input_tokens': usage_data.get('prompt_tokens', 0),
                'output_tokens': usage_data.get('completion_tokens', 0)
            }
        
        # Method 3: From llm_output
        if response.llm_output and 'usage' in response.llm_output:
            usage_data = response.llm_output['usage']
            return {
                'total_tokens': usage_data.get('total_tokens', 0),
                'input_tokens': usage_data.get('prompt_tokens', 0),
                'output_tokens': usage_data.get('completion_tokens', 0)
            }
        
        return None

    def _update_llm_stats(self, usage_metadata: dict, model_name: str):
        """Update LLM stats with cost calculation"""
        
        total_tokens = usage_metadata.get('total_tokens', 0)
        input_tokens = usage_metadata.get('input_tokens', 0)
        output_tokens = usage_metadata.get('output_tokens', 0)
        
        input_details = usage_metadata.get('input_token_details', {})
        output_details = usage_metadata.get('output_token_details', {})
        cached_tokens = input_details.get('cache_read', 0)
        reasoning_tokens = output_details.get('reasoning', 0)
        
        with self._lock:
            # Update LLM stats
            self.llm_stats.total_tokens += total_tokens
            self.llm_stats.prompt_tokens += input_tokens
            self.llm_stats.completion_tokens += output_tokens
            self.llm_stats.prompt_tokens_cached += cached_tokens
            self.llm_stats.reasoning_tokens += reasoning_tokens
            self.llm_stats.successful_requests += 1
            
            # Update per-model usage
            if model_name not in self.usage_metadata:
                self.usage_metadata[model_name] = usage_metadata.copy()
            else:
                try:
                    from langchain_core.messages.ai import add_usage
                    self.usage_metadata[model_name] = add_usage(
                        self.usage_metadata[model_name], usage_metadata
                    )
                except ImportError:
                    existing = self.usage_metadata[model_name]
                    self.usage_metadata[model_name] = {
                        'total_tokens': existing.get('total_tokens', 0) + total_tokens,
                        'input_tokens': existing.get('input_tokens', 0) + input_tokens,
                        'output_tokens': existing.get('output_tokens', 0) + output_tokens
                    }
            
            # Calculate cost
            if self.llm_provider_name and model_name:
                try:
                    cost = PricingRegistry.calculate_llm_cost(
                        provider=self.llm_provider_name,
                        model=model_name,
                        usage=usage_metadata
                    )
                    if cost is not None:
                        self.llm_stats.total_cost += cost
                        if self.verbose:
                            logger.info(f"LLM Cost: ${cost:.6f}")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not calculate LLM cost: {e}")

    def on_retriever_start(self, serialized, query, *, run_id, **kwargs):
        if run_id not in self._start_times:
            self._start_times[run_id] = time.time()
            if self.verbose:
                logger.info(f"Retriever Started - Query: {query[:50]}...")
        # Calculate embedding cost for retrieval
        self._calculate_embedding_cost_for_retrieval(kwargs.get('query', ''), 0)

    def on_retriever_end(self, documents: list[Document], *, run_id, **kwargs):
        if run_id in self._start_times:
            latency = time.time() - self._start_times.pop(run_id)
            
            with self._lock:
                self.tool_stats.retrieval_calls += 1
                self.tool_stats.total_latency += latency
            
            if self.verbose:
                logger.success(f"Retrieval Completed - {len(documents)} docs - Latency: {latency:.3f}s")
            
            self.retrieved_contexts = [doc.page_content for doc in documents]
            
            # self._calculate_embedding_cost_for_retrieval(kwargs.get('query', ''), latency)
            self._log_current_stats()

    def _calculate_embedding_cost_for_retrieval(self, query: str, latency: float):
        """Calculate and track embedding cost for a retrieval operation"""
        if not self.embedding_provider_name or not self.embedding_model_name:
            return
        
        try:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(query) // 4
            
            usage_metadata = {
                'input_tokens': estimated_tokens,
                'output_tokens': 0,
                'total_tokens': estimated_tokens
            }
            
            cost = PricingRegistry.calculate_embedding_cost(
                provider=self.embedding_provider_name,
                model=self.embedding_model_name,
                usage=usage_metadata
            )
            
            if cost is not None:
                with self._lock:
                    self.embedding_stats.total_tokens += estimated_tokens
                    self.embedding_stats.total_requests += 1
                    self.embedding_stats.total_cost += cost
                    self.embedding_stats.total_latency += latency
                    
                if self.verbose:
                    logger.info(f"Embedding Cost: ${cost:.6f}")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not calculate embedding cost: {e}")

    def track_reranking(self, documents: list[Document], latency: float):
        """Track reranking operation cost and stats"""
        if not self.reranker_provider_name or not self.reranker_model_name:
            return
        
        try:
            cost = PricingRegistry.calculate_reranker_cost(
                provider=self.reranker_provider_name,
                model=self.reranker_model_name,
                docs=documents,
                num_requests=1
            )
            
            if cost is not None:
                with self._lock:
                    self.reranker_stats.total_documents += len(documents)
                    self.reranker_stats.total_requests += 1
                    self.reranker_stats.total_cost += cost
                    self.reranker_stats.total_latency += latency
                    
                if self.verbose:
                    logger.info(f"Reranker Cost: ${cost:.6f}")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not calculate reranker cost: {e}")

    def on_agent_action(self, action, **kwargs):
        if self.verbose:
            logger.info(f"Agent Action: {action.tool}")
    

    def _log_current_stats(self):
        """Log current stats after operations"""
        if self.verbose:
            logger.info(
                f"Current Stats - LLM Requests: {self.llm_stats.successful_requests}, "
                f"Tokens: {self.llm_stats.total_tokens}, "
                f"LLM Cost: ${self.llm_stats.total_cost:.6f}, "
                f"Embedding Cost: ${self.embedding_stats.total_cost:.6f}, "
                f"Reranker Cost: ${self.reranker_stats.total_cost:.6f}, "
                f"Total Cost: ${self.get_total_cost():.6f}"
            )

    def get_total_cost(self) -> float:
        """Get total cost across all components"""
        return (self.llm_stats.total_cost + 
                self.embedding_stats.total_cost + 
                self.reranker_stats.total_cost)

    def get_summary(self) -> str:
        """Get a summary of all tracked metrics"""
        avg_llm_latency = (
            self.llm_stats.total_latency / max(1, self.llm_stats.successful_requests)
        )
        avg_tool_latency = (
            self.tool_stats.total_latency / max(1, self.tool_stats.tool_calls)
        )
        avg_embedding_latency = (
            self.embedding_stats.total_latency / max(1, self.embedding_stats.total_requests)
        )
        avg_reranker_latency = (
            self.reranker_stats.total_latency / max(1, self.reranker_stats.total_requests)
        )
        
        return f"""
                RAG Pipeline Summary:
                ===================
                LLM Stats:
                - Requests: {self.llm_stats.successful_requests}
                - Total Tokens: {self.llm_stats.total_tokens}
                - Prompt Tokens: {self.llm_stats.prompt_tokens}
                - Completion Tokens: {self.llm_stats.completion_tokens}
                - Total Cost: ${self.llm_stats.total_cost:.6f}
                - Avg Latency: {avg_llm_latency:.3f}s

                Embedding Stats:
                - Requests: {self.embedding_stats.total_requests}
                - Total Tokens: {self.embedding_stats.total_tokens}
                - Total Cost: ${self.embedding_stats.total_cost:.6f}
                - Avg Latency: {avg_embedding_latency:.3f}s

                Reranker Stats:
                - Requests: {self.reranker_stats.total_requests}
                - Documents Processed: {self.reranker_stats.total_documents}
                - Total Cost: ${self.reranker_stats.total_cost:.6f}
                - Avg Latency: {avg_reranker_latency:.3f}s

                Tool Stats:
                - Tool Calls: {self.tool_stats.tool_calls}
                - Retrieval Calls: {self.tool_stats.retrieval_calls}
                - Avg Tool Latency: {avg_tool_latency:.3f}s

                Total Cost: ${self.get_total_cost():.6f}
                Models Used: {list(self.usage_metadata.keys())}
                """

    def reset(self):
        """Reset all counters"""
        with self._lock:
            self.llm_stats = LLMUsageStats()
            self.embedding_stats = EmbeddingUsageStats()
            self.reranker_stats = RerankerUsageStats()
            self.tool_stats = ToolUsageStats()
            self.usage_metadata.clear()
            self._start_times.clear()
            self._retrieved_contexts.clear()