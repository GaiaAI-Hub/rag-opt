from rag_opt.eval.metrics.base import BaseMetric, MetricCategory
from rag_opt._prompts import CONTEXT_PRECISION_PROMPT, CONTEXT_RECALL_PROMPT
from langchain_core.messages import BaseMessage
from rag_opt.dataset import EvaluationDataset
import rag_opt._utils as _utils
from abc import ABC
from loguru import logger
from rag_opt.llm import RAGEmbedding
import math
import json
from collections import OrderedDict
from typing import Optional
import asyncio

TEXT_SIMILARITY_THRESHOLD = 0.8


class LRUCache:
    """Simple LRU cache with max size to prevent unbounded growth"""
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[list[float]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: list[float]):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        """Clear all cached embeddings"""
        self.cache.clear()
    
    def __len__(self):
        return len(self.cache)


class RetrievalMetrics(BaseMetric, ABC):
    """Base class for retrieval metrics with optimized async embedding caching"""
    category: MetricCategory = MetricCategory.RETRIEVAL
    is_llm_based: bool = False
    is_embedding_based: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit_contexts = kwargs.get("limit_contexts", 3)
        self.embedding_model: RAGEmbedding = kwargs.get("embedding_model", None)
        
        self.max_prompt_tokens = kwargs.get("max_prompt_tokens", 16000)
        self.max_context_length = kwargs.get("max_context_length", 4000)
        
        cache_size = kwargs.get("embedding_cache_size", 1000)
        self._embedding_cache = LRUCache(max_size=cache_size)
        
        # REDUCED: Smaller embedding batch size to prevent hanging
        self.embedding_batch_size = kwargs.get("embedding_batch_size", 10)  # Was 50
        
        # Token encoder cache 
        self._token_encoder = None
        try:
            import tiktoken
            self._token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"tiktoken not available: {e}. Using approximation.")

        if self.is_embedding_based and not self.embedding_model:
            logger.error(f"Embedding is required to evaluate {self.name}")
            raise ValueError(f"Embedding is required to evaluate {self.name}")
    
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        self.clear_cache()
        self._token_encoder = None
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        if hasattr(self, '_embedding_cache'):
            self._embedding_cache.clear()
            logger.debug(f"Cleared embedding cache for {self.name}")
    
    @property
    def limit_contexts(self):
        return self._limit_contexts

    @limit_contexts.setter
    def limit_contexts(self, value: int):
        self._limit_contexts = value

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count with cached encoder"""
        if self._token_encoder:
            return len(self._token_encoder.encode(text))
        return len(text) // 4

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text to max_chars, trying to preserve sentence boundaries"""
        if len(text) <= max_chars:
            return text
        
        truncated = text[:max_chars]
        last_period = truncated.rfind('. ')
        last_newline = truncated.rfind('\n')
        boundary = max(last_period, last_newline)
        
        if boundary > max_chars * 0.8:
            return truncated[:boundary + 1] + "..."
        
        return truncated + "..."

    def _fit_contexts_to_token_limit(
        self, 
        contexts: list[str], 
        template_overhead: str,
        max_tokens: int = None
    ) -> list[str]:
        """Dynamically adjust contexts to fit within token limit"""
        if max_tokens is None:
            max_tokens = self.max_prompt_tokens
        
        template_tokens = self._estimate_tokens(template_overhead)
        available_tokens = max_tokens - template_tokens - 100
        
        if available_tokens <= 0:
            logger.error("Template overhead exceeds token limit!")
            return []
        
        limited_contexts = contexts[:self.limit_contexts]
        contexts_text = "\n".join(limited_contexts)
        contexts_tokens = self._estimate_tokens(contexts_text)
        
        if contexts_tokens <= available_tokens:
            return limited_contexts
        
        num_contexts = self.limit_contexts
        while num_contexts > 0:
            limited_contexts = contexts[:num_contexts]
            contexts_text = "\n".join(limited_contexts)
            contexts_tokens = self._estimate_tokens(contexts_text)
            
            if contexts_tokens <= available_tokens:
                return limited_contexts
            
            num_contexts -= 1
        
        if contexts:
            max_chars = available_tokens * 4
            truncated = self._truncate_text(contexts[0], max_chars)
            logger.warning(
                f"Single context exceeds limit. Truncated from "
                f"{len(contexts[0])} to {len(truncated)} characters"
            )
            return [truncated]
        
        return []

    def _build_prompt_with_token_limit(
        self, 
        template: str, 
        contexts: list[str],
        **kwargs
    ) -> str:
        """Build prompt ensuring it fits within token limit"""
        template_overhead = template.format(context="[CONTEXTS_PLACEHOLDER]", **kwargs)
        fitted_contexts = self._fit_contexts_to_token_limit(contexts, template_overhead)
        return template.format(context=fitted_contexts, **kwargs)

    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        if self.is_llm_based:
            raise NotImplementedError
    
    async def _verify_with_llm_async(self, prompts: list[str]) -> list[float]:
        """ASYNC LLM verification using sequential invoke (no batching)"""
        if not self.llm:
            logger.error(f"LLM is required to evaluate {self.name}")
            raise ValueError(f"LLM is required to evaluate {self.name}")
        
        # Process prompts
        processed_prompts = []
        for idx, prompt in enumerate(prompts):
            token_count = self._estimate_tokens(prompt)
            
            if token_count > self.max_prompt_tokens:
                logger.warning(
                    f"Prompt {idx} exceeds token limit: {token_count} > {self.max_prompt_tokens}. Truncating..."
                )
                prompt = self._truncate_prompt_fallback(prompt, self.max_prompt_tokens)
            
            processed_prompts.append(prompt)
        
        # SEQUENTIAL invoke instead of batching
        all_responses = []
        for idx, prompt in enumerate(processed_prompts):
            try:
                # Use invoke instead of batch
                response = await asyncio.to_thread(self.llm.invoke, prompt)
                all_responses.append(response)
            except Exception as e:
                logger.error(f"LLM invoke failed for prompt {idx}: {e}")
                all_responses.append(None)
        
        return self._parse_llm_responses(all_responses)
    
    def _verify_with_llm(self, prompts: list[str]) -> list[float]:
        """Wrapper to call async version"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._verify_with_llm_async(prompts))

    def _truncate_prompt_fallback(self, prompt: str, max_tokens: int) -> str:
        """Fallback truncation for prompts that exceed limit"""
        max_chars = max_tokens * 4
        
        if len(prompt) <= max_chars:
            return prompt
        
        if "Context:" in prompt or "context:" in prompt:
            parts = prompt.split("\n")
            header_parts = []
            context_parts = []
            footer_parts = []
            
            in_context = False
            for part in parts:
                if "context:" in part.lower():
                    in_context = True
                    header_parts.append(part)
                elif "question:" in part.lower() or "answer:" in part.lower():
                    in_context = False
                    footer_parts.append(part)
                elif in_context:
                    context_parts.append(part)
                elif footer_parts:
                    footer_parts.append(part)
                else:
                    header_parts.append(part)
            
            header_text = "\n".join(header_parts)
            footer_text = "\n".join(footer_parts)
            overhead = len(header_text) + len(footer_text) + 100
            available_for_context = max(max_chars - overhead, max_chars // 2)
            
            context_text = "\n".join(context_parts)
            if len(context_text) > available_for_context:
                context_text = self._truncate_text(context_text, available_for_context)
                context_text += "\n[... contexts truncated due to length ...]"
            
            return f"{header_text}\n{context_text}\n{footer_text}"
        
        return self._truncate_text(prompt, max_chars) + "\n[Content truncated to fit token limit]"

    def _get_cached_embedding(self, text: str) -> list[float]:
        """Get embedding with LRU caching to prevent unbounded growth"""
        text_clean = text.strip()
        
        cached = self._embedding_cache.get(text_clean)
        if cached is not None:
            return cached
        
        embedding = self.embedding_model.embed_query(text_clean)
        self._embedding_cache.put(text_clean, embedding)
        
        return embedding
    
    async def _precompute_embeddings_async(self, contexts_list: list[list[str]]):
        """ASYNC: Precompute embeddings in small batches with progress tracking"""
        if not self.embedding_model:
            return
        
        unique_contexts = set()
        for contexts in contexts_list:
            for ctx in contexts:
                unique_contexts.add(ctx.strip())
        
        uncached_contexts = [
            ctx for ctx in unique_contexts 
            if self._embedding_cache.get(ctx) is None
        ]
        
        if not uncached_contexts:
            logger.debug("All embeddings already cached")
            return
        
        total_contexts = len(uncached_contexts)
        logger.info(f"Precomputing {total_contexts} embeddings (batch_size={self.embedding_batch_size})...")
        
        try:
            for i in range(0, total_contexts, self.embedding_batch_size):
                batch = uncached_contexts[i:i + self.embedding_batch_size]
                batch_num = i // self.embedding_batch_size + 1
                total_batches = (total_contexts - 1) // self.embedding_batch_size + 1
                
                # Async embedding batch
                try:
                    if hasattr(self.embedding_model, 'embed_documents'):
                        embeddings = await asyncio.to_thread(
                            self.embedding_model.embed_documents, 
                            batch
                        )
                        for ctx, emb in zip(batch, embeddings):
                            self._embedding_cache.put(ctx, emb)
                    else:
                        # Sequential fallback
                        for ctx in batch:
                            emb = await asyncio.to_thread(
                                self.embedding_model.embed_query, 
                                ctx
                            )
                            self._embedding_cache.put(ctx, emb)
                except Exception as e:
                    logger.warning(f"Batch {batch_num} failed: {e}. Using sequential.")
                    for ctx in batch:
                        try:
                            emb = await asyncio.to_thread(
                                self.embedding_model.embed_query, 
                                ctx
                            )
                            self._embedding_cache.put(ctx, emb)
                        except Exception as seq_e:
                            logger.error(f"Failed to embed: {seq_e}")
                
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"Progress: {min(i + self.embedding_batch_size, total_contexts)}/{total_contexts}")
            
            logger.info(f"Embedding precomputation complete. Cache size: {len(self._embedding_cache)}")
        
        except Exception as e:
            logger.error(f"Embedding precomputation failed: {e}. Will compute on-demand.")
    
    def _precompute_embeddings(self, contexts_list: list[list[str]]):
        """Wrapper for async precompute"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self._precompute_embeddings_async(contexts_list))
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _is_match(self, retrieved_ctx: str, ground_truth_contexts: list[str]) -> bool:
        """Check if retrieved context matches any ground truth (uses cached embeddings)"""
        retrieved_clean = retrieved_ctx.strip()

        for gt_ctx in ground_truth_contexts:
            gt_clean = gt_ctx.strip()
            
            if retrieved_clean == gt_clean or retrieved_clean.lower() == gt_clean.lower():
                return True
            
            # Word-level overlap check
            r_words = set(retrieved_clean.lower().split())
            g_words = set(gt_clean.lower().split())
            overlap = len(r_words & g_words)
            
            if overlap > 0:
                word_overlap = overlap / max(len(r_words | g_words), 1)
                if word_overlap >= 0.6:
                    return True
        
        # Semantic similarity check with cached embeddings
        if self.embedding_model:
            try:
                retrieved_embedding = self._get_cached_embedding(retrieved_clean)
                
                for gt_ctx in ground_truth_contexts:
                    gt_clean = gt_ctx.strip()
                    gt_embedding = self._get_cached_embedding(gt_clean)
                    
                    cosine_sim = self._cosine_similarity(retrieved_embedding, gt_embedding)
                    if cosine_sim >= 0.7:
                        return True
            except Exception as e:
                logger.warning(f"Embedding-based similarity failed: {e}")
        
        return False


# *************************
# Context-Based Metrics 
# *************************
class ContextPrecision(RetrievalMetrics):
    """Context Precision: Measures relevance of retrieved documents using Average Precision"""
    
    name: str = "context_precision"
    _prompt_template: str = CONTEXT_PRECISION_PROMPT
    is_llm_based: bool = True
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "context_precision")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _calculate_context_precision(self, contexts_verifications: list[int]) -> list[float]:
        """Calculate context precision using Average Precision (AP) formula"""
        if not contexts_verifications:
            logger.warning("No relevant contexts found")
            return []

        tot_sum = sum(contexts_verifications)
        
        if tot_sum == 0:
            num = 0.0
        else:
            num = sum([
                sum(contexts_verifications[:i+1]) / (i+1) * contexts_verifications[i] 
                for i in range(len(contexts_verifications))
            ])
        return [num / tot_sum] if tot_sum > 0 else [0.0]
        
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Calculate context precision for a single query"""
        contexts_verifications = self._verify_contexts(dataset, **kwargs)
        return self._calculate_context_precision(contexts_verifications)
    
    def _verify_contexts(self, dataset: EvaluationDataset, **kwargs) -> list[int]:
        """Verify if contexts are relevant using LLM"""
        prompts = []
        
        for item in dataset.items:
            prompt = self._build_prompt_with_token_limit(
                template=self._prompt_template,
                contexts=item.contexts,
                question=item.question,
                answer=item.answer
            )
            prompts.append(prompt)

        return self._verify_with_llm(prompts)
    
    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[int]:
        """Parse LLM responses into context verifications"""
        items = []
        for i, response in enumerate(responses):
            if response is None:
                logger.warning(f"Response {i} is None, defaulting to 0")
                items.append(0)
                continue
                
            try:
                content = response.content.strip()
                items.append(int(content))
            except (json.JSONDecodeError, ValueError):
                fallback_item = _utils.extract_num_from_text(str(response.content))
                if fallback_item is not None:
                    items.append(fallback_item)
                else:
                    logger.warning(f"Failed to parse response {i}: '{response.content}' - defaulting to 0")
                    items.append(0)
            except Exception as e:
                logger.error(f"Error parsing response {i}: {e}")
                items.append(0)
        
        return items

class ContextRecall(RetrievalMetrics):
    """Context Recall: Measures how well retrieval finds ALL relevant information"""
    
    name: str = "context_recall"
    _prompt_template: str = CONTEXT_RECALL_PROMPT
    is_llm_based: bool = True
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "context_recall")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Verify ground truth statements in retrieved contexts"""
        prompts = []
        
        for item in dataset.items:
            combined_text = f"Contexts: {item.contexts}\nGround Truth: {item.ground_truth.contexts}"
            token_count = self._estimate_tokens(combined_text)
            
            if token_count > self.max_prompt_tokens * 0.8:
                available_tokens = int(self.max_prompt_tokens * 0.4)
                
                fitted_contexts = self._fit_contexts_to_token_limit(
                    item.contexts,
                    "placeholder",
                    max_tokens=available_tokens
                )
                
                fitted_ground_truth = self._fit_contexts_to_token_limit(
                    item.ground_truth.contexts,
                    "placeholder", 
                    max_tokens=available_tokens
                )
                
                prompt = self._prompt_template.format(
                    contexts=fitted_contexts,
                    ground_truth=fitted_ground_truth,
                    question=item.question
                )
            else:
                prompt = self._prompt_template.format(
                    contexts=item.contexts,
                    ground_truth=item.ground_truth.contexts,  
                    question=item.question
                )
            
            prompts.append(prompt)
        
        return self._verify_with_llm(prompts)
    
    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        """Parse LLM responses into attribution scores"""
        attributions = []
        for i, response in enumerate(responses):
            if response is None:
                logger.warning(f"Response {i} is None, defaulting to 0.0")
                attributions.append(0.0)
                continue
                
            try:
                data = json.loads(response.content)
                if isinstance(data, list):
                    attributions.extend(data)
                else:
                    attributions.append(float(data))
            except (json.JSONDecodeError, ValueError):
                fallback = _utils.extract_num_from_text(str(response.content))
                if fallback is not None:
                    attributions.append(fallback)
                else:
                    logger.warning(f"Failed to parse LLM response {i}: {response.content}")
                    attributions.append(0.0)
        
        return attributions


# *************************
# Ranking-Based Metrics
# *************************
class MRR(RetrievalMetrics):
    """Mean Reciprocal Rank: Position of first relevant result"""
    
    name: str = "mrr"
    is_llm_based: bool = False
    is_embedding_based: bool = True

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "mrr")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Calculate MRR with precomputed embeddings (OPTIMIZED)"""
        # PRECOMPUTE ALL EMBEDDINGS FIRST - Major speedup!
        all_contexts = []
        for item in dataset.items:
            all_contexts.append(item.contexts)
            all_contexts.append(item.ground_truth.contexts)
        
        self._precompute_embeddings(all_contexts)
        
        # Now calculate MRR with cached embeddings
        reciprocal_ranks = []
        
        for item in dataset.items:
            rank = self._find_first_relevant_rank(
                retrieved_contexts=item.contexts,
                ground_truth_contexts=item.ground_truth.contexts
            )
            
            rr = 1.0 / rank if rank > 0 else 0.0
            reciprocal_ranks.append(rr)
            
            if rr == 0.0:
                logger.warning(
                    f"MRR: No relevant context found. "
                    f"Retrieved: {len(item.contexts)}, Ground truth: {len(item.ground_truth.contexts)}"
                )
        
        return reciprocal_ranks
    
    def _find_first_relevant_rank(
        self, 
        retrieved_contexts: list[str], 
        ground_truth_contexts: list[str]
    ) -> int:
        """Find rank (1-indexed) of first relevant context"""
        for rank, retrieved_ctx in enumerate(retrieved_contexts, start=1):
            if self._is_match(retrieved_ctx, ground_truth_contexts):
                return rank
        
        logger.debug("No relevant context found in retrieved results")
        return 0


class NDCG(RetrievalMetrics):
    """Normalized Discounted Cumulative Gain: Ranking quality with graded relevance"""
    
    name: str = "ndcg"
    is_llm_based: bool = False
    is_embedding_based: bool = True
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "ndcg")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Calculate NDCG with precomputed embeddings"""
        if not dataset.items:
            return []
        
        # precompute all embeddings first (for speedup) 
        all_contexts = []
        for item in dataset.items:
            all_contexts.append(item.contexts)
            all_contexts.append(item.ground_truth.contexts)
        
        self._precompute_embeddings(all_contexts)
        
        # Calculate NDCG with cached embeddings
        ndcg_scores = []
        for item in dataset.items:
            score = self._calculate_ndcg(item.ground_truth.contexts, item.contexts)
            ndcg_scores.append(score)
            
            if score == 0.0:
                logger.warning(
                    f"NDCG: Zero score. "
                    f"Retrieved: {len(item.contexts)}, Ground truth: {len(item.ground_truth.contexts)}"
                )
        
        return ndcg_scores
    
    def _calculate_ndcg(self, ground_truth: list[str], retrieved: list[str]) -> float:
        """Calculate NDCG for a single query"""
        relevance_map = self._get_relevance_scores(ground_truth, retrieved)
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = sum(
            (2 ** relevance_map.get(doc, 0) - 1) / math.log2(i + 2)
            for i, doc in enumerate(retrieved)
        )
        
        # Calculate IDCG (Ideal DCG)
        ideal_scores = sorted(relevance_map.values(), reverse=True)
        idcg = sum(
            (2 ** score - 1) / math.log2(i + 2)
            for i, score in enumerate(ideal_scores)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _get_relevance_scores(self, ground_truth: list[str], retrieved: list[str]) -> dict[str, int]:
        """Map retrieved docs to graded relevance scores"""
        relevance_scores = {}
        
        for doc in retrieved:
            rank = self._get_context_rank(doc, ground_truth)
            if rank >= 0:
                relevance_scores[doc] = len(ground_truth) - rank
            else:
                relevance_scores[doc] = 0
        
        return relevance_scores
    
    def _get_context_rank(self, retrieved_context: str, ground_truth_contexts: list[str]) -> int:
        """Find rank (0-indexed) of retrieved context in ground truth"""
        retrieved_clean = retrieved_context.strip()
        
        for idx, gt_ctx in enumerate(ground_truth_contexts):
            gt_clean = gt_ctx.strip()
            
            if retrieved_clean == gt_clean or retrieved_clean.lower() == gt_clean.lower():
                return idx
            
            # Word overlap check
            r_words = set(retrieved_clean.lower().split())
            g_words = set(gt_clean.lower().split())
            overlap = len(r_words & g_words)
            
            if overlap > 0:
                word_overlap = overlap / max(len(r_words | g_words), 1)
                if word_overlap >= 0.6:
                    return idx
        
        # Semantic similarity check with cached embeddings
        if self.embedding_model:
            try:
                retrieved_embedding = self._get_cached_embedding(retrieved_clean)
                
                for idx, gt_ctx in enumerate(ground_truth_contexts):
                    gt_clean = gt_ctx.strip()
                    gt_embedding = self._get_cached_embedding(gt_clean)
                    
                    cosine_sim = self._cosine_similarity(retrieved_embedding, gt_embedding)
                    if cosine_sim >= 0.7:
                        return idx
            except Exception as e:
                logger.warning(f"Embedding-based matching failed: {e}")
        
        return -1
