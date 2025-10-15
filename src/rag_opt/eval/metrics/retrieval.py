"""
Evaluation metrics for Embedding + VectorStore + Reranker
"""
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

TEXT_SIMILARITY_THRESHOLD = 0.8 


class RetrievalMetrics(BaseMetric, ABC):
    """Base class for retrieval metrics"""
    category: MetricCategory = MetricCategory.RETRIEVAL
    is_llm_based: bool = False # by default
    is_embedding_based: bool = False # used for similarity

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit_contexts = kwargs.get("limit_contexts", 3)
        self.embedding_model: RAGEmbedding = kwargs.get("embedding_model", None)
        
        # Token management configuration
        self.max_prompt_tokens = kwargs.get("max_prompt_tokens", 16000)
        self.batch_size = kwargs.get("batch_size", 10)
        self.max_context_length = kwargs.get("max_context_length", 4000) 

        if self.is_embedding_based and not self.embedding_model:
            logger.error(f"Embedding is required to evaluate {self.name}")
            raise ValueError(f"Embedding is required to evaluate {self.name}")
    
    @property
    def limit_contexts(self):
        return self._limit_contexts

    @limit_contexts.setter
    def limit_contexts(self, value: int):
        self._limit_contexts = value

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text. 
        Uses rough approximation: ~4 characters per token.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            logger.warning(f"Failed to estimate tokens with tiktoken: {e} using rough approximation")
        return len(text) // 4

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text to max_chars, trying to preserve sentence boundaries"""
        if len(text) <= max_chars:
            return text
        
        truncated = text[:max_chars]
        # Try to end at sentence or newline boundary
        last_period = truncated.rfind('. ')
        last_newline = truncated.rfind('\n')
        boundary = max(last_period, last_newline)
        
        if boundary > max_chars * 0.8:  # Keep at least 80%
            return truncated[:boundary + 1] + "..."
        
        return truncated + "..."

    def _fit_contexts_to_token_limit(
        self, 
        contexts: list[str], 
        template_overhead: str,
        max_tokens: int = None
    ) -> list[str]:
        """
        Dynamically adjust contexts to fit within token limit.
        
        Strategy:
        1. Start with limit_contexts
        2. If exceeds token limit, reduce number of contexts
        3. If still exceeds, truncate individual contexts
        
        Args:
            contexts: List of context strings
            template_overhead: The prompt template with placeholder for contexts
            max_tokens: Maximum tokens allowed (defaults to self.max_prompt_tokens)
        
        Returns:
            List of contexts that fit within token limit
        """
        if max_tokens is None:
            max_tokens = self.max_prompt_tokens
        
        # Estimate template overhead (without contexts)
        template_tokens = self._estimate_tokens(template_overhead)
        available_tokens = max_tokens - template_tokens - 100  # 100 token buffer
        
        if available_tokens <= 0:
            logger.error("Template overhead exceeds token limit!")
            return []
        
        # Strategy 1: Try with original limit_contexts
        limited_contexts = contexts[:self.limit_contexts]
        contexts_text = "\n".join(limited_contexts)
        contexts_tokens = self._estimate_tokens(contexts_text)
        
        if contexts_tokens <= available_tokens:
            return limited_contexts
        
        # Strategy 2: Reduce number of contexts
        num_contexts = self.limit_contexts
        while num_contexts > 0:
            limited_contexts = contexts[:num_contexts]
            contexts_text = "\n".join(limited_contexts)
            contexts_tokens = self._estimate_tokens(contexts_text)
            
            if contexts_tokens <= available_tokens:
                # if num_contexts < self.limit_contexts:
                #     logger.warning(
                #         f"Reduced contexts from {self.limit_contexts} to {num_contexts} to fit token limit"
                #     )
                return limited_contexts
            
            num_contexts -= 1
        
        # Strategy 3: If even 1 context is too large, truncate it
        if contexts:
            max_chars = available_tokens * 4  # rough conversion
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
        """
        Build prompt ensuring it fits within token limit.
        
        Args:
            template: Prompt template with {context} placeholder
            contexts: List of context strings
            **kwargs: Other template variables (question, answer, etc.)
        
        Returns:
            Formatted prompt that fits within token limit
        """
        # Create template overhead by replacing context with placeholder
        template_overhead = template.format(context="[CONTEXTS_PLACEHOLDER]", **kwargs)
        
        # Fit contexts to available token space
        fitted_contexts = self._fit_contexts_to_token_limit(
            contexts, 
            template_overhead
        )
        
        # Build final prompt
        return template.format(context=fitted_contexts, **kwargs)

    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        if self.is_llm_based:
            raise NotImplementedError
    
    def _verify_with_llm(self, prompts: list[str]) -> list[float]:
        """Common LLM verification logic with comprehensive token limit handling"""
        if not self.llm:
            logger.error(f"LLM is required to evaluate {self.name}")
            raise ValueError(f"LLM is required to evaluate {self.name}")
        
        # Step 1: Validate and truncate individual prompts
        processed_prompts = []
        for idx, prompt in enumerate(prompts):
            token_count = self._estimate_tokens(prompt)
            
            if token_count > self.max_prompt_tokens:
                logger.warning(
                    f"Prompt {idx} exceeds token limit: {token_count} > {self.max_prompt_tokens}. "
                    "This shouldn't happen with proper context fitting. Truncating as fallback..."
                )
                prompt = self._truncate_prompt_fallback(prompt, self.max_prompt_tokens)
            
            processed_prompts.append(prompt)
        
        # Step 2: Process in batches to avoid overwhelming the API
        all_responses = []
        total_batches = (len(processed_prompts) - 1) // self.batch_size + 1
        
        for batch_idx in range(0, len(processed_prompts), self.batch_size):
            batch = processed_prompts[batch_idx:batch_idx + self.batch_size]
            logger.debug(
                f"Processing batch {batch_idx//self.batch_size + 1}/{total_batches} "
                f"({len(batch)} prompts)"
            )
            
            try:
                responses = self.llm.batch(batch)
                all_responses.extend(responses)
            except Exception as e:
                logger.error(f"Batch processing failed for batch {batch_idx//self.batch_size + 1}: {e}")
                # Add placeholder responses for failed batch
                all_responses.extend([None] * len(batch))
        
        return self._parse_llm_responses(all_responses)

    def _truncate_prompt_fallback(self, prompt: str, max_tokens: int) -> str:
        """
        Fallback truncation for prompts that still exceed limit.
        This should rarely be called if _fit_contexts_to_token_limit works correctly.
        """
        max_chars = max_tokens * 4
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Try to identify and truncate the contexts section
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
            
            # Calculate space available for contexts
            header_text = "\n".join(header_parts)
            footer_text = "\n".join(footer_parts)
            overhead = len(header_text) + len(footer_text) + 100
            available_for_context = max(max_chars - overhead, max_chars // 2)
            
            # Truncate context parts
            context_text = "\n".join(context_parts)
            if len(context_text) > available_for_context:
                context_text = self._truncate_text(context_text, available_for_context)
                context_text += "\n[... contexts truncated due to length ...]"
            
            return f"{header_text}\n{context_text}\n{footer_text}"
        
        # Simple truncation
        return self._truncate_text(prompt, max_chars) + "\n[Content truncated to fit token limit]"


# *************************
# Context-Based Metrics
# *************************
class ContextPrecision(RetrievalMetrics):
    """
    Context Precision: Measures the proportion of retrieved documents that are relevant,
    considering their position in the ranked list (Average Precision).
    
    CP = (Relevant chunks retrieved) / (Total chunks retrieved) = TP / (TP + FP)
    Answers: How many retrieved documents are relevant?
    """
    
    name: str = "context_precision"
    _prompt_template: str = CONTEXT_PRECISION_PROMPT
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "context_precision")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _calculate_context_precision(self, contexts_verifications: list[int]) -> list[float]:
        """
        Calculate context precision using Average Precision (AP) formula:
        
        AP = Σ((Σ y_j from j=1..i) / i) * y_i / (Σ y_i + ε)
        
        where y_i = 1 if i-th item is relevant, else 0
        """
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
        """Verify if contexts are relevant using LLM with dynamic token management"""
        prompts = []
        
        for item in dataset.items:
            # Build prompt with automatic context fitting
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
                logger.warning(f"Response {i} is None (failed batch), defaulting to 0")
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
    """
    Context Recall: Measures how well the retrieval finds ALL relevant information.
    
    CR = (Ground truth statements found in contexts) / (Total ground truth statements)
    CR = TP / (TP + FN)
    
    Answers: Did I retrieve ALL the information needed to answer correctly?
    """
    
    name: str = "context_recall"
    _prompt_template: str = CONTEXT_RECALL_PROMPT
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "context_recall")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Verify which ground truth statements can be attributed to retrieved contexts"""
        prompts = []
        
        for item in dataset.items:
            # Build prompt with automatic context fitting
            # Note: For context recall, we need to fit both retrieved contexts and ground truth
            combined_text = f"Contexts: {item.contexts}\nGround Truth: {item.ground_truth.contexts}"
            token_count = self._estimate_tokens(combined_text)
            
            if token_count > self.max_prompt_tokens * 0.8:  # Leave 20% buffer
                # Reduce contexts proportionally
                available_tokens = int(self.max_prompt_tokens * 0.4)  # 40% each for contexts and ground truth
                
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
        """Parse LLM responses into attribution list"""
        attributions = []
        for i, response in enumerate(responses):
            if response is None:
                logger.warning(f"Response {i} is None (failed batch), defaulting to 0.0")
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
    """
    Mean Reciprocal Rank: Focuses on the position of the first relevant result.
    
    For each query: 1 / rank_of_first_relevant_result
    Simple metric that emphasizes getting at least one good result early.
    Particularly useful for evaluating reranker effectiveness.
    """
    name: str = "mrr"
    is_llm_based: bool = False  
    is_embedding_based: bool = True 

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "mrr")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """
        Calculate Mean Reciprocal Rank across all queries.
        
        For each query, finds the rank of the first relevant context
        and calculates 1/rank. If no relevant context found, score is 0.
        """
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
                    f"MRR: No relevant context found for query. "
                    f"Retrieved: {len(item.contexts)} contexts, "
                    f"Ground truth: {len(item.ground_truth.contexts)} contexts"
                )
        
        return reciprocal_ranks
    
    def _find_first_relevant_rank(
        self, 
        retrieved_contexts: list[str], 
        ground_truth_contexts: list[str]
    ) -> int:
        """
        Find the rank (1-indexed) of the first relevant context.
        Returns 0 if no relevant context is found.
        """
        for rank, retrieved_ctx in enumerate(retrieved_contexts, start=1):
            if self._is_match(retrieved_ctx, ground_truth_contexts):
                return rank
        
        logger.debug("No relevant context found in retrieved results")
        return 0
    
    def _is_match(self, retrieved_ctx: str, ground_truth_contexts: list[str]) -> bool:
        """
        Check if a retrieved context matches any ground truth context.
        
        Uses multiple matching strategies:
        1. Exact match (case-sensitive and case-insensitive)
        2. Word overlap similarity
        3. Semantic similarity via embeddings (if available)
        """
        retrieved_clean = retrieved_ctx.strip()

        # Quick exact match
        for gt_ctx in ground_truth_contexts:
            gt_clean = gt_ctx.strip()
            
            if retrieved_clean == gt_clean:
                return True
            
            if retrieved_clean.lower() == gt_clean.lower():
                return True
            
            # Word-level overlap check
            r_words = set(retrieved_clean.lower().split())
            g_words = set(gt_clean.lower().split())
            overlap = len(r_words & g_words)
            
            if overlap > 0:
                word_overlap = overlap / max(len(r_words | g_words), 1)
                if word_overlap >= 0.6:
                    return True
        
        # Semantic similarity check
        if self.embedding_model:
            try:
                retrieved_embedding = self.embedding_model.embed_query(retrieved_clean)
                
                for gt_ctx in ground_truth_contexts:
                    gt_clean = gt_ctx.strip()
                    gt_embedding = self.embedding_model.embed_query(gt_clean)
                    
                    cosine_sim = self._cosine_similarity(retrieved_embedding, gt_embedding)
                    if cosine_sim >= 0.7:
                        return True
            except Exception as e:
                logger.warning(f"Embedding-based similarity failed: {e}")
        
        return False

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        """Not used for MRR."""
        raise NotImplementedError("MRR does not use batch LLM processing")


class NDCG(RetrievalMetrics):
    """
    Normalized Discounted Cumulative Gain: Measures ranking quality with graded relevance.
    
    Takes into account:
        1. Graded relevance scores based on position in ground truth
        2. Position in the ranked list (logarithmic discount for lower positions)
    
    Evaluates ranking quality by comparing against ideal ranking.
    """
    name: str = "ndcg"
    is_llm_based: bool = False  
    is_embedding_based: bool = True 
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "ndcg")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        super().__init__(*args, **kwargs)
    
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Calculate NDCG for each query."""
        if not dataset.items:
            return []
        
        ndcg_scores = []
        for item in dataset.items:
            score = self._calculate_ndcg(item.ground_truth.contexts, item.contexts)
            ndcg_scores.append(score)
            
            if score == 0.0:
                logger.warning(
                    f"NDCG: Zero score for query. "
                    f"Retrieved: {len(item.contexts)} contexts, "
                    f"Ground truth: {len(item.ground_truth.contexts)} contexts"
                )
        
        return ndcg_scores
    
    def _calculate_ndcg(self, ground_truth: list[str], retrieved: list[str]) -> float:
        """
        Calculate NDCG for a single query.
        Uses graded relevance based on position in ground truth ranking.
        """
        relevance_map = self._get_relevance_scores(ground_truth, retrieved)
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = sum(
            (2 ** relevance_map.get(doc, 0) - 1) / math.log2(i + 2)
            for i, doc in enumerate(retrieved)
        )
        
        # Calculate IDCG (Ideal DCG - perfect ranking)
        ideal_scores = sorted(relevance_map.values(), reverse=True)
        idcg = sum(
            (2 ** score - 1) / math.log2(i + 2)
            for i, score in enumerate(ideal_scores)
        )
        
        result = dcg / idcg if idcg > 0 else 0.0
        return result
    
    def _get_relevance_scores(self, ground_truth: list[str], retrieved: list[str]) -> dict[str, int]: 
        """
        Map retrieved docs to graded relevance scores based on ground truth position.
        
        Scoring:
        - Higher position in ground truth = higher relevance score
        - Score = (len(ground_truth) - position) for matched contexts
        - Score = 0 for non-matching contexts
        
        Example: If ground_truth has 5 items:
        - 1st position gets score of 5
        - 2nd position gets score of 4
        - Not found gets score of 0
        """
        relevance_scores = {}
        
        for doc in retrieved:
            rank = self._get_context_rank(doc, ground_truth)
            if rank >= 0:
                relevance_scores[doc] = len(ground_truth) - rank
            else:
                relevance_scores[doc] = 0
        
        return relevance_scores
    
    def _get_context_rank(self, retrieved_context: str, ground_truth_contexts: list[str]) -> int:
        """
        Find the rank (0-indexed) of a retrieved context in ground truth.
        Returns -1 if no match is found.
        
        Uses multiple matching strategies:
        1. Exact match (case-sensitive and case-insensitive)
        2. Word overlap similarity
        3. Semantic similarity via embeddings (if available)
        """
        retrieved_clean = retrieved_context.strip()
        
        for idx, gt_ctx in enumerate(ground_truth_contexts):
            gt_clean = gt_ctx.strip()
            
            # Exact match
            if retrieved_clean == gt_clean:
                return idx
            
            # Case-insensitive match
            if retrieved_clean.lower() == gt_clean.lower():
                return idx
            
            # Word overlap check
            r_words = set(retrieved_clean.lower().split())
            g_words = set(gt_clean.lower().split())
            overlap = len(r_words & g_words)
            
            if overlap > 0:
                word_overlap = overlap / max(len(r_words | g_words), 1)
                if word_overlap >= 0.6:
                    return idx
        
        # Semantic similarity check
        if self.embedding_model:
            try:
                retrieved_embedding = self.embedding_model.embed_query(retrieved_clean)
                
                for idx, gt_ctx in enumerate(ground_truth_contexts):
                    gt_clean = gt_ctx.strip()
                    gt_embedding = self.embedding_model.embed_query(gt_clean)
                    
                    cosine_sim = self._cosine_similarity(retrieved_embedding, gt_embedding)
                    if cosine_sim >= 0.7:
                        return idx
            except Exception as e:
                logger.warning(f"Embedding-based matching failed: {e}")
        
        return -1
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        """Not used for NDCG."""
        raise NotImplementedError("NDCG does not use batch LLM processing")