"""
Evaluation metrics for Embedding + VectorStore + Reranker
"""
from rag_opt.eval.metrics.base import BaseMetric, MetricCategory
from rag_opt._prompts import CONTEXT_PRECISION_PROMPT, CONTEXT_RECALL_PROMPT
from langchain_core.messages import BaseMessage
from rag_opt.dataset import EvaluationDataset
import rag_opt._utils as _utils
from abc import abstractmethod, ABC
from loguru import logger
import statistics as st
import textdistance
import math
import json


TEXT_SIMILARITY_THRESHOLD = 0.8 


class RetrievalMetrics(BaseMetric, ABC):
    """Base class for retrieval metrics"""
    category: MetricCategory = MetricCategory.RETRIEVAL
    is_llm_based: bool = False # by default

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit_contexts = kwargs.get("limit_contexts", 3)
    
    @property
    def limit_contexts(self):
        return self._limit_contexts

    @limit_contexts.setter
    def limit_contexts(self, value: int):
        self._limit_contexts = value

    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        if self.is_llm_based:
            raise NotImplementedError
    
    def _verify_with_llm(self, prompts: list[str]) -> list[float]:
        """Common LLM verification logic"""
        if not self.llm:
            logger.error(f"LLM is required to evaluate {self.name}")
            raise ValueError(f"LLM is required to evaluate {self.name}")
        
        responses = self.llm.batch(prompts)
        return self._parse_llm_responses(responses)


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
        if not contexts_verifications or not sum(contexts_verifications):
            logger.warning("No relevant contexts found")
            return []
        
        num = sum([
            sum(contexts_verifications[:i+1]) / (i+1) * contexts_verifications[i] 
            for i in range(len(contexts_verifications))
        ])
        return [num / sum(contexts_verifications)]
        

    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Calculate context precision for a single query"""
        contexts_verifications = self._verify_contexts(dataset, **kwargs)
        return self._calculate_context_precision(contexts_verifications)
    
    def _verify_contexts(self, dataset: EvaluationDataset, **kwargs) -> list[int]:
        """Verify if contexts are relevant using LLM"""
        # NOTE:: This structure could be moved to abstract method (repeated in many classes)
        prompts = []
        for item in dataset.items:
            if len(item.contexts) > self.limit_contexts:
                logger.warning(
                    f"Number of contexts ({len(item.contexts)}) exceeds limit. "
                    f"Limiting to {self.limit_contexts} contexts"
                )
            prompt = self._prompt_template.format(
                context=item.contexts[:self.limit_contexts],
                question=item.question,
                answer=item.answer
            )
            prompts.append(prompt)

        return self._verify_with_llm(prompts)
    
    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[int]:
        """Parse LLM responses into context verifications"""
        items = []
        for response in responses:
            try:
                items.append(int(response.content))
            except (json.JSONDecodeError, ValueError):
                fallback_item = _utils.extract_num_from_text(str(response.content))
                if fallback_item is not None:
                    items.append(fallback_item)
                else:
                    logger.warning(f"Failed to parse LLM response: {response.content}")
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
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
            prompt = self._prompt_template.format(
                contexts=item.contexts,
                ground_truth=item.ground_truth.contexts,  
                question=item.question
            )
            prompts.append(prompt)
        
        # NOTE:: this could be non-llm using regex for ex , sentence similarity
        return self._verify_with_llm(prompts)
    
    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        """Parse LLM responses into attribution list"""
        attributions = []
        for response in responses:
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
                    logger.warning(f"Failed to parse LLM response: {response.content}")
        
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
            
            rr = 1.0 / rank if rank > 0 else 0.0 # limit it to 0 > 1
            reciprocal_ranks.append(rr)
        
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
            # Check if this retrieved context matches any ground truth context
            if self._is_relevant(retrieved_ctx, ground_truth_contexts):
                return rank
        # NOTE:: llmfallback could be done here instead 
        return 0 
    
    
    def _is_relevant(self, retrieved_ctx: str, ground_truth_contexts: list[str]) -> bool:
        """
        Check if a retrieved context is relevant by comparing with ground truth.

        Steps:
        1. Quick word-based equality/overlap check
        2. Use multiple textdistance algorithms and take the average similarity
        3. Semantic similarity (if self.embedding_model is available)
        4. LLM fallback (if self.llm is available)
        """
        retrieved_clean = retrieved_ctx.strip()

        if retrieved_clean in set(ground_truth_contexts):
            return True 

        # Define algorithms to use from textdistance
        algos = [
            textdistance.levenshtein.normalized_similarity,
            textdistance.jaccard.normalized_similarity,
            textdistance.cosine.normalized_similarity,
            textdistance.sorensen_dice.normalized_similarity,
            textdistance.overlap.normalized_similarity,
        ]

        for gt_ctx in ground_truth_contexts:
            gt_clean = gt_ctx.strip()

            # 1. Word-level quick checks
            if retrieved_clean.lower() == gt_clean.lower():
                return True

            r_words, g_words = retrieved_clean.split(), gt_clean.split()
            word_overlap = len(set(r_words) & set(g_words)) / max(len(set(r_words)), 1)
            if word_overlap >= TEXT_SIMILARITY_THRESHOLD: 
                return True

            # 2. Average across similarity algorithms
            scores = [algo(retrieved_clean, gt_clean) for algo in algos]
            avg_score = st.mean(scores)
            if avg_score >= TEXT_SIMILARITY_THRESHOLD:  
                return True


        #  LLM fallback 
        if self.is_llm_based and self.llm is not None:

            if len(ground_truth_contexts) > self.limit_contexts:
                logger.warning(
                    f"Number of ground truth contexts ({len(ground_truth_contexts)}) exceeds limit. "
                    f"Limiting to {self.limit_contexts} contexts."
                )
            try:
                prompt = (
                    f"Does the retrieved context match / exist in the ground truth context?\n\n"
                    f">> Retrieved:\n {retrieved_clean}\n\n"
                    f">> Ground Truth:\n {gt_clean}\n\n"
                    f"Answer 'yes' or 'no'. Dont say anything else. only yes or no"
                )
                response = self.llm.invoke(prompt)
                if "yes" in response.content:
                    return True
            except Exception as e:
                logger.error(f"Failed to evaluate with LLM: {e}")

        return False

    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        """Not used for MRR - no LLM needed for basic implementation"""
        # llm should return bool 
        raise NotImplementedError("MRR does not require LLM-based evaluation")
    
class NDCG(RetrievalMetrics):
    """
    Normalized Discounted Cumulative Gain: Measures ranking quality with graded relevance.
    
    Takes into account:
        1. Graded relevance scores (not just binary relevant/irrelevant)
        2. Position in the ranked list (logarithmic discount for lower positions)
    
    it evaluates our ranking (Reranker, retrieval) based on the ground truth relevance and order
    it asks: How close is your ranking to the ideal ranking?
    """
    is_llm_based: bool = False
    name: str = "ndcg"
    
    def __init__(self):
        super().__init__("ndcg", MetricCategory.RETRIEVAL)
    
    def _evaluate(self, dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Calculate NDCG averaged across all queries."""
        
        if not dataset.items:
            return []
        
        ndcg_scores = [
            self._calculate_ndcg(item.ground_truth.contexts, item.contexts)
            for item in dataset.items
        ]
        return ndcg_scores
        
    
    def _calculate_ndcg(self, ground_truth: list[str], retrieved: list[str]) -> float:
        """Calculate NDCG for a single query."""
        relevance_map = self._get_relevance_scores(ground_truth, retrieved) # {doc1: 0, doc2: 1, ..}
        
        # Calculate DCG
        dcg = sum(
            (2 ** relevance_map.get(doc, 0) - 1) / math.log2(i + 2)
            for i, doc in enumerate(retrieved)
        )
        
        # Calculate IDCG (ideal ranking)
        ideal_scores = sorted(relevance_map.values(), reverse=True)
        
        idcg = sum(
            (2 ** score - 1) / math.log2(i + 2)
            for i, score in enumerate(ideal_scores)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    

    def _get_relevance_scores(self, ground_truth: list[str], retrieved: list[str]) -> dict[str, int]: 
        """
        Map retrieved docs to relevance scores.
        Binary: 1 if in ground truth, 0 otherwise.
        """
        return {doc: (1 if self._get_context_rank(doc, ground_truth) >= 0 else 0) for doc in retrieved}
    

    def _get_context_rank(self,retrieved_context:str, ground_truth_contexts: list[str]) -> int:
        """
        Find the rank (0-indexed) of the first relevant context.
        Returns -1 if no relevant context is found.
        """
        if retrieved_context in ground_truth_contexts:
            return ground_truth_contexts.index(retrieved_context)
        
        # fallback
        for idx, gt_ctx in enumerate(ground_truth_contexts):
            if textdistance.levenshtein.normalized_similarity(retrieved_context, gt_ctx) >= TEXT_SIMILARITY_THRESHOLD:
                return idx
        
        # NOTE:: llm could be used here
        return -1
    
