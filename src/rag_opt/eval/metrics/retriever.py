from rag_opt.eval.metrics.base import BaseMetric, MetricResult, MetricCategory, MetricScope
from rag_opt._prompts import CONTEXT_PRECISION_PROMPT
from langchain_core.prompts import PromptTemplate, get_template_variables
from langchain_core.messages import BaseMessage
from rag_opt.llm import RAGLLM
from rag_opt.dataset import RAGDataset
from loguru import logger
import json
import re 
# These Metrics works for both (retriever and reranker)

# (both) Context Precision (Measures the proportion of retrieved documents that are actually relevant. Retriever brings candidates; reranker improves precision by reordering the most relevant.)
# (retriever) Context Recall (Measures coverage: does the retriever find most of the relevant documents? Reranker cannot improve recall if documents aren’t retrieved.)
# (both) Context Entities Recall: Focuses on whether the key entities/faracts are included in the retrieved set. Both components matter: retriever finds them, reranker prioritizes the important ones.

class ContextPrecision(BaseMetric):
    """Context Precision (Measures the proportion of retrieved documents that are actually relevant. 
       it also take into consideration the position of the retrieved documents (reranker precision).)"""
    
    name: str = "context_precision"
    _prompt_template: str = CONTEXT_PRECISION_PROMPT
    
    def __init__(self,*args, **kwargs):
        kwargs.setdefault("name", "context_precision")
        kwargs.setdefault("category", MetricCategory.RETRIEVAL)
        kwargs.setdefault("scope", MetricScope.COMPONENT_LEVEL)
        super().__init__(*args,
                         **kwargs)

        limit_contexts = kwargs.get("limit_contexts", 3) 
        self._limit_contexts = limit_contexts
    
    @property
    def limit_contexts(self):
        return self._limit_contexts

    @limit_contexts.setter
    def limit_contexts(self, value: int):
        self._limit_contexts = value

    def _evaluate(self,contexts_verifications: list[int], **kwargs) -> MetricResult:
        """calculate context precision from list of verifications estimated by LLM
            Average Precision (AP) formula:
            
                       Σ ( (Σ y_j from j=1..i) / i ) * y_i
            AP = -------------------------------------------------
                           Σ y_i   +  ε
            
            where:
              - y_i = 1 if the i-th item is relevant, else 0
              - i   = position index in the ranked list
              - ε   = small constant (1e-10) to avoid division by zero
        """
        if not contexts_verifications:
            logger.warning("No relevant contexts found")
            return MetricResult(
                name=self.name,
                value=0.0
            )
        den = sum(contexts_verifications)
        if not den:
            logger.warning("No relevant contexts found")
            return MetricResult(
                name=self.name,
                value=0.0
            )
        # TODO:: Should we make it float precision instead 0-1 ??!!
        num = sum([sum(contexts_verifications[:i+1])/(i+1) * contexts_verifications[i] for i in range(len(contexts_verifications))])
        return MetricResult(
            name=self.name,
            value=num/den
        )

    def evaluate(self, dataset:RAGDataset, **kwargs) -> MetricResult:
        """Calculate context precision"""
        contexts_verifications = self._verify_contexts(dataset, **kwargs)
        return self._evaluate(contexts_verifications)
    
    def _verify_contexts(self, dataset:RAGDataset, **kwargs) -> list[int]:
        """check if contexts are relevant"""
        if not self.llm:
            logger.error("llm is required to evaluate context precision")
            raise ValueError("llm is required to evaluate context precision")
        

        # create list of prompts 
        prompts = []
        for item in dataset.items:
            if len(item.contexts) > self.limit_contexts:
                logger.warning(f"Number of contexts ({len(item.contexts)}) exceeds limit. Limiting contexts to {self.limit_contexts} for prompt generation")
                logger.info(f"if you want to increase the limit, set limit_contexts to the number you want")
            prompt = self.prompt.format(
                context=item.contexts[0:self.limit_contexts], # TODO:: limit the contexts to 3 
                question=item.question,
                answer=item.answer
            )
            prompts.append(prompt)

        # batch 
        responses = self.llm.batch(prompts)
        contexts_verifications = self._parse_llm_responses(responses)
        return contexts_verifications
    
    def _parse_llm_responses(self, 
                             responses: list[BaseMessage], 
                             ) -> list[int]:
        """Parse LLM responses into context verifications"""
        items = []
        for response in responses:
            try:
                data = int(response.content)
                items.append(data)
            except json.JSONDecodeError:
                # Fallback to heuristic parsing
                fallback_item = self._extract_num_from_text(str(response.content))
                if fallback_item:
                    items.append(fallback_item)
                else:
                    logger.warning(f"Failed to parse LLM response: {response.content}")
        return items


    def _extract_num_from_text(self, text: str) -> int | None:
        """
        Fallback function to get a number from text in case the LLM 
        generates extra info in its response.

        - Extracts the first integer found in the string.
        - Falls back to 0 if no number is found.
        """
        if not text:
            return None
        match = re.search(r"-?\d+", text)  # captures first integer, including negatives
        if match:
            try:
                return int(match.group())
            except ValueError:
                logger.error(f"Failed to parse number from text: {text}")
                return 
        return None
    
class ContextRecall(BaseMetric):
    def __init__(self):
        super().__init__("context_recall_non_llm", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Non-LLM based context recall"""
        pass

class ContextEntitiesRecall(BaseMetric):
    def __init__(self):
        super().__init__("context_entities_recall", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Context entities recall evaluation"""
        pass

class RetrievalPrecision(BaseMetric):
    def __init__(self):
        super().__init__("precision", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate retrieval precision"""
        pass

class RetrievalRecall(BaseMetric):
    def __init__(self):
        super().__init__("recall", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate retrieval recall"""
        pass

class RetrievalF1(BaseMetric):
    def __init__(self):
        super().__init__("f1_score", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate F1 score for retrieval"""
        pass

class MRR(BaseMetric):
    def __init__(self):
        super().__init__("mrr", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate Mean Reciprocal Rank"""
        pass

class MAP(BaseMetric):
    def __init__(self):
        super().__init__("map", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate Mean Average Precision"""
        pass

class NDCG(BaseMetric):
    def __init__(self):
        super().__init__("ndcg", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate Normalized Discounted Cumulative Gain"""
        pass

class TokenPrecision(BaseMetric):
    def __init__(self):
        super().__init__("token_precision", MetricCategory.RETRIEVAL, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate token-level precision"""
        pass

class TokenRecall(BaseMetric):
    def __init__(self):
        super().__init__("token_recall", MetricCategory.RETRIEVAL, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate token-level recall"""
        pass

class TokenF1(BaseMetric):
    def __init__(self):
        super().__init__("token_f1", MetricCategory.RETRIEVAL, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate token-level F1 score"""
        pass


