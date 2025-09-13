from rag_opt.eval.metrics.base import BaseMetric, MetricResult, MetricCategory, MetricScope
from rag_opt.dataset import RAGDataset
from loguru import logger

# These Metrics works for both (retriever and reranker)

# (both) Context Precision (Measures the proportion of retrieved documents that are actually relevant. Retriever brings candidates; reranker improves precision by reordering the most relevant.)
# (retriever) Context Recall (Measures coverage: does the retriever find most of the relevant documents? Reranker cannot improve recall if documents aren’t retrieved.)
# (both) Context Entities Recall: Focuses on whether the key entities/faracts are included in the retrieved set. Both components matter: retriever finds them, reranker prioritizes the important ones.

class ContextPrecision(BaseMetric):
    """Context Precision (Measures the proportion of retrieved documents that are actually relevant. 
       it also take into consideration the position of the retrieved documents (reranker precision).)"""
    
    name: str = "context_precision"
    def __init__(self):
        super().__init__("context_precision_llm_no_ref", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
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
        den = sum(contexts_verifications)
        if not den:
            logger.warning("No relevant contexts found")
            return MetricResult(
                name=self.name,
                value=0.0
            )
        num = sum([sum(contexts_verifications[:i+1])/(i+1) * contexts_verifications[i] for i in range(len(contexts_verifications)+1)])
        return MetricResult(
            name=self.name,
            value=num/den
        )

    def evaluate(self, dataset:RAGDataset, **kwargs) -> MetricResult:
        """Calculate context precision"""
        # TODO:: generate list of verifications estimated by LLM
        contexts_verifications = []
        return self._evaluate(contexts_verifications)
    
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


