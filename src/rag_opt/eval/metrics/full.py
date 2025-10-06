"""
Those list of metrics evaluate the full RAG pipeline response
"""
from rag_opt.eval.metrics.base import BaseMetric,MetricCategory
from rag_opt.dataset import EvaluationDataset

# Those metrics are taken from
# https://arxiv.org/abs/2502.18635

# NOTE:: we need to make it parameter to update dynamimcally
WORST_COST_PER_QUERY = 0.2 # DOLLAR 
WORST_LATENCY_PER_QUERY = 7 # seconds

class CostMetric(BaseMetric):
    """ cost per embedding, reranker, vectorstore, llm """
    name: str = "cost"
    negate: bool = True
    category: MetricCategory = MetricCategory.FULL
    is_llm_based:bool = False
    # defined here https://arxiv.org/abs/2502.18635
    def __init__(self,*args, **kwargs):
        super().__init__( *args, **kwargs)
    
    def _evaluate(self, dataset:EvaluationDataset, **kwargs) -> list[float]:
        """Evaluate avg total cost per token used including API, compute, and storage costs"""
        if not dataset.items:
            return []
        return [item.cost.total for item in dataset.items]

    @property
    def worst_value(self):
        """ worse cost per query. You can override this """
        return WORST_COST_PER_QUERY


class LatencyMetric(BaseMetric):
    """ total latency over the full RAG pipeline from embedding > generation"""
    # defined here https://arxiv.org/abs/2502.18635
    name: str = "latency"
    negate: bool = True 
    category: MetricCategory = MetricCategory.FULL
    is_llm_based:bool = False
    def __init__(self,*args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def _evaluate(self, dataset:EvaluationDataset, **kwargs) -> list[float]:
        """Evaluate avg latency taken to answer the dataset questions """
        if not dataset.items:
            return []
        return [item.latency.total for item in dataset.items]

    @property
    def worst_value(self):
        """ worse latency per query. You can override this """
        return WORST_LATENCY_PER_QUERY