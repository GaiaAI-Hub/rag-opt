from rag_opt.eval.metrics.base import BaseMetric, MetricResult, MetricCategory, MetricScope


# Noise Sensitivity (full) (How much the full RAG pipeline changes with noise)
# Response Relevancy (generator) (How relevant the generated response is to the query)
# Faithfulness (generator) (Whether the generated answer is supported by the retrieved context and doesn’t hallucinate facts.)


class NoiseSensitivity(BaseMetric):
    def __init__(self):
        super().__init__("noise_sensitivity", MetricCategory.RETRIEVAL, MetricScope.COMPONENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate sensitivity to noisy/irrelevant context"""
        pass

# RAGAS Generation Metrics
class ResponseRelevancy(BaseMetric):
    def __init__(self):
        super().__init__("response_relevancy", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate response relevancy to query"""
        pass

class Faithfulness(BaseMetric):
    def __init__(self):
        super().__init__("faithfulness", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate faithfulness to retrieved context"""
        pass

class FaithfulnessHHEM(BaseMetric):
    def __init__(self):
        super().__init__("faithfulness_hhem", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate faithfulness using HHEM approach"""
        pass


# AutoRAG Generation Metrics
class BleuScore(BaseMetric):
    def __init__(self):
        super().__init__("bleu", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate BLEU score"""
        pass

class RougeScore(BaseMetric):
    def __init__(self):
        super().__init__("rouge", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate ROUGE score"""
        pass

class MeteorScore(BaseMetric):
    def __init__(self):
        super().__init__("meteor", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate METEOR score"""
        pass

class SemanticScore(BaseMetric):
    def __init__(self):
        super().__init__("sem_score", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate semantic similarity score"""
        pass

class GEval(BaseMetric):
    def __init__(self):
        super().__init__("g_eval", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Perform G-Eval evaluation"""
        pass

class BertScore(BaseMetric):
    def __init__(self):
        super().__init__("bert_score", MetricCategory.GENERATION, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Calculate BERTScore"""
        pass
