from rag_opt.eval.metrics.base import BaseMetric, MetricResult, MetricCategory, MetricScope

# Those metrics are taken from
# https://arxiv.org/abs/2502.18635

# Performance Metrics
class CostMetric(BaseMetric):
    def __init__(self):
        super().__init__("cost", MetricCategory.PERFORMANCE, MetricScope.SYSTEM_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate total cost including API, compute, and storage costs"""
        pass

class LatencyMetric(BaseMetric):
    def __init__(self):
        super().__init__("latency", MetricCategory.PERFORMANCE, MetricScope.SYSTEM_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate system latency including retrieval and generation time"""
        pass

# Safety Metrics
class SafetyMetric(BaseMetric):
    def __init__(self):
        super().__init__("safety", MetricCategory.SAFETY, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate content safety and harmful content detection"""
        pass

class AlignmentMetric(BaseMetric):
    def __init__(self):
        super().__init__("alignment", MetricCategory.SAFETY, MetricScope.CONTENT_LEVEL)
    
    def evaluate(self, **kwargs) -> MetricResult:
        """Evaluate alignment with intended behavior and values"""
        pass
