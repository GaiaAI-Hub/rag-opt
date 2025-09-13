from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from rag_opt.dataset import RAGDataset
from enum import Enum 

class MetricCategory(Enum):
    """Categories for different types of metrics"""
    PERFORMANCE = "performance"  # Cost, Latency
    SAFETY = "safety"           # Safety, Alignment
    RETRIEVAL = "retrieval"     # Context-related metrics
    GENERATION = "generation"   # Response quality metrics

class MetricScope(Enum):
    """Scope of metric evaluation"""
    SYSTEM_LEVEL = "system"     # Overall system metrics (cost, latency)
    COMPONENT_LEVEL = "component"  # Specific component metrics
    CONTENT_LEVEL = "content"   # Content quality metrics

@dataclass
class MetricResult:
    """Standard result structure for all metrics"""
    name: str
    value: float # all metrics should be qualitative / digitalized
    metadata: Optional[dict[str, Any]] = None
    error: Optional[str] = None

class BaseMetric(ABC):
    """Base class for all metrics"""
    
    def __init__(self, name: str, category: MetricCategory, scope: MetricScope):
        self.name = name
        self.category = category
        self.scope = scope
    
    @abstractmethod
    def evaluate(self, dataset:RAGDataset, **kwargs) -> MetricResult:
        """Evaluate the metric and return structured result"""
        raise NotImplementedError("evaluate method not implemented")
