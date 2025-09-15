from langchain_core.prompts import PromptTemplate, get_template_variables
from typing_extensions import Annotated, Doc, Optional, Any
from rag_opt.dataset import RAGDataset
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rag_opt.llm import RAGLLM
from loguru import logger
from enum import Enum 
import re 

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

    def __repr__(self):
        if self.error:
            return f"{self.name}: {self.error}"
        return f"{self.name}: {self.value}"

def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()

class BaseMetric(ABC):
    """Base class for all metrics"""
    _prompt_template: str = None
    prompt: PromptTemplate = None

    def __init__(self, 
                 llm: Annotated[RAGLLM, Doc("the llm to be used in the dataset generation process")],
                 prompt: Annotated[str, Doc("the prompt template to be used for evaluating context precision")] = None,
                 *,
                 name: Annotated[str, Doc("the name of the metric")] = None,
                 category: Annotated[MetricCategory, Doc("the category of the metric")] = MetricCategory.RETRIEVAL, 
                 scope: Annotated[MetricScope, Doc("the scope of the metric")] = MetricScope.COMPONENT_LEVEL, 
                 ):
        self.name = name or self._generate_name_from_cls()
        self.category = category
        self.scope = scope
        self.llm = llm

        if prompt:
            self._validate_prompt(prompt)
            
        self.prompt = PromptTemplate(template = self._prompt_template or prompt, 
                                         input_variables=get_template_variables(self._prompt_template, "f-string"))
    @property
    @abstractmethod
    def _prompt_template(self) -> str:
        """Every metric must define a prompt template"""

    @abstractmethod
    def evaluate(self, dataset:RAGDataset, **kwargs) -> MetricResult:
        """Evaluate the metric and return structured result"""
        raise NotImplementedError("evaluate method not implemented")

    def _generate_name_from_cls(self):
        return _camel_to_snake(self.__class__.__name__)

    def _validate_prompt(self,prompt_template:str):
        vars = get_template_variables(prompt_template, "f-string")
        prompt_vars = get_template_variables(self._prompt_template, "f-string")
        needed_vars = set(prompt_vars) - set(vars)
        diff = set(vars) ^ set(prompt_vars)
        if needed_vars:
            logger.error(f"Your prompt is missing the following variables: {needed_vars}")
            raise ValueError(f"Your prompt is missing the following variables: {needed_vars}")
        
        if diff:
            logger.error(f"Your prompt contains extra variables: {diff} \n your prompt must only contain the following variables only:  {prompt_vars}")
            raise ValueError(f"Your prompt contains extra variables: {diff}")
    

    