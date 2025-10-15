from langchain_core.prompts import PromptTemplate, get_template_variables
from typing_extensions import Annotated, Doc, Optional, Any
from rag_opt.dataset import EvaluationDataset
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rag_opt.llm import RAGLLM
import rag_opt._utils as _utils
from loguru import logger
from enum import Enum 
import re 

class MetricCategory(Enum):
    """Categories for different types of metrics"""
    FULL = "FULL"        
    RETRIEVAL = "retrieval"     
    GENERATION = "generation"   



@dataclass
class MetricResult:
    """Standard result structure for all metrics"""
    name: str
    value: float
    category: MetricCategory
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



EPSILON_OFFSET = 1e-6 # for numerical stability

class BaseMetric(ABC):
    """Base class for all metrics
    Note: ALl Metrics will be llm-based (which means llm will be the main source of truth)
    """
    name: str 
    category: MetricCategory
    prompt: PromptTemplate = None
    _prompt_template: str = None
    is_llm_based: bool = True
    negate: Annotated[bool, "if True the metric value will be negated, which means we need to minimize this metric"] = False
    _worst_value: Annotated[float, "Worst-case value for this metric (for reference point estimation)"] = EPSILON_OFFSET

    def __init__(self, 
                 llm: Annotated[Optional[RAGLLM], Doc("the llm to be used in the dataset evaluation process")] = None,
                 prompt: Annotated[str, Doc("the prompt template to be used for evaluating context precision")] = None,
                 **kwargs
                 ):
        self.name = self.name or self._generate_name_from_cls()

        if self.is_llm_based and not llm:
            logger.error("LLM is required in this metric")
            raise ValueError("LLM is required in this metric")
        
        self.llm = llm

        if self.is_llm_based and prompt:
            _utils.validate_prompt(self._prompt_template,prompt, raise_error=True)

        
        if self.is_llm_based:    
            self.prompt = PromptTemplate(template = self._prompt_template or prompt, 
                                         input_variables=get_template_variables(self._prompt_template, "f-string"))
    
    @property
    def worst_value(self) -> float:
        """Worst-case value (per query) for this metric (for reference point estimation)"""
        if self._worst_value is not None:
            return self._worst_value
        return EPSILON_OFFSET
    
    @property
    def _prompt_template(self) -> str:
        """Every metric must define a prompt template"""
        if self.is_llm_based and not self._prompt_template:
            logger.error("Prompt template is required in this metric")
            raise ValueError("Prompt template is required in this metric")

    @abstractmethod
    def _evaluate(self, dataset:EvaluationDataset, **kwargs) -> list[float]:
        """Evaluate the metric and return structured result"""
        raise NotImplementedError("evaluate method not implemented")
    

    def evaluate(self, 
                dataset:EvaluationDataset,
                **kwargs) -> MetricResult:
        """ Main Evaluation Logic"""
        scores =  self._evaluate(dataset, **kwargs)
        if not scores:
            return MetricResult(name=self.name, value=self.worst_value, category=self.category)
        
        return MetricResult(name=self.name, 
                            value=sum(scores)/len(scores) - EPSILON_OFFSET * (1 if self.negate else -1) , 
                            category=self.category, 
                            metadata={"scores": scores})
        

    def _generate_name_from_cls(self):
        return _camel_to_snake(self.__class__.__name__)

    
    
