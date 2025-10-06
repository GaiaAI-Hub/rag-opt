"""
Those list of metrics evaluate the quality of the generated response
"""
from rag_opt._prompts import SAFETY_PROMPT, ALIGNMENT_PROMPT, RESPONSE_RELEVANCY_PROMPT
from rag_opt.eval.metrics.base import BaseMetric,  MetricCategory
from langchain_core.messages import BaseMessage
from rag_opt.dataset import EvaluationDataset
import rag_opt._utils as _utils
from rag_opt.llm import RAGLLM
from loguru import logger
import json 


class _LLMGenerationMetric(BaseMetric):
    """ a generateion int-score based LLM Metric"""
    is_llm_based: bool = True
    category: MetricCategory = MetricCategory.GENERATION

    def __init__(self,llm:RAGLLM, prompt: str=None, *args, **kwargs):
        super().__init__(llm,prompt, *args, **kwargs)
        self._limit_contexts = kwargs.get("limit_contexts", 3)
    
    def _prepare_prompts(self, dataset:EvaluationDataset) -> list[str]:
        """ Prepare prompts for LLM 

        NOTE:: this consider that all subclasses use the same prompt parameters (contexts, question, answer)
        """
        prompts = []
        for item in dataset.items:
            prompt = self.prompt.format(
                contexts=item.contexts[:self._limit_contexts],
                question=item.question,
                answer=item.answer
            )
            prompts.append(prompt)
        return prompts

    def generate_metric_score(self, dataset: EvaluationDataset) -> list[float]:
        """Common LLM verification logic return metric score (0-1)"""
        if not self.llm:
            logger.error(f"LLM is required to evaluate {self.name}")
            raise ValueError(f"LLM is required to evaluate {self.name}")
        
        prompts = self._prepare_prompts(dataset)
        responses = self.llm.batch(prompts)
        scores =  self._parse_llm_responses(responses) # 0-100
        return [score / 100 for score in scores]

    def _parse_llm_responses(self, responses: list[BaseMessage]) -> list[float]:
        """Parse LLM responses into list of int scores"""
        items = []
        for response in responses:
            try:
                items.append(float(response.content))
            except (json.JSONDecodeError, ValueError, TypeError):
                fallback_item = _utils.extract_num_from_text(str(response.content))
                if fallback_item is not None:
                    items.append(fallback_item)
                else:
                    logger.warning(f"Failed to parse LLM response: {response.content}")
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                items.append(0)
        return items
    
    def _evaluate(self,dataset: EvaluationDataset, **kwargs) -> list[float]:
        """Main Evaluation Logic"""
        return self.generate_metric_score(dataset)
    

class SafetyMetric(_LLMGenerationMetric):
    """ check whether or not the response from a RAG system is supported by the context (Faithfullness)"""
    # defined here https://arxiv.org/abs/2502.18635
    _prompt_template: str = SAFETY_PROMPT
    name: str = "safety"

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)


class AlignmentMetric(_LLMGenerationMetric):
    """ judges how useful, detailed and unambiguous a response is (Helpfulness)

    Here we mean by aligned helpful not just it answers the question but it is also well structured, clear and unambiguous
    """
    _prompt_template: str = ALIGNMENT_PROMPT
    name: str = "alignment"

    # defined here https://arxiv.org/abs/2502.18635
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    

class ResponseRelevancy(_LLMGenerationMetric):
    """ How relevant the generated response is to the query

    Here we focus on the answer relevancy to the question (not how helpful it is )
    """
    _prompt_template: str = RESPONSE_RELEVANCY_PROMPT
    name: str = "response_relevancy"

    def __init__(self,*args, **kwargs):
        super().__init__( *args, **kwargs)