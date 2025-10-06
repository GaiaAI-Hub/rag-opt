from rag_opt.eval.metrics import (
                                    ContextPrecision, ContextRecall, MRR, NDCG,
                                    ResponseRelevancy, SafetyMetric, AlignmentMetric,
                                    CostMetric, LatencyMetric, BaseMetric
                                )
from rag_opt.llm import RAGLLM



def all_metrics_factory(llm: RAGLLM, *args, **kwargs) -> list[BaseMetric]:
    cost = CostMetric()
    latency = LatencyMetric()
    # generation
    response_relevancy = ResponseRelevancy(llm=llm, *args, **kwargs)
    safety = SafetyMetric(llm=llm, *args, **kwargs)
    alignment = AlignmentMetric(llm=llm, *args, **kwargs)

    # retrieval
    context_precision = ContextPrecision(llm=llm, *args, **kwargs)
    context_recall = ContextRecall(llm=llm,*args, **kwargs)
    mrr = MRR(llm=llm, *args, **kwargs)
    ndcg = NDCG(*args, **kwargs)
    return [cost, latency, response_relevancy, safety, alignment,context_precision, context_recall, mrr, ndcg]
