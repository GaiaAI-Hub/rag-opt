from rag_opt.eval.metrics import MetricResult, MetricCategory, BaseMetric
from concurrent.futures import Future, Executor
from rag_opt.dataset import EvaluationDataset
from typing import Optional, Literal
from rag_opt.llm import RAGLLM, RAGEmbedding
import rag_opt._utils as _utils
from loguru import logger
import torch

NormalizationStrategy = Literal["sum", "softmax", "min-max", "z-score"]

# Default weights for scalarization (NOT used for MOBO)
DEFAULT_WEIGHTS = {
    "cost": 0.3,
    "latency": 0.2,
    "safety": 0.5,
    "alignment": 0.5,
    "response_relevancy": 0.5,
    "context_precision": 0.5,
    "context_recall": 0.3,
    "mrr": 0.3,
    "ndcg": 0.25
}


class RAGEvaluator:
    """
    Multi-objective evaluator for RAG systems.
    
    For Bayesian Optimization:
    - Use evaluate() with normalize=False, return_tensor=True
    - Returns raw objective values (with negation for minimize metrics)
    - Use ref_point property for hypervolume calculation
    """
    
    def __init__(
        self,
        evaluator_llm: Optional[RAGLLM] = None,
        evaluator_embedding: Optional[RAGEmbedding] = None,
        metrics: Optional[list[BaseMetric]] = None,
        *,
        objective_weights: Optional[dict[str, float]] = None,
        auto_initialize_metrics: bool = True,
        executor: Optional[Executor] = None,
        **kwargs
    ):
        self.evaluator_llm = evaluator_llm
        self._metrics: dict[str, BaseMetric] = {}
        self.objective_weights: dict[str, float] = {}
        
        # Load metrics
        if not metrics and auto_initialize_metrics:
            self._initialize_default_metrics(evaluator_llm, evaluator_embedding, **kwargs)
        
        if metrics:
            self.add_metrics(metrics)

        if not self._metrics:
            raise ValueError("No metrics loaded")
        
        self._initialize_weights(objective_weights or DEFAULT_WEIGHTS)
        self._thread_executor = executor or _utils.get_shared_executor()

    @property
    def ref_point(self) -> torch.Tensor:
        """
        Reference point for multi-objective optimization.
        Slightly worse than worst achievable value in transformed space.
        """
        ref_values = []
        for metric in self._metrics.values():
            if metric.negate:
                worst = metric.worst_value if metric.worst_value > 0 else 1.0
                ref_values.append(-worst - 0.1)
            else:
                worst = metric.worst_value if metric.worst_value is not None else 0.0
                ref_values.append(worst - 0.1)
        return torch.tensor(ref_values, dtype=torch.float64)
    
    @property
    def metric_names(self) -> set[str]:
        return set(self._metrics.keys())
    
    @property
    def retrieval_metrics(self) -> dict[str, BaseMetric]:
        return {
            name: metric for name, metric in self._metrics.items()
            if metric.category == MetricCategory.RETRIEVAL
        }
    
    @property
    def generation_metrics(self) -> dict[str, BaseMetric]:
        return {
            name: metric for name, metric in self._metrics.items()
            if metric.category == MetricCategory.GENERATION
        }
    
    @property
    def full_metrics(self) -> dict[str, BaseMetric]:
        return {
            name: metric for name, metric in self._metrics.items()
            if metric.category == MetricCategory.FULL
        }
    
    def _initialize_default_metrics(
        self, 
        llm: RAGLLM, 
        evaluator_embedding: Optional[RAGEmbedding] = None, 
        **kwargs
    ) -> None:
        """Load all default metrics"""
        from rag_opt.eval import all_metrics_factory
        self.add_metrics(all_metrics_factory(llm, evaluator_embedding, **kwargs))
    
    def _initialize_weights(self, weights: dict[str, float]) -> None:
        """Initialize objective weights (only used for scalarization)"""
        if not self._metrics:
            raise ValueError("Cannot initialize weights without metrics")
        
        for name, weight in weights.items():
            if name in self.metric_names:
                self.objective_weights[name] = weight
        
        # Ensure all metrics have weights
        for name in self.metric_names:
            if name not in self.objective_weights:
                logger.warning(f"Metric '{name}' has no weight, defaulting to 0.0")
                self.objective_weights[name] = 0.0
    
    def add_metrics(self, metrics: list[BaseMetric]) -> None:
        """Add multiple metrics"""
        for metric in metrics:
            self.add_metric(metric)
    
    def add_metric(self, metric: BaseMetric, weight: float = 0.0) -> None:
        """Add a single metric with optional weight"""
        if metric.name in self.metric_names:
            logger.warning(f"Overwriting existing metric '{metric.name}'")
        
        self._metrics[metric.name] = metric
        self.objective_weights[metric.name] = weight
    
    def remove_metric(self, name: str) -> None:
        """Remove a metric by name"""
        if name in self._metrics:
            del self._metrics[name]
            self.objective_weights.pop(name, None)
        else:
            logger.warning(f"Cannot remove unknown metric '{name}'")
    
    def evaluate(
        self,
        eval_dataset: EvaluationDataset,
        *,
        return_tensor: bool = True,
        metrics: Optional[dict[str, BaseMetric]] = None,
        normalize: bool = False,
        normalization_strategy: NormalizationStrategy = "sum",
        **kwargs
    ) -> dict[str, MetricResult] | torch.Tensor:
        """
        Evaluate metrics on dataset.
        
        For MOBO: use normalize=False, return_tensor=True
        Returns raw objectives with negation applied.
        
        Args:
            eval_dataset: Dataset to evaluate
            metrics: Specific metrics (defaults to all)
            normalize: Apply normalization (NOT recommended for MOBO)
            return_tensor: Return as tensor
            normalization_strategy: Normalization method
            
        Returns:
            Dictionary of results or tensor of objectives
        """
        metrics_to_eval = metrics or self._metrics
        results: dict[str, MetricResult] = {}
        
        for name, metric in metrics_to_eval.items():
            try:
                result = metric.evaluate(dataset=eval_dataset, **kwargs)
                results[name] = result
            except Exception as e:
                logger.error(f"Error evaluating metric '{name}': {e}")
                results[name] = MetricResult(
                    name=name,
                    value=metric.worst_value,
                    category=metric.category,
                    error=str(e)
                )
        
        if not return_tensor:
            return results
            
        if normalize:
            logger.warning(
                "Normalization enabled. NOT recommended for MOBO as it distorts objective space."
            )
            return self._get_normalized_weighted_scores(
                results, 
                normalization_strategy,
                return_tensor=return_tensor
            )
        
        return self._get_raw_objectives(results, return_tensor=return_tensor)

    def evaluate_batch(
        self,
        eval_datasets: list[EvaluationDataset],
        return_tensor: bool = True,
        **kwargs
    ) -> list[dict[str, MetricResult]] | torch.Tensor:
        """
        Evaluate multiple datasets in parallel.
        
        Args:
            eval_datasets: Datasets to evaluate
            return_tensor: Return stacked tensor
            
        Returns:
            List of results or stacked tensor
        """
        if not eval_datasets:
            return torch.empty(0) if return_tensor else []

        futures: dict[int, Future] = {}
        for index, dataset in enumerate(eval_datasets):
            futures[index] = self._thread_executor.submit(
                self.evaluate, 
                dataset, 
                return_tensor=return_tensor,
                **kwargs
            )

        results: dict[int, dict[str, MetricResult] | torch.Tensor] = {
            index: future.result() for index, future in futures.items()
        }

        if return_tensor:
            return torch.stack([results[i] for i in range(len(eval_datasets))])
        return [results[i] for i in range(len(eval_datasets))]
    
    def _get_raw_objectives(
        self,
        results: dict[str, MetricResult],
        *,
        return_tensor: bool = True
    ) -> torch.Tensor | list[float]:
        """
        Get raw objective vector for MOBO.
        Returns values with negation applied (all "maximize").
        """
        values = []
        
        for name, result in results.items():
            metric = self._metrics[name]
            value = -result.value if metric.negate else result.value
            values.append(value)
        
        return torch.tensor(values, dtype=torch.float64) if return_tensor else values
    
    def _normalize_scores(
        self, 
        scores: list[float], 
        strategy: NormalizationStrategy = "sum"
    ) -> list[float]:
        """Normalize scores using specified strategy"""
        if not scores:
            return []
        
        if strategy == "sum":
            total = sum(scores)
            if total == 0:
                return [1.0 / len(scores)] * len(scores)
            return [w / total for w in scores]
        
        elif strategy == "softmax":
            import math
            max_w = max(scores)
            exp_scores = [math.exp(w - max_w) for w in scores]
            total = sum(exp_scores)
            return [ew / total for ew in exp_scores]
        
        elif strategy == "min-max":
            min_val, max_val = min(scores), max(scores)
            if max_val == min_val:
                return [1.0 / len(scores)] * len(scores)
            scaled = [(w - min_val) / (max_val - min_val) for w in scores]
            total = sum(scaled)
            return [s / total for s in scaled] if total > 0 else scaled
        
        elif strategy == "z-score":
            import statistics
            if len(scores) < 2:
                return scores
            mean = statistics.mean(scores)
            std = statistics.stdev(scores)
            if std == 0:
                return [1.0 / len(scores)] * len(scores)
            z_scores = [(w - mean) / std for w in scores]
            min_z = min(z_scores)
            shifted = [z - min_z for z in z_scores]
            total = sum(shifted)
            return [s / total for s in shifted] if total > 0 else [1.0 / len(scores)] * len(scores)
        
        raise ValueError(f"Unknown normalization strategy: {strategy}")
    
    def _get_normalized_weighted_scores(
        self, 
        results: dict[str, MetricResult],
        normalization_strategy: str = "sum",
        apply_weights: bool = True,
        *,
        return_tensor: bool = True
    ) -> torch.Tensor | list[float]:
        """
        Get weighted normalized scores for scalarization.
        WARNING: NOT for MOBO - only for single-objective or visualization.
        """
        values = []
        weights = []

        for name, result in results.items():
            metric = self._metrics[name]
            value = -result.value if metric.negate else result.value
            values.append(value)
            weights.append(self.objective_weights.get(name, 0.0))

        normalized_values = self._normalize_scores(values, strategy=normalization_strategy)

        if apply_weights:
            normalized_weights = self._normalize_scores(weights, strategy="sum")
            weighted_values = [v * w for v, w in zip(normalized_values, normalized_weights)]
        else:
            weighted_values = normalized_values

        return torch.tensor(weighted_values, dtype=torch.float64) if return_tensor else weighted_values
    
    def compute_objective_score(
        self,
        results: dict[str, MetricResult],
        normalization_strategy: NormalizationStrategy = "sum"
    ) -> float:
        """
        Compute single aggregated score for scalarization.
        NOTE: NOT used for MOBO - only for debugging/visualization.
        """
        values = []
        weights = []
        
        for name, result in results.items():
            metric = self._metrics[name]
            value = -result.value if metric.negate else result.value
            values.append(value)
            weights.append(self.objective_weights.get(name, 0.0))
        
        normalized_weights = self._normalize_scores(weights, strategy=normalization_strategy)
        return sum(v * w for v, w in zip(values, normalized_weights))
    
    def evaluate_retrieval(
        self,
        eval_dataset: EvaluationDataset,
        **kwargs
    ) -> dict[str, MetricResult]:
        """Evaluate only retrieval metrics"""
        return self.evaluate(
            eval_dataset, 
            metrics=self.retrieval_metrics,
            return_tensor=False,
            **kwargs
        )
    
    def evaluate_generation(
        self,
        eval_dataset: EvaluationDataset,
        **kwargs
    ) -> dict[str, MetricResult]:
        """Evaluate only generation metrics"""
        return self.evaluate(
            eval_dataset,
            metrics=self.generation_metrics,
            return_tensor=False,
            **kwargs
        )
    
    def evaluate_full(
        self,
        eval_dataset: EvaluationDataset,
        **kwargs
    ) -> dict[str, MetricResult]:
        """Evaluate full pipeline metrics"""
        return self.evaluate(
            eval_dataset,
            metrics=self.full_metrics,
            return_tensor=False,
            **kwargs
        )
    
    def available_metrics(self) -> list[str]:
        """List all available metric names"""
        return list(self.metric_names)