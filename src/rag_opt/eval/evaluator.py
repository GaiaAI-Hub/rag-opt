from rag_opt.eval.metrics import MetricResult, MetricCategory, BaseMetric
from concurrent.futures import Future, Executor
from rag_opt.dataset import EvaluationDataset
from typing import Optional, Literal
from rag_opt.llm import RAGLLM, RAGEmbedding
import rag_opt._utils as _utils
from loguru import logger
import torch


# Default weights for computing overall objective (only used for scalarization)
# NOTE:: order is important 
DEFAULT_WEIGHTS = {
    # Full
    "cost": 0.3,
    "latency": 0.2,
    
    # Generation
    "safety": 0.5,
    "alignment": 0.5,
    "response_relevancy": 0.5,
    
    # Retrieval
    "context_precision": 0.5,
    "context_recall": 0.3,
    "mrr": 0.3,
    "ndcg": 0.25
}


NormalizationStrategy = Literal["sum", "softmax", "min-max", "z-score"]


class _NormalizerMixin:
    """Mixin for weight normalization strategies"""
    
    def normalize(
        self, 
        scores: list[float], 
        strategy: NormalizationStrategy = "sum"
    ) -> list[float]:
        """Normalize scores to sum to 1.0"""

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
        
        elif strategy == "z-score":
            import statistics
            if len(scores) < 2:
                return scores
            mean = statistics.mean(scores)
            std = statistics.stdev(scores)
            if std == 0:
                n = len(scores)
                return [1.0 / n] * n
            z_scores = [(w - mean) / std for w in scores]
            min_z = min(z_scores)
            shifted = [z - min_z for z in z_scores]
            total = sum(shifted)
            return [s / total for s in shifted] if total > 0 else [1.0 / len(scores)] * len(scores)
        
        elif strategy == "min-max":
            min_val = min(scores)
            max_val = max(scores)
            if max_val == min_val:
                n = len(scores)
                return [1.0 / n] * n
            scaled = [(w - min_val) / (max_val - min_val) for w in scores]
            total = sum(scaled)
            return [s / total for s in scaled]
        raise ValueError(f"Unknown normalization strategy: {strategy}")


class RAGEvaluator(_NormalizerMixin):
    """
    Evaluator for RAG systems with support for multi-objective optimization.
    
    Handles retrieval, generation, and full pipeline metrics with configurable
    metrics for Pareto-optimal configuration search.
    
    For multi-objective Bayesian Optimization:
    - Use evaluate() with normalize=False and return_tensor=True
    - This returns raw objective values (with negation applied for minimize metrics)
    - Use ref_point property for hypervolume calculation
    """
    
    def __init__(
        self,
        evaluator_llm: RAGLLM,
        evaluator_embedding: Optional[RAGEmbedding] = None,
        metrics: Optional[list[BaseMetric]] = None,
        *,
        objective_weights: Optional[dict[str, float]] = None,
        auto_initialize_metrics: bool = True,
        executor: Optional[Executor] = None,
        **kwargs
    ):
        """
        Args:
            evaluator_llm: LLM instance for evaluation
            metrics: Custom metric instances to add
            objective_weights: Weight configuration per metric (only used for scalarization)
            auto_initialize_metrics: Whether to load default metrics
        """
        self.evaluator_llm = evaluator_llm
        self._metrics: dict[str, BaseMetric] = {}
        self.objective_weights: dict[str, float] = {}
        
        if auto_initialize_metrics:
            self._initialize_default_metrics(evaluator_llm,evaluator_embedding=evaluator_embedding, **kwargs)
        
        if metrics:
            self.add_metrics(metrics)
        
        self._initialize_weights(objective_weights or DEFAULT_WEIGHTS)
        self._thread_executor = executor or _utils.get_shared_executor()
    

    @property
    def ref_point(self) -> torch.Tensor:
        """
        Reference point for multi-objective optimization (worst case).
        
        The reference point must be in the TRANSFORMED space (after negation).
        It should be slightly worse than the worst achievable value.
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
        """Available metric names"""
        return set(self._metrics.keys())
    
    @property
    def retrieval_metrics(self) -> dict[str, BaseMetric]:
        """Metrics for retrieval evaluation"""
        return {
            name: metric for name, metric in self._metrics.items()
            if metric.category == MetricCategory.RETRIEVAL
        }
    
    @property
    def generation_metrics(self) -> dict[str, BaseMetric]:
        """Metrics for generation evaluation"""
        return {
            name: metric for name, metric in self._metrics.items()
            if metric.category == MetricCategory.GENERATION
        }
    
    @property
    def full_metrics(self) -> dict[str, BaseMetric]:
        """Full pipeline metrics (cost, latency)"""
        return {
            name: metric for name, metric in self._metrics.items()
            if metric.category == MetricCategory.FULL
        }
    
    def _initialize_default_metrics(self, llm: RAGLLM,evaluator_embedding: Optional[RAGEmbedding] = None, **kwargs) -> None:
        """Load all default metrics"""
        from rag_opt.eval import all_metrics_factory
        self.add_metrics(all_metrics_factory(llm,evaluator_embedding, **kwargs))
    
    def _initialize_weights(self, weights: dict[str, float]) -> None:
        """Initialize and validate objective weights (only used for scalarization)"""
        if not self._metrics:
            raise ValueError("Cannot initialize weights without metrics")
        
        for name, weight in weights.items():
            if name not in self.metric_names:
                logger.warning(f"Weight for unknown metric '{name}' will be ignored")
            else:
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
        Evaluate all or specified metrics on dataset.
        
        IMPORTANT: For multi-objective Bayesian Optimization, use:
        - normalize=False (default)
        - return_tensor=True
        This returns raw objectives with negation applied (all "maximize").
        
        Args:
            eval_dataset: Dataset to evaluate
            metrics: Optional specific metrics to evaluate (defaults to all)
            normalize: If True, apply normalization (NOT recommended for MOBO)
            return_tensor: Return tensor of metric values
            normalization_strategy: Strategy for normalizing (only if normalize=True)
            
        Returns:
            Dictionary of metric results or tensor of objective values
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
                "Normalization is enabled. This is NOT recommended for "
                "multi-objective Bayesian Optimization as it distorts the "
                "objective space. Use normalize=False for MOBO."
            )
            return self._get_normalized_weighted_scores(
                results, 
                normalization_strategy,
                return_tensor=return_tensor
            )
        else:
            return self._get_raw_objectives(results, return_tensor=return_tensor)

    def evaluate_batch(
        self,
        eval_datasets: list[EvaluationDataset],
        return_tensor: bool = True,
        **kwargs
    ) -> list[dict[str, MetricResult]] | torch.Tensor:
        """
        Evaluate multiple datasets in parallel while preserving input order.

        Args:
            eval_datasets: List of datasets to evaluate
            return_tensor: If True, return stacked tensor of objectives

        Returns:
            List of metric results or stacked tensor of objectives
        """
        futures: dict[int, Future] = {}

        for index, dataset in enumerate(eval_datasets):
            futures[index] = self._thread_executor.submit(
                self.evaluate, 
                dataset, 
                return_tensor=return_tensor,
                **kwargs
            )

        results: dict[int, dict[str, MetricResult] | torch.Tensor] = {}
        for index, future in futures.items():
            results[index] = future.result()

        if return_tensor:
            return torch.stack([results[i] for i in range(len(eval_datasets))]) if eval_datasets else torch.empty(0)
        return [results[i] for i in range(len(eval_datasets))]
    
    def _get_raw_objectives(
        self,
        results: dict[str, MetricResult],
        *,
        return_tensor: bool = True
    ) -> torch.Tensor | list[float]:
        """
        Get raw objective vector for multi-objective optimization.
        
        This is the PREFERRED method for Pareto front optimization.
        Returns metric values with negation applied so all objectives
        are "maximize", but NO normalization or weighting.
        
        Args:
            results: Metric evaluation results
            return_tensor: Return as tensor or list
            
        Returns:
            Tensor/list of objective values (negated for minimize metrics)
        """
        values = []
        
        for name, result in results.items():
            metric = self._metrics[name]
            value = result.value
            
            if metric.negate:
                value = -value
            
            values.append(value)
        
        return torch.tensor(values, dtype=torch.float64) if return_tensor else values
    
    def _get_normalized_weighted_scores(
        self, 
        results: dict[str, MetricResult],
        normalization_strategy: str = "sum",
        apply_weights: bool = True,
        *,
        return_tensor: bool = True
    ) -> torch.Tensor | list[float]:
        """
        Convert results to weighted, normalized tensor for scalarization.
        
        WARNING: Do NOT use this for multi-objective Bayesian Optimization.
        This is only for single-objective optimization or visualization.

        Process:
        1. Apply metric.negate for metrics that should be minimized
        2. Normalize metric values to make them comparable
        3. Optionally apply weights
        4. Normalize weights to sum to 1.0

        Args:
            results: Metric evaluation results
            normalization_strategy: How to normalize metric values
            apply_weights: Whether to multiply by normalized weights
            return_tensor: Return as tensor or list

        Returns:
            Tensor/list of weighted normalized metric values
        """
        values = []
        weights = []

        for name, result in results.items():
            metric = self._metrics[name]
            value = -result.value if metric.negate else result.value
            values.append(value)
            weights.append(self.objective_weights.get(name, 0.0))

        # Normalize metric values
        normalized_values = self.normalize(values, strategy=normalization_strategy)

        if apply_weights:
            normalized_weights = self.normalize(weights, strategy="sum")
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
        Compute single aggregated objective score for scalarization.
        
        NOTE: This is NOT used for multi-objective Bayesian Optimization.
        It's only for single-objective optimization or debugging.
        
        Args:
            results: Metric evaluation results
            normalization_strategy: Weight normalization strategy
            
        Returns:
            Weighted aggregate score (higher is better)
        """
        values = []
        weights = []
        
        for name, result in results.items():
            metric = self._metrics[name]
            value = result.value
            
            if metric.negate:
                value = -value
            
            values.append(value)
            weights.append(self.objective_weights.get(name, 0.0))
        
        normalized_weights = self.normalize(weights, strategy=normalization_strategy)
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
        """Evaluate full pipeline metrics (cost, latency)"""
        return self.evaluate(
            eval_dataset,
            metrics=self.full_metrics,
            return_tensor=False,
            **kwargs
        )
    
    def available_metrics(self) -> list[str]:
        """List all available metric names"""
        return list(self.metric_names)