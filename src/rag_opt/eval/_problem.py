from rag_opt.eval.evaluator import RAGEvaluator
from rag_opt._manager import RAGPipelineManager
from rag_opt.dataset import EvaluationDataset
from fastmobo.problems import FastMoboProblem
from typing_extensions import Annotated, Doc
from rag_opt._sampler import SamplerType
from rag_opt.dataset import TrainDataset
from rag_opt._config import RAGConfig
from functools import cached_property
from langchain.schema import Document
from rag_opt.llm import RAGLLM, RAGEmbedding
from loguru import logger
import torch
import time

class RAGOptimizationProblem:
    """
    Sequential multi-objective optimization problem (simplified)
    """
    
    def __init__(
        self,
        train_dataset: Annotated[TrainDataset, Doc("Dataset for ground truth and querying")],
        rag_pipeline_manager: Annotated[RAGPipelineManager, Doc("Service for sampling and creating RAG instances")], 
        evaluator_llm: Annotated[RAGLLM | str, Doc("LLM for metric evaluation")]=None,
        *, 
        evaluator_embedding: Annotated[RAGEmbedding | str, Doc("Embedding for metric evaluation")]=None,
        evaluator: Annotated[RAGEvaluator, Doc("Evaluator for response metrics")]=None,
    ):
        if not evaluator_llm and not evaluator:
            logger.error("Must provide either evaluator or evaluator_llm")
            raise ValueError("Must provide either evaluator or evaluator_llm")
        
        self.evaluator = evaluator or RAGEvaluator(
            evaluator_llm=evaluator_llm, 
            evaluator_embedding=evaluator_embedding,
            batch_timeout=600
        )
        self.rag_pipeline_manager = rag_pipeline_manager
        self.train_dataset = train_dataset

        self.problem: FastMoboProblem = None
    
    def generate_initial_data(self, n_samples: int=20, **kwargs) -> tuple[list[RAGConfig], list[EvaluationDataset]]:
        """Prepare train_x, train_y for optimization"""
        return self.rag_pipeline_manager.generate_initial_data(
            train_data=self.train_dataset,
            n_samples=n_samples,
            sampler_type=SamplerType.SOBOL,
            **kwargs
        )

    @property
    def documents(self) -> list[Document]:
        """List of dataset documents for indexing"""
        return self.train_dataset.to_langchain_docs()
    
    @property
    def bounds(self) -> torch.Tensor:
        """Get bounds tensor from search space"""
        return self.rag_pipeline_manager.get_problem_bounds()
    
    @cached_property
    def ref_point(self) -> torch.Tensor:
        """Reference point for hypervolume calculation"""
        ref_values = self.evaluator.ref_point
        return ref_values.detach().clone().double()
    
    @cached_property
    def objectives(self) -> list[str]:
        return list(self.evaluator.metric_names)

    def objective_function(self, x: torch.Tensor) -> torch.Tensor:
        """
         multi-objective function
        
        Args:
            x: [batch_size, num_hyperparams] 
        Returns: 
            [batch_size, num_objectives]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        logger.info(f"Evaluating batch of {batch_size} configurations ")
        start_time = time.time()
        
        objectives_list = []
        succeeded = 0
        failed = 0
        
        for i in range(batch_size):
            try:
                logger.debug(f"Evaluating config {i+1}/{batch_size}")
                config_start = time.time()
                
                objectives = self._evaluate_single_config(x[i])
                objectives_list.append(objectives)
                succeeded += 1
                
                config_time = time.time() - config_start
                logger.debug(f"Config {i+1} completed in {config_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Config {i+1} evaluation failed: {e}")
                objectives_list.append(self._get_worst_objectives())
                failed += 1
        
        elapsed = time.time() - start_time
        logger.info(
            f"Batch evaluation completed: {succeeded} succeeded, {failed} failed "
            f"in {elapsed:.2f}s ({elapsed/batch_size:.2f}s per config)"
        )
        
        return torch.stack(objectives_list)
    
    def _evaluate_single_config(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Evaluate single configuration
        """
        # Decode sample to RAG config
        rag_config = self.rag_pipeline_manager.decode_sample_to_rag_config(sample=sample)
        
        # Generate evaluation data
        eval_dataset = self.rag_pipeline_manager.generate_evaluation_data(
            config=rag_config,
            train_data=self.train_dataset
        )
        
        # Evaluate metrics
        objectives = self.evaluator.evaluate(eval_dataset, return_tensor=True) 
        
        return objectives
    
    def _get_worst_objectives(self) -> torch.Tensor:
        """
        Return worst possible objective values for failed evaluations
        """
        worst_values = []
        for metric in self.evaluator._metrics.values():
            if metric.negate:
                # For negated metrics (minimize), worst is largest negative
                worst = -metric.worst_value if metric.worst_value > 0 else -1.0
            else:
                # For maximization, worst is smallest value
                worst = metric.worst_value if metric.worst_value is not None else 0.0
            worst_values.append(worst)
        
        return torch.tensor(worst_values, dtype=torch.float64)
        
    def create_fastmobo_problem(self, device: torch.device=None) -> FastMoboProblem:
        """Create FastMoBo problem instance"""
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        noise_std = torch.tensor([0.1] * len(self.objectives), dtype=torch.float, device=device)
        bounds = self.bounds.to(device)
        
        self.problem = FastMoboProblem(
            objective_func=self.objective_function,
            bounds=bounds,
            ref_point=self.ref_point,
            num_objectives=len(self.objectives),
            noise_std=noise_std,
            negate=False  # We do negation internally per metric
        )
        
        return self.problem